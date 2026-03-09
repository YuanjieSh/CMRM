import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# def focal_loss(input_values, gamma):
#     """Computes the focal loss"""
#     p = torch.exp(-input_values)
#     loss = (1 - p) ** gamma * input_values
#     return loss.mean()

# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, gamma=0.):
#         super(FocalLoss, self).__init__()
#         assert gamma >= 0
#         self.gamma = gamma
#         self.weight = weight

#     def forward(self, input, target):
#         return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0., reduction='mean'):
        super().__init__()
        # assert gamma >= 0, "gamma must be non-negative"
        # assert reduction in ('none', 'mean', 'sum')
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_per_sample = F.cross_entropy(
            input, target,
            weight=self.weight,
            reduction='none'
        )                         

        focal_per_sample = focal_loss(ce_per_sample, self.gamma)  # [N]

        # 3) reduction
        if self.reduction == 'none':
            return focal_per_sample
        elif self.reduction == 'sum':
            return focal_per_sample.sum()
        else:  # 'mean'
            return focal_per_sample.mean()

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, reduction='mean'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.reduction = reduction  

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight, reduction=self.reduction)


# class CVaR_LDAMLoss(nn.Module):
#     def __init__(self, cls_num_list, alpha=0.9, max_m=0.5, weight=None, s=30):
#         super(CVaR_LDAMLoss, self).__init__()
#         m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
#         m_list = m_list * (max_m / np.max(m_list))
#         self.m_list = torch.FloatTensor(m_list).cuda()
#         self.s = s
#         self.weight = weight
#         self.alpha = alpha

#     def forward(self, x, target):
#         index = torch.zeros_like(x, dtype=torch.uint8)
#         index.scatter_(1, target.data.view(-1, 1), 1)
#         index_float = index.type(torch.cuda.FloatTensor)

#         # Apply LDAM margin
#         batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
#         batch_m = batch_m.view((-1, 1))
#         x_m = x - batch_m
#         output = torch.where(index, x_m, x)

#         # LDAM loss per sample
#         per_sample_loss = F.cross_entropy(self.s * output, target, weight=self.weight, reduction='none')

#         # CVaR loss: differentiable upper bound of worst α-tail
#         with torch.no_grad():
#             tau = torch.quantile(per_sample_loss, self.alpha)

#         cvar_loss = tau + (1.0 / self.alpha) * torch.mean(torch.clamp(per_sample_loss - tau, min=0.0))
#         return cvar_loss


# class CVaRRegularizedLoss(nn.Module):
#     def __init__(self, base_loss_fn, alpha=0.9, reg=0.1):
#         super(CVaRRegularizedLoss, self).__init__()
#         self.base_loss_fn = base_loss_fn
#         self.alpha = alpha
#         self.reg = reg

#     def forward(self, outputs, targets):
#         per_sample_loss = self.base_loss_fn(outputs, targets)
#         base_loss = per_sample_loss.mean()

#         with torch.no_grad():
#             tau = torch.quantile(per_sample_loss, self.alpha)

#         cvar_reg = tau + (1.0 / self.alpha) * torch.mean(torch.clamp(per_sample_loss - tau, min=0.0))
#         return base_loss + self.reg * cvar_reg

# class MarginRegularizedLoss(nn.Module):
#     def __init__(self, base_loss_fn, alpha=0.9, reg=0.1):
#         super().__init__()
#         if not (0 < alpha < 1):
#             raise ValueError("alpha must be in (0,1)")
#         self.base_loss_fn = base_loss_fn      # reduction='none'
#         self.alpha = alpha
#         self.reg = reg

#     def forward(self, outputs, targets):
#         loss_i = self.base_loss_fn(outputs, targets)      # shape [N]
#         base_loss = loss_i.mean()

#         # VaR (τ)
#         with torch.no_grad():
#             tau = torch.quantile(loss_i, self.alpha)

#         # CVaR term
#         tail = torch.clamp(tau - loss_i, min=0.0)
#         margin = tau + tail.mean() / (1.0 - self.alpha)

#         return base_loss + self.reg * margin


class MarginRegularizedLoss(nn.Module):
    def __init__(self, base_loss_fn, alpha=0.9, reg=0.1, temp=1.0):
        super().__init__()
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0,1)")
        self.base_loss_fn = base_loss_fn     # reduction='none'
        self.alpha = alpha
        self.reg = reg
        self.temp = temp

    def forward(self, outputs, targets):
        loss_i = self.base_loss_fn(outputs, targets)      
        base_loss = loss_i.mean()

        with torch.no_grad():
            tau = torch.quantile(loss_i, self.alpha)      

        weights = torch.sigmoid(-(loss_i - tau) / self.temp)

        margin_loss = (weights * loss_i).sum() / (weights.sum() + 1e-8)

        margin_loss_final = self.reg * margin_loss

        return base_loss, margin_loss_final, base_loss + self.reg * margin_loss

# class MargindRegularizedLoss_2(nn.Module):
#     def __init__(self, base_loss_fn, alpha=0.1, reg=0.1, temp=1.0):
#         super().__init__()
#         self.base_loss_fn = base_loss_fn   
#         self.alpha = alpha
#         self.reg = reg
#         self.temp = temp

#     def _compute_margins(logits: torch.Tensor,
#                          targets: torch.Tensor) -> torch.Tensor:
#         """f(x)_y - max_{y'≠y} f(x)_{y'}  (no softmax needed)."""
#         true_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
#         tmp = logits.clone()
#         tmp.scatter_(1, targets.unsqueeze(1), float("-inf"))
#         max_other_logits, _ = tmp.max(dim=1)
#         return true_logits - max_other_logits

#     def forward(self, outputs, targets):
#         loss_i = self.base_loss_fn(outputs, targets)   # [N]
#         base_loss = loss_i.mean()
        
#         margins = self._compute_margins(outputs, targets)  

#         tau = torch.quantile(margins.detach(), self.alpha)

#         weights = torch.sigmoid((margins - tau) / self.temp)

#         margin_loss = (weights * margins).sum() / (weights.sum() + 1e-8)

#         return base_loss - self.reg * margin_loss


# class MarginRegularizedLoss_2(nn.Module):
#     def __init__(self,
#                  base_loss_fn: nn.Module | None = None,
#                  alpha: float = 0.9,
#                  reg: float = 0.1,
#                  temp: float = 1.0):
#         super().__init__()
#         assert 0. < alpha < 1., "`alpha` must be in (0,1)"
#         self.alpha = alpha
#         self.reg   = reg          
#         self.temp  = temp         

#         if base_loss_fn is None:
#             base_loss_fn = nn.CrossEntropyLoss(reduction='none')
#         else:
#             assert getattr(base_loss_fn, 'reduction', 'none') == 'none', \
#                 "base_loss_fn must output per‑sample losses (reduction='none')"
#         self.base_loss_fn = base_loss_fn

#     @staticmethod
#     def _compute_margins(logits: torch.Tensor,
#                          targets: torch.Tensor) -> torch.Tensor:

#         true_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
#         tmp = logits.clone()
#         tmp.scatter_(1, targets.unsqueeze(1), float('-inf'))
#         max_other_logits, _ = tmp.max(dim=1)
#         return true_logits - max_other_logits  

#     def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
#         loss_i = self.base_loss_fn(outputs, targets)  # [N]
#         base_loss = loss_i.mean()

#         margins = self._compute_margins(outputs, targets)  # [N]

#         tau = torch.quantile(margins.detach(), 1 - self.alpha)

#         weights = torch.sigmoid((margins - tau) / self.temp)  # ∈(0,1)

#         margin_loss = (weights * margins).sum() / (weights.sum() + 1e-8)

#         return base_loss - self.reg * margin_loss


class MarginRegularizedLoss_2(nn.Module):
    def __init__(self,
                 base_loss_fn: nn.Module | None = None,
                 alpha: float = 0.9,
                 reg: float = 0.1,
                 temp: float = 1.0):
        super().__init__()
        assert 0. < alpha < 1., "`alpha` must be in (0,1)"
        self.alpha = alpha
        self.reg   = reg          
        self.temp  = temp         
        self.last_masked_indices = []  

        if base_loss_fn is None:
            base_loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            assert getattr(base_loss_fn, 'reduction', 'none') == 'none', \
                "base_loss_fn must output per‑sample losses (reduction='none')"
        self.base_loss_fn = base_loss_fn

    @staticmethod
    def _compute_margins(logits: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:

        true_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        tmp = logits.clone()
        tmp.scatter_(1, targets.unsqueeze(1), float('-inf'))
        max_other_logits, _ = tmp.max(dim=1)
        return true_logits - max_other_logits  

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        loss_i = self.base_loss_fn(outputs, targets)  # [N]
        base_loss = loss_i.mean()

        margins = self._compute_margins(outputs, targets)  # [N]
        tau = torch.quantile(margins.detach(), 1 - self.alpha)

        weights = torch.sigmoid((margins - tau) / self.temp)  # ∈(0,1)
        margin_loss = (weights * margins).sum() / (weights.sum() + 1e-8)

        mask = margins.detach() < tau
        self.last_masked_indices = mask.nonzero(as_tuple=False).squeeze(1).detach().cpu().tolist()

        margin_loss_final = - self.reg * margin_loss

        return base_loss, margin_loss_final, base_loss - self.reg * margin_loss

    def get_last_masked_indices(self):
        """
        Return the indices of samples masked out (i.e., margin < tau)
        from the most recent forward pass.
        """
        return self.last_masked_indices

class MarginLoss(nn.Module):
    def __init__(self, base_loss_fn, alpha=0.9, temp=1.0):
        super().__init__()
        self.base_loss_fn = base_loss_fn     # reduction='none'
        self.alpha = alpha
        self.temp = temp

    def forward(self, outputs, targets):
        loss_i = self.base_loss_fn(outputs, targets)

        with torch.no_grad():
            tau = torch.quantile(loss_i, self.alpha)

        weights = torch.sigmoid(-(loss_i - tau) / self.temp)

        margin_loss = (weights * loss_i).sum() / (weights.sum() + 1e-8)
        return margin_loss



