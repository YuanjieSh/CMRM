# python libraries
import argparse
import time
import warnings

import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms, models

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

import os
from pathlib import Path

import pandas as pd

torch.set_default_dtype(torch.float64)
# local files
# from model.simclr_model import *


# Define command-line arguments
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Train a good baseline model on Noisy CIFAR-10/100 dataset')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Choose the dataset')
parser.add_argument('--noise_mode', type=str, default='clean_label',
                    choices=['clean_label', 'aggre_label', 'rand_1_label', 'rand_2_label', 'rand_3_label', 'worst_label', 'symmetric_flip_label', 'promix_100_label',
                             'promix_400_label'],
                    help='Noise mode for labels: random or human')
parser.add_argument('--symmetric_flip_prob', type=float, default=None, help='Probability of symmetric label flipping')
parser.add_argument('--feature_type', type=str, default='original',
                    choices=['original', 'transfer_learning', 'contrastive_learning', 'foundation_model'],
                    help='feature type for training linear model')
parser.add_argument('--encoder_name', type=str, default=None, choices=[None, 'resnet18', 'resnet34', 'resnet50', 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'])

# parser.add_argument('--num_epochs', type=int, default=20,
#                     help='Number of epochs/iterations for training')
parser.add_argument("--batch_size", default=64, type=int, help="Batch size used during feature extraction.")

# add parser: optimizer - sgd or adam or sgd with lr scheduler
# parser.add_argument('--optimizer', type=str, default='lbfgs', choices=['lbfgs'])
parser.add_argument('--cache_mode', type=str, default='off', choices=['off', 'save', 'load'],
                    help="off: not using cache; save: save cache; load: load the saved cache")
parser.add_argument('--feat_cache', type=str, default=None,
                    help="Path for cache saving(.npz), including X_train, y_train, X_test, y_test")
parser.add_argument('--encoder_ckpt', type=str, default=None,
                    help="save or load the encoder")
# parser.add_argument('--C', type=float, default=10, choices=['off', 'save', 'load'],
#                     help="hyerparameter for l2")
parser.add_argument('--max_iter', type=int, default=100,
                    help="hyerparameter for total iteration")
parser.add_argument('--alpha', default=0.1, type=float,
                    metavar='alpha', help='alpha for cr2m', dest='alpha')
parser.add_argument('--reg', default=0.1, type=float,
                    metavar='regular', help='regularization for cr2m', dest='reg')

args = parser.parse_args()


def save_features_npz(path, X_train, X_test, y_test):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def _to_np(x):
        import numpy as np, torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x
    np.savez_compressed(path,
                        X_train=_to_np(X_train),
                        X_test=_to_np(X_test),
                        y_test=_to_np(y_test))

def load_features_npz(path):
    data = np.load(path, allow_pickle=False)
    return data['X_train'], data['X_test'], data['y_test']

class MultinomialLogReg(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=True)
    def forward(self, x):
        return self.fc(x)

def numpy_to_tensor(X, y, *, device="cpu", float_dtype=torch.float64):
    # X
    if isinstance(X, np.ndarray):
        Xt = torch.from_numpy(X)
    else:
        Xt = X  # already tensor
    Xt = Xt.to(device=device, dtype=float_dtype)

    if isinstance(y, np.ndarray):
        yt = torch.from_numpy(y)
    else:
        yt = y
    yt = yt.to(device=device, dtype=torch.long)

    return Xt, yt

def logistic_objective(model, X, y, C):
    logits = model(X)
    ce = F.cross_entropy(logits, y, reduction='mean')  
    W = model.fc.weight
    l2 = 0.5 * (1.0 / C) * (W * W).sum()
    return ce + l2

def _compute_margins(logits: torch.Tensor,
                    targets: torch.Tensor) -> torch.Tensor:

    true_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    tmp = logits.clone()
    tmp.scatter_(1, targets.unsqueeze(1), float('-inf'))
    max_other_logits, _ = tmp.max(dim=1)
    return true_logits - max_other_logits  

def CR2M_objective(model, X, y, C,  alpha=0.9, reg=1.0):
    logits = model(X)
    margins =_compute_margins(logits, y)

    tau = torch.quantile(margins.detach(), 1 - alpha)

    weights = torch.sigmoid((margins - tau) / 1.0)  # ∈(0,1)
    margin_loss = (weights * margins).sum() / (weights.sum() + 1e-8)
    margin_loss_final = - reg * margin_loss

    loss_i = F.cross_entropy(logits, y, reduction='none') 
    base_loss = loss_i.mean()

    W = model.fc.weight
    l2 = 0.5 * (1.0 / C) * (W * W).sum()

    return base_loss + l2 + margin_loss_final

# def _logit_margin(logits, y):
#     # m = z_y - max_{k≠y} z_k
#     zy = logits.gather(1, y.view(-1,1)).squeeze(1)
#     tmp = logits.clone()
#     tmp.scatter_(1, y.view(-1,1), float('-inf'))
#     zmax_other = tmp.max(dim=1).values
#     return zy - zmax_other

# def CR2M_objective(
#     model, X, y, C,
#     alpha=0.05,          
#     lam_margin=1.0,      
#     kappa=20.0,               
#     detach_tau=True,
#     eps=1e-8
# ):
#     logits = model(X)

#     margins = _logit_margin(logits, y)                  # [N]
#     tau = torch.quantile(margins.detach() if detach_tau else margins, 1 - alpha)

#     w = torch.sigmoid(kappa * (margins - tau))          # [N], in (0,1)
#     w_sum = w.sum()

#     ce_i = F.cross_entropy(logits, y, reduction='none')
#     ce = ce_i.mean()                            

#     if w_sum > 0:
#         reward_margin = (w * margins).sum() / (w_sum + eps)
#     else:
#         reward_margin = margins.mean()                  

#     W = model.fc.weight
#     l2 = 0.5 * (1.0 / C) * (W * W).sum()

#     loss = ce + l2 - lam_margin * reward_margin
#     return loss

@torch.no_grad()
def accuracy(model, X, y, batch_size=4096, device='cpu'):
    model.eval()
    n = X.size(0)
    correct = 0
    for i in range(0, n, batch_size):
        xb = X[i:i+batch_size].to(device)
        yb = y[i:i+batch_size].to(device)
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
    return correct / n

# def fit_lbfgs(model, X, y, C, max_iter=300, tol=1e-6, device='cpu', two_stage=True):
#     model = model.to(device)

#     def make_opt(lr, max_iter):
#         return torch.optim.LBFGS(
#             model.parameters(),
#             lr=lr,
#             max_iter=max_iter,
#             tolerance_grad=min(tol, 1e-7),
#             tolerance_change=min(tol, 1e-9),
#             history_size=100,
#             line_search_fn='strong_wolfe'
#         )

#     def closure():
#         opt.zero_grad(set_to_none=True)
#         loss = logistic_objective(model, X, y, C)
#         loss.backward()
#         return loss

#     opt = make_opt(lr=1.0, max_iter=max_iter)
#     opt.step(closure)

#     if two_stage:
#         opt = make_opt(lr=0.5, max_iter=60)
#         opt.step(closure)

#     with torch.no_grad():
#         final_loss = logistic_objective(model, X, y, C).item()
#     return final_loss

def fit_lbfgs(model, X, y, C, max_iter=100, tol=1e-6, device='cpu',
              two_stage=True, alpha=0.1, reg=1.0):
    model = model.to(device)

    def make_opt(lr, max_iter):
        return torch.optim.LBFGS(
            model.parameters(),
            lr=lr,
            max_iter=max_iter,
            tolerance_grad=min(tol, 1e-7),
            tolerance_change=min(tol, 1e-9),
            history_size=100,
            line_search_fn='strong_wolfe'
        )

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = CR2M_objective(model, X, y, C, alpha=alpha, reg=reg)
        loss.backward()
        return loss

    opt = make_opt(lr=0.5, max_iter=max_iter)
    opt.step(closure)

    if two_stage:
        opt = make_opt(lr=0.25, max_iter=60)
        opt.step(closure)

    with torch.no_grad():
        final_loss = CR2M_objective(model, X, y, C, alpha=alpha, reg=reg).item()
    return final_loss


def torch_grid_search_lbfgs(
    X_train, y_train, X_test, y_test,
    param_grid=None, cv=5, tol=1e-6, device=None, float_dtype=torch.float64, standardize=True
):
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    X_train_np = X_train if isinstance(X_train, np.ndarray) else X_train.cpu().numpy()
    y_train_np = y_train if isinstance(y_train, np.ndarray) else y_train.cpu().numpy()
    X_test_np  = X_test  if isinstance(X_test,  np.ndarray) else X_test.cpu().numpy()
    y_test_np  = y_test  if isinstance(y_test,  np.ndarray) else y_test.cpu().numpy()

    in_dim = X_train_np.shape[1]
    num_classes = int(np.max(y_train_np) + 1)

    if param_grid is None:
        param_grid = {
            'C': [10],
            'max_iter': [100],
            'alpha': [0.99, 0.95, 0.92, 0.9, 0.87, 0.85],
            'reg': [0.01, 0.02, 0.05, 0.07, 0.1],
        }

    from itertools import product
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    skf = StratifiedKFold(n_splits=cv, shuffle=False)
    best_score = -1.0
    best_params = None

    def fit_standardizer(X):
        mean = X.mean(axis=0, keepdims=True)
        std  = X.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        return mean, std

    def apply_standardizer(X, mean, std):
        return (X - mean) / std

    for combo in combos:
        cfg = dict(zip(keys, combo))
        C = float(cfg['C'])
        max_iter = int(cfg['max_iter'])
        alpha = float(cfg['alpha'])
        reg = float(cfg['reg'])

        val_scores = []

        for tr_idx, va_idx in skf.split(X_train_np, y_train_np):
            X_tr_np, y_tr_np = X_train_np[tr_idx], y_train_np[tr_idx]
            X_va_np, y_va_np = X_train_np[va_idx], y_train_np[va_idx]

            if standardize:
                m, s = fit_standardizer(X_tr_np)
                X_tr_np_std = apply_standardizer(X_tr_np, m, s)
                X_va_np_std = apply_standardizer(X_va_np, m, s)
            else:
                X_tr_np_std, X_va_np_std = X_tr_np, X_va_np

            X_tr, y_tr = numpy_to_tensor(X_tr_np_std, y_tr_np, device=device, float_dtype=float_dtype)
            X_va, y_va = numpy_to_tensor(X_va_np_std, y_va_np, device=device, float_dtype=float_dtype)

            model = MultinomialLogReg(in_dim, num_classes).to(device).to(float_dtype)

            _ = fit_lbfgs(model, X_tr, y_tr, C=C, max_iter=max_iter, tol=tol,
                          device=device, two_stage=True, alpha=alpha, reg=reg)

            acc = accuracy(model, X_va, y_va, device=device)
            val_scores.append(acc)

        avg_val = float(np.mean(val_scores))
        print(f"[Grid] {cfg} | val_acc={avg_val*100:.2f}%")
        if avg_val > best_score:
            best_score = avg_val
            best_params = cfg

    print("\nBest parameters based on CV (noisy data):", best_params)
    print(f"Best CV Acc: {best_score*100:.2f}%\n")

    C = float(best_params['C']); max_iter = int(best_params['max_iter'])
    alpha = float(best_params['alpha']); reg = float(best_params['reg'])

    if standardize:
        mean_full, std_full = fit_standardizer(X_train_np)
        X_train_np_std = apply_standardizer(X_train_np, mean_full, std_full)
        X_test_np_std  = apply_standardizer(X_test_np,  mean_full, std_full)
    else:
        X_train_np_std, X_test_np_std = X_train_np, X_test_np
        mean_full = std_full = None  

    X_train_t, y_train_t = numpy_to_tensor(X_train_np_std, y_train_np, device=device, float_dtype=float_dtype)
    X_test_t,  y_test_t  = numpy_to_tensor(X_test_np_std,  y_test_np,  device=device, float_dtype=float_dtype)

    final_model = MultinomialLogReg(in_dim, num_classes).to(device).to(float_dtype)
    _ = fit_lbfgs(final_model, X_train_t, y_train_t, C=C, max_iter=max_iter, tol=tol,
                  device=device, two_stage=True, alpha=alpha, reg=reg)

    test_acc = accuracy(final_model, X_test_t, y_test_t, device=device)
    print(f"Testing Accuracy (torch eval): {test_acc * 100:.2f}%")

    with torch.no_grad():
        y_pred = final_model(X_test_t).argmax(1).cpu().numpy()
    print(f"Testing Accuracy (sklearn metrics): {metrics.accuracy_score(y_test_np, y_pred) * 100:.2f}%\n")

    final_scaler = {'mean': mean_full, 'std': std_full} if standardize else None
    return final_model, best_params

def main():

    warnings.filterwarnings('ignore') # turn off the warnings (especially for sklearn convergence)

    print()
    print(f"================ {args.dataset} with {args.noise_mode} (symmetric flip prob = {args.symmetric_flip_prob}) ========================")
    # print()
    print(f"Linear model + feature by {args.feature_type}, encoder: {args.encoder_name}")
    # print(f"num_epochs: {args.num_epochs}, optimizer: {args.optimizer}, batch_size: {args.batch_size}")
    print()

    if args.feature_type == 'original' or args.feature_type == 'contrastive_learning': # in these cases, not need to upscale cifar images
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.feature_type == 'transfer_learning' or args.feature_type == 'foundation_model': # need to upscale, cuz pre-trained on ImageNet
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    # Load CIFAR-10 dataset with noisy labels
    dataset_path = './data'

    # Load CIFAR-10 dataset
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transform)

    # Check if CUDA is available and set PyTorch to use GPU or CPU
    # Move model to GPU if available
    try:
        if torch.cuda.is_available():
            print()
            print(torch.cuda.get_device_name(torch.cuda.current_device()))
            device = torch.device("cuda:0")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    except AttributeError:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    print("Using device:", device)
    print()

    start_time = time.time()
    # extract the features and get the training and testing data
    if args.feature_type == 'original': # directly get all the data
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

        # Extract the data and labels
        X_train, _ = next(iter(trainloader))    
        X_test, y_test = next(iter(testloader))

        X_train = X_train.view(X_train.size(0), -1).numpy()
        X_test = X_test.view(X_test.size(0), -1).numpy()
        y_test = y_test.numpy()
    else: # extract the features in a batch manner (otherwise, may run out of memory)

        # Load the  pre-trained feature extractor
        if args.feature_type == 'foundation_model':
            feature_extractor = torch.hub.load('facebookresearch/dinov2', args.encoder_name)
        elif args.feature_type == 'transfer_learning':
            if args.encoder_name == 'resnet18':
                feature_extractor = models.resnet18(pretrained=True)
            elif args.encoder_name == 'resnet34':
                feature_extractor = models.resnet34(pretrained=True)
            elif args.encoder_name == 'resnet50':
                feature_extractor = models.resnet50(pretrained=True)
            feature_extractor.fc = nn.Identity() # Replace the classification layer with an identity function
        elif args.feature_type == 'contrastive_learning':
            pretrained_model = torch.load(f'trained model/ckpt_{args.dataset}_{args.encoder_name}.pth')
            sd = {}
            for ke in pretrained_model['model']:
                nk = ke.replace('module.', '')
                sd[nk] = pretrained_model['model'][ke]
            feature_extractor = Encoder_cl(name=args.encoder_name)
            feature_extractor.load_state_dict(sd, strict=False)

        feature_extractor.to(device)
        feature_extractor.eval()

        cache_dir = Path("feature")
        cache_dir.mkdir(exist_ok=True)
        args.feat_cache = cache_dir / f"{args.dataset}_{args.noise_mode}_feature.npz"


        if args.cache_mode == 'load' and args.feat_cache and Path(args.feat_cache).exists():
            print(f"[Feature Cache] loading from {args.feat_cache}")
            X_train, X_test, y_test = load_features_npz(args.feat_cache)
        else:

            # Extract the features
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
            testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

            X_train = []
            X_test, y_test = [], []

            # Extract features from training data (training labels are noisy, load later)
            for i, (inputs, _) in enumerate(trainloader):
                inputs = inputs.to(device)
                with torch.no_grad():
                    features = feature_extractor(inputs)
                    features = torch.flatten(features, start_dim=1).cpu().numpy()  # Flatten the features
                X_train.append(features)
                if (i+1) % 50 == 0:  # Check if (i+1) is divisible by 10
                    print(f'Batch {i+1}/{len(trainloader)} of train data processed.')

            # Extract features and labels from test data
            for i, (inputs, labels) in enumerate(testloader):
                inputs = inputs.to(device)
                with torch.no_grad():
                    features = feature_extractor(inputs)
                    features = torch.flatten(features, start_dim=1).cpu().numpy()  # Flatten the features
                X_test.append(features)
                y_test.append(labels.numpy())
                if (i+1) % 50 == 0:  # Check if (i+1) is divisible by 10
                    print(f'Batch {i+1}/{len(testloader)} of test data processed.')

            X_train = np.concatenate(X_train, axis=0)
            X_test = np.concatenate(X_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)

            if args.cache_mode == 'save' and args.feat_cache:
                print(f"[Feature Cache] saving to {args.feat_cache}")
                save_features_npz(args.feat_cache, X_train, X_test, y_test)


    # load noisy training labels
    if args.dataset == 'cifar10':
        noise_file_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        noise_file_path = './data/CIFAR-100_human.pt'

    noise_file = torch.load(noise_file_path)

    if args.noise_mode == 'clean_label':
        y_train = noise_file['clean_label']
    elif args.noise_mode == 'aggre_label':
        y_train = noise_file['aggre_label']
    elif args.noise_mode == 'rand_1_label':
        y_train = noise_file['random_label1']
    elif args.noise_mode == 'rand_2_label':
        y_train = noise_file['random_label2']
    elif args.noise_mode == 'rand_3_label':
        y_train = noise_file['random_label3']
    elif args.noise_mode == 'worst_label':
        if args.dataset == 'cifar10':
            y_train = noise_file['worse_label']
        elif args.dataset == 'cifar100':
            y_train = noise_file['noisy_label']
    elif args.noise_mode == 'symmetric_flip_label':
        y_train = torch.load(f'{dataset_path}/CIFAR-10_symmetric_{args.symmetric_flip_prob}.pt')

    
    param_grid = {
    'C': [10],
    'max_iter': [100],
    'alpha': [0.99, 0.95, 0.92, 0.9, 0.87, 0.85],
    'reg': [0.01, 0.02, 0.05, 0.07, 0.1],
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, best_params = torch_grid_search_lbfgs(
        X_train, y_train, X_test, y_test,
        param_grid=param_grid,
        cv=5,            
        tol=1e-6,        
        device=device
    )

    print()
    print("Best parameters based on cross validation on noisy data:", best_params)
    print()

    param = next(model.parameters())
    float_dtype = param.dtype
    dev = param.device

    with torch.no_grad():
        if isinstance(X_test, np.ndarray):
            Xt = torch.from_numpy(X_test).to(device=dev, dtype=float_dtype)
        else:
            Xt = X_test.to(device=dev, dtype=float_dtype)

        # Xt = torch.from_numpy(X_test).float().to(device) if isinstance(X_test, np.ndarray) else X_test.to(device).float()
        logits = model(Xt)
        y_pred = logits.argmax(1).cpu().numpy()
        probs  = F.softmax(logits, dim=1).cpu().numpy() 

    y_test_np = y_test.cpu().numpy() if torch.is_tensor(y_test) else np.asarray(y_test)

    print(f"Testing Accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")
    print()

    end_time = time.time()
    print('Total training time: {:.2f} seconds'.format(end_time - start_time))

    os.makedirs("checkpoint", exist_ok=True)
    save_path = f"checkpoint/{args.dataset}_{args.noise_mode}_cr2m_model.pt"
    
    os.makedirs("record", exist_ok=True)
    npz_path = f"record/{args.dataset}_{args.noise_mode}_cr2m.npz"
    np.savez_compressed(npz_path, softmax=probs, labels=y_test_np)
    print(f"[Saved] softmax & labels → {npz_path}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved PyTorch model to {save_path}")
    
# ==============================================================

if __name__ == '__main__':
    main()


