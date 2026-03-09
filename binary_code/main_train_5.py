import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import os
import sys
import random
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from conformal_learning.pytorch_ops import soft_rank, soft_sort

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib

sys.path.insert(0, './')

# ---------------- Arguments ------------------
parser = argparse.ArgumentParser(description='PyTorch UAI Training')
parser.add_argument('--dataset', default='adult', help='dataset setting')
parser.add_argument('--method', default='CE', help='method setting')
parser.add_argument('--noise_rho', default=0.01, type=float, help='noise fraction')
# parser.add_argument('--rho', default=0.01, type=float, help='imbalance fraction')
parser.add_argument('--model', type=str, default='LR', 
                    help='Model"')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lambda_reg', type=float, default=0.1, help='regularization strength for svm')
parser.add_argument('--lambda_plus', type=float, default=0.0, help='regularization strength for plus')
parser.add_argument('--lambda_minus', type=float, default=0.0, help='regularization strength for minus')
parser.add_argument('--b_plus', type=float, default=0.5, help='regularization threshold for plus')
parser.add_argument('--b_minus', type=float, default=0.5, help='regularization threshold for minus')
parser.add_argument('--alpha_minus', type=float, default=0.1, help='FPR rate')
parser.add_argument('--alpha_plus', type=float, default=0.1, help='FNR rate')
parser.add_argument('--pretrain', type=str, default=None, help='Path to pre-trained model for fine-tuning')
parser.add_argument('--finetune', type=bool, default=False, help='Enable fine-tuning from a pretrained model')
parser.add_argument('--scheduler', default='none', help='lr scheduler')
args = parser.parse_args()
REG_STRENGTH = 0.1

# ---------------- Set seed ------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ---------------- Data Loader ------------------
def load_data(args):
    name = args.dataset
    if name == "adult":
        df = pd.read_csv("./data/adult.csv")
        df.drop("fnlwgt", axis=1, inplace=True)
        df = df[df.ne('?').all(axis=1)]
        df['income'] = df['income'].apply(lambda x: 1 if x == '<=50K' else 0)
        x = df.drop("income", axis=1)
        y = df['income']
        obj_cols = x.select_dtypes(include=['object']).columns
        encoder = OrdinalEncoder()
        x[obj_cols] = encoder.fit_transform(x[obj_cols])

    elif name == "email":
        DATASET_PATH="./data/spambase.data"
        dfx = pd.read_csv(DATASET_PATH)
        TARGET_COLUMN = 'is_spam'
        # Use .iloc to select by integer-location
        y = dfx.iloc[:, -1]  # Get the last column        
        # Remove the last column for x
        x = dfx.iloc[:, :-1]        
        # Changing the labels
        y = 1 - y

    elif name == "credit":
        DATASET_PATH="./data/default of credit card clients.xls"
        dfx = pd.read_excel(DATASET_PATH, header=1)
        dfx = dfx.drop(dfx.columns[0], axis=1)
        TARGET_COLUMN = 'default payment next month'
        x=dfx.drop([TARGET_COLUMN],axis=1)
        y=dfx[TARGET_COLUMN]       
        # Changing the labels
        y = 1 - y

    elif name == "bank":
        DATASET_PATH="./data/bank-additional-full.csv"
        dfx = pd.read_csv(DATASET_PATH, delimiter=';')
        TARGET_COLUMN = 'y'
        dfx[TARGET_COLUMN] = dfx[TARGET_COLUMN].apply(lambda x: 1 if x=='no' else 0 )
        x=dfx.drop([TARGET_COLUMN],axis=1)
        y=dfx[TARGET_COLUMN]       
        # Changing the labels
        y = 1 - y
        obj=[]
        for i in dfx.columns:
            if dfx[i].dtype=='object':
                obj.append(i)
        encoder = OrdinalEncoder()
        for i in obj:
            x[i] = encoder.fit_transform(np.array(x[i]).reshape(-1,1))
    else:
        raise ValueError("Unsupported dataset")

    return x, y

def inject_label_noise(y, noise_ratio=0.1, seed=42):
    """
    Flip labels in y with a given noise ratio.

    Args:
        y (pd.Series): Binary labels (0 or 1).
        noise_ratio (float): Fraction of labels to flip.
        seed (int): Random seed for reproducibility.

    Returns:
        y_noisy (pd.Series): Label vector with flipped labels.
    """
    y_noisy = y.copy()
    n_samples = len(y)
    n_noisy = int(noise_ratio * n_samples)
    np.random.seed(seed)
    noise_indices = np.random.choice(n_samples, n_noisy, replace=False)
    y_noisy.iloc[noise_indices] = 1 - y_noisy.iloc[noise_indices]

    return y_noisy

# ---------------- Imbalance Sampler ------------------
def imbalance_sample(x, y, factor):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if factor <= len(neg_idx)/len(pos_idx):
        sampled_pos = np.random.choice(pos_idx, len(pos_idx), replace=False)
        sampled_neg = np.random.choice(neg_idx, int(len(pos_idx)*factor), replace=False)
    else:
        sampled_neg = np.random.choice(neg_idx, len(neg_idx), replace=False)
        sampled_pos = np.random.choice(pos_idx, int(len(neg_idx)/factor), replace=False)
    indices = np.concatenate([sampled_pos, sampled_neg])
    return x.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)

# ---------------- Model ------------------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.model(x)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)  # Binary classification

    def forward(self, x):
        return self.linear(x)

class LinearSVM(nn.Module):
    def __init__(self, input_dim):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        # nn.init.xavier_uniform_(self.fc.weight)
        # nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)

# ---------------- Method ------------------
class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [N, 2], targets: [N]
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)  # pt = softmax probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def hinge_loss(outputs, labels):
    labels = labels.view(-1, 1)
    return torch.mean(torch.clamp(1 - outputs * labels, min=0))

# def hinge_loss(outputs, labels, pos_weight=1.0, neg_weight=1.0):
#     labels = labels.view(-1, 1).float()
#     margins = 1 - outputs * labels
#     losses = torch.clamp(margins, min=0)
#     weights = torch.where(labels == 1, pos_weight, neg_weight)
#     return torch.mean(losses * weights)

def get_class_weights(y_tensor):
    """
    Compute class weights inversely proportional to class frequency.
    For hinge loss with labels in {-1, +1}
    """
    pos_count = (y_tensor == 1).sum().float()
    neg_count = (y_tensor == 0).sum().float()
    
    total = pos_count + neg_count

    # Inverse frequency
    pos_weight = total / (2.0 * pos_count)
    neg_weight = total / (2.0 * neg_count)

    return pos_weight.item(), neg_weight.item()

def train_svm(x_train, y_train, x_test, y_test, model_path):
    clf = LinearSVC()
    model = CalibratedClassifierCV(clf, method='sigmoid')  
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    probs_train = model.predict_proba(x_train)
    probs_test = model.predict_proba(x_test)

    scores_train = probs_train[:, 1]
    scores_test = probs_test[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    metrics = {
        'accuracy': [accuracy_score(y_test, y_pred_test)],
        'f1': [f1_score(y_test, y_pred_test)],
        'precision': [precision_score(y_test, y_pred_test)],
        'recall': [recall_score(y_test, y_pred_test)],
        'fnr': [fn / (fn + tp) if (fn + tp) > 0 else 0.0],
        'fpr': [fp / (fp + tn) if (fp + tn) > 0 else 0.0]
    }

    metric_path = os.path.join(os.path.dirname(model_path), 'metrics.npz')
    np.savez(metric_path, **metrics)

    joblib.dump(model, model_path)

    print(f"[INFO] SVM model saved to {model_path}.")

    return y_train, y_pred_train, scores_train, y_test, y_pred_test, scores_test
# ---------------- Regularization ------------------
# def Smoothquantile(scores, alpha, device):
#     n = len(scores)
#     sorted_score = soft_sort(scores.reshape((1, n)), regularization_strength=REG_STRENGTH, device = device).flatten()
#     #print(f"sorted_score = {sorted_score}, {sorted_score.shape}")
#     index = int(n*(1.0-alpha))
#     scores_q_t = sorted_score[index]
#     #print(f"scores_q_t = {scores_q_t}")
#     return scores_q_t

def get_neg_threshold(sigmoid_scores, label, alpha, device):

    # label_np = label.cpu().numpy()
    idx = (label == 0)
    # neg_sigmoid_scores_np = sigmoid_scores.cpu().numpy()
    neg_scores = sigmoid_scores[idx]

    n = len(neg_scores)
    if n == 0:
        return torch.tensor(0.0, device=device)  # fallback for edge case

    sorted_score = soft_sort(neg_scores.reshape((1, n)), regularization_strength=REG_STRENGTH, device = device).flatten()
    index = int(n*(1.0-alpha))
    scores_q_t = sorted_score[index]

    return scores_q_t

def get_pos_threshold(sigmoid_scores, label, alpha, device):
    # Convert pandas Series to numpy array
    # label_np = label.cpu().numpy()
    idx = (label == 1)
    # sigmoid_scores_np = sigmoid_scores.cpu().numpy()
    pos_scores = sigmoid_scores[idx]

    n = len(pos_scores)
    if n == 0:
        return torch.tensor(0.0, device=device)  # fallback for edge case

    sorted_score = soft_sort(pos_scores.reshape((1, n)), regularization_strength=REG_STRENGTH, device = device).flatten()
    index = int(n*alpha)
    scores_q_t = sorted_score[index]

    # n = len(pos_scores)
    # threshold = np.ceil((n-1)*alpha)/n

    # if threshold > 1:
    #     qhat = np.quantile(pos_scores, 1, method='inverted_cdf')
    # else:
    #     qhat = np.quantile(pos_scores, threshold, method='inverted_cdf')

    return scores_q_t 

def margin(sigmoid_scores, label, args, device):

    tau_plus = get_pos_threshold(sigmoid_scores, label, args.alpha_plus, device).detach()

    tau_minus = get_neg_threshold(sigmoid_scores, label, args.alpha_minus, device).detach()

    # margin = tau_minus - tau_plus

    return tau_plus, tau_minus

def margin_2(sigmoid_scores, label, args, device):
    
    # Prevent gradient from flowing through threshold computation
    tau_plus = get_pos_threshold(sigmoid_scores, label, args.alpha_plus, device).detach()
    tau_minus = get_neg_threshold(sigmoid_scores, label, args.alpha_minus, device).detach()

    pos_mask = (label == 1) & (sigmoid_scores <= tau_plus)
    neg_mask = (label == 0) & (sigmoid_scores >= tau_minus)

    pos_tail = sigmoid_scores[pos_mask]
    neg_tail = sigmoid_scores[neg_mask]

    if len(pos_tail) > 0:
        pos_loss = F.relu(args.b_plus - pos_tail).mean()
    else:
        pos_loss = torch.tensor(0.0, device=device)

    if len(neg_tail) > 0:
        neg_loss = F.relu(neg_tail - args.b_minus).mean()
    else:
        neg_loss = torch.tensor(0.0, device=device)

    margin_loss_plus = args.lambda_plus * pos_loss

    margin_loss_minus =  args.lambda_minus * neg_loss

    # margin_loss = args.lambda_plus * pos_loss + args.lambda_minus * neg_loss


    # margin_plus = args.b_plus - tau_plus

    # margin_minus = tau_minus - args.b_minus 

    return tau_plus, tau_minus, margin_loss_plus, margin_loss_minus

def margin_3(sigmoid_scores, label, args, device):

    tau_plus = get_pos_threshold(sigmoid_scores, label, args.alpha_plus, device).detach()
    tau_minus = get_neg_threshold(sigmoid_scores, label, args.alpha_minus, device).detach()

    pos_mask = (label == 1) & (sigmoid_scores <= tau_plus)
    neg_mask = (label == 0) & (sigmoid_scores >= tau_minus)

    pos_tail = sigmoid_scores[pos_mask]
    neg_tail = sigmoid_scores[neg_mask]

    if len(pos_tail) > 0:
        pos_loss = F.relu(tau_plus - pos_tail).mean()
    else:
        pos_loss = torch.tensor(0.0, device=device)

    if len(neg_tail) > 0:
        neg_loss = F.relu(neg_tail - tau_minus).mean()
    else:
        neg_loss = torch.tensor(0.0, device=device)


    margin_loss_plus = args.lambda_plus * pos_loss

    margin_loss_minus =  args.lambda_minus * neg_loss

    # margin_loss = args.lambda_plus * pos_loss + args.lambda_minus * neg_loss


    # margin_plus = args.b_plus - tau_plus

    # margin_minus = tau_minus - args.b_minus 

    return tau_plus, tau_minus, margin_loss_plus, margin_loss_minus

def margin_4(sigmoid_scores, label, args, device):

    tau_plus = get_pos_threshold(sigmoid_scores, label, args.alpha_plus, device).detach()
    tau_minus = get_neg_threshold(sigmoid_scores, label, args.alpha_minus, device).detach()

    pos_mask = (label == 1) & (sigmoid_scores <= tau_plus)
    neg_mask = (label == 0) & (sigmoid_scores >= tau_minus)

    pos_tail = sigmoid_scores[pos_mask]
    neg_tail = sigmoid_scores[neg_mask]

    if len(pos_tail) > 0:
        pos_loss = - torch.mean(pos_tail)
    else:
        pos_loss = torch.tensor(0.0, device=device)

    if len(neg_tail) > 0:
        neg_loss = torch.mean(neg_tail)
    else:
        neg_loss = torch.tensor(0.0, device=device)

    margin_loss_plus = args.lambda_plus * pos_loss

    margin_loss_minus =  args.lambda_minus * neg_loss

    # margin_loss = args.lambda_plus * pos_loss + args.lambda_minus * neg_loss


    # margin_plus = args.b_plus - tau_plus

    # margin_minus = tau_minus - args.b_minus 

    return tau_plus, tau_minus, margin_loss_plus, margin_loss_minus

# ---------------- Training ------------------
def train_model(x_train, y_train, x_test, y_test, input_dim, model_path, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.model == 'LR':
        model = LogisticRegressionModel(input_dim).to(device)
    elif args.model == 'MLP':
        model = MLP(input_dim).to(device)
    elif args.model == 'SVM':
        model = LinearSVM(input_dim).to(device)
    else:
        raise ValueError("Unsupported model")
    # print("Initial weight norm:", torch.norm(model.fc.weight).item())

    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

    
    if args.finetune and args.pretrain is not None:
        print(f"[INFO] Fine-tuning from pre-trained model: {args.pretrain}")
        
        checkpoint = torch.load(args.pretrain, map_location=device)
    
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint  # assume it's already a state_dict
        model.load_state_dict(state_dict)

    if os.path.exists(model_path):
        print(f"[INFO] Found existing model at {model_path}. Skipping training...")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # model.load_state_dict(torch.load(model_path))
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        elif args.scheduler == 'none':
            scheduler = None

        metrics = {
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'fnr': [],
            'fpr': []
        }


        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(x_train_tensor)
            # sigmoid_scores = torch.sigmoid(logits)

            if args.method == 'CE':

                criterion = nn.CrossEntropyLoss()
                loss = criterion(logits, y_train_tensor) 
                probs = torch.softmax(logits, dim=1)
            
                tau_plus, tau_minus = margin(probs[:, 1], y_train_tensor, args, device)

                tau_plus_all.append(tau_plus.item())
                tau_minus_all.append(tau_minus.item())

                loss_all.append(loss.item()) 

            elif args.method == 'FOCAL':

                criterion = SoftmaxFocalLoss(gamma=2.0)
                loss = criterion(logits, y_train_tensor)  
                probs = torch.softmax(logits, dim=1)
            
                # loss = SoftmaxFocalLoss(gamma=2.0)
                tau_plus, tau_minus = margin(probs[:, 1], y_train_tensor, args, device)

                tau_plus_all.append(tau_plus.item())
                tau_minus_all.append(tau_minus.item())

                loss_all.append(loss.item()) 
            
            elif args.method == 'SVM':

                # criterion = hinge_loss()
                # pos_weight, neg_weight = get_class_weights(y_train_tensor)
                y_train_svm_tensor = y_train_tensor * 2 - 1

                loss = hinge_loss(logits, y_train_svm_tensor)

                l2_reg = 0.5 * torch.norm(model.fc.weight) ** 2
                loss = loss + args.lambda_reg * l2_reg

                # loss = SoftmaxFocalLoss(gamma=2.0)
                probs = torch.sigmoid(logits).view(-1)
                tau_plus, tau_minus = margin(probs, y_train_tensor, args, device)

                tau_plus_all.append(tau_plus.item())
                tau_minus_all.append(tau_minus.item())

                loss_all.append(loss.item()) 
                
            elif args.method == 'Margin_8':

                criterion = nn.CrossEntropyLoss()
                probs = torch.softmax(logits, dim=1)
                tau_plus, tau_minus, margin_loss_plus, margin_loss_minus = margin_3(probs[:, 1], y_train_tensor, args, device)
                tau_plus_all.append(tau_plus.item())
                tau_minus_all.append(tau_minus.item())

                loss_ce = criterion(logits, y_train_tensor)
                loss_margin = margin_loss_plus + margin_loss_minus
                loss = loss_ce + loss_margin
                
                loss_all.append(loss.item()) 
                loss_ce_all.append(loss_ce.item()) 
                loss_margin_all.append(loss_margin.item())
                loss_margin_plus_all.append(margin_loss_plus.item())
                loss_margin_minus_all.append(margin_loss_minus.item())


            elif args.method == 'Margin_9':
                criterion = SoftmaxFocalLoss(gamma=2.0)

                probs = torch.softmax(logits, dim=1)
                tau_plus, tau_minus, margin_loss_plus, margin_loss_minus = margin_3(probs[:, 1], y_train_tensor, args, device)
                tau_plus_all.append(tau_plus.item())
                tau_minus_all.append(tau_minus.item())

                loss_cl = criterion(logits, y_train_tensor)
                loss_margin = margin_loss_plus + margin_loss_minus
                loss = loss_cl + loss_margin
                
                loss_all.append(loss.item()) 
                loss_ce_all.append(loss_cl.item()) 
                loss_margin_all.append(loss_margin.item())
                loss_margin_plus_all.append(margin_loss_plus.item())
                loss_margin_minus_all.append(margin_loss_minus.item())


            elif args.method == 'Margin_10':
               
                pos_weight, neg_weight = get_class_weights(y_train_tensor)
                y_train_svm_tensor = y_train_tensor * 2 - 1

                loss_cl = hinge_loss(logits, y_train_svm_tensor)
                # loss_cl = hinge_loss(logits, y_train_svm_tensor, pos_weight, neg_weight)

                l2_reg = 0.5 * torch.norm(model.fc.weight) ** 2
                loss_cl = loss_cl + args.lambda_reg * l2_reg

                probs = torch.sigmoid(logits).view(-1)

                tau_plus, tau_minus, margin_loss_plus, margin_loss_minus = margin_3(probs, y_train_tensor, args, device)
                tau_plus_all.append(tau_plus.item())
                tau_minus_all.append(tau_minus.item())

                # loss_cl = criterion(logits, y_train_tensor)
                loss_margin = margin_loss_plus + margin_loss_minus
                loss = loss_cl + loss_margin
                
                loss_all.append(loss.item()) 
                loss_ce_all.append(loss_cl.item()) 
                loss_margin_all.append(loss_margin.item())
                loss_margin_plus_all.append(margin_loss_plus.item())
                loss_margin_minus_all.append(margin_loss_minus.item())

            else:
                raise ValueError("Unsupported method")  

            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            model.eval()
            with torch.no_grad():
                y_pred_eval = model(x_test_tensor).argmax(dim=1).cpu().numpy()
                y_true_eval = y_test_tensor.cpu().numpy()
                tn, fp, fn, tp = confusion_matrix(y_true_eval, y_pred_eval).ravel()
        
        checkpoint = {
            'model_state_dict': model.state_dict()           
        }

        # torch.save(model.state_dict(), model_path)
        torch.save(checkpoint, model_path)
        print(f"[INFO] Model saved to {model_path}.")

        metric_path = os.path.join(os.path.dirname(model_path), 'metrics.npz')
        np.savez(metric_path, **metrics)

    model.eval()
    with torch.no_grad():
        # -------- Training Data --------
        logits_train = model(x_train_tensor)

        if args.method == 'SVM' or args.method == 'Margin_10':
            probs_train = torch.sigmoid(logits_train).view(-1).cpu().numpy()
            scores_train = probs_train
            y_pred_train = (logits_train > 0).long().view(-1).cpu().numpy() 
        else: 
            probs_train = torch.softmax(logits_train, dim=1).cpu().numpy()
            scores_train = probs_train[:, 1]
            y_pred_train = logits_train.argmax(dim=1).cpu().numpy()

        # print("Logits min:", logits_train.min().item(), "max:", logits.max().item())
        # print("Sigmoid mean:", torch.sigmoid(logits_train).mean().item())
        # print("Final weight norm:", torch.norm(model.fc.weight).item())
        # -------- Test Data --------
        logits_test = model(x_test_tensor)

        if args.method == 'SVM' or args.method == 'Margin_10':
            probs_test = torch.sigmoid(logits_test).view(-1).cpu().numpy()
            scores_test = probs_test
            y_pred_test = (logits_test > 0).long().view(-1).cpu().numpy() 
        else: 
            probs_test = torch.softmax(logits_test, dim=1).cpu().numpy()
            scores_test = probs_test[:, 1]
        # probs_test = torch.softmax(logits_test, dim=1).cpu().numpy()
        # scores_test = probs_test[:, 1]
            y_pred_test = logits_test.argmax(dim=1).cpu().numpy()

    return y_train.values, y_pred_train, scores_train, y_test.values, y_pred_test, scores_test

# ---------------- Evaluation ------------------
    # plt.grid(True)

def compute_metrics(pred, target):
    tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
    return {
        "accuracy": accuracy_score(target, pred),
        "f1": f1_score(target, pred, average="macro"),
        "precision": precision_score(target, pred, average="macro"),
        "recall": recall_score(target, pred, average="macro"),
        "FNR": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
    }

def compute_all_metrics(y_true, y_pred, y_prob, n_bins=10):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    balanced_acc = 0.5 * (tp / (tp + fn) + tn / (tn + fp)) if (tp + fn) > 0 and (tn + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Calibration metrics
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    ece = np.abs(prob_true - prob_pred).mean()
    mce = np.abs(prob_true - prob_pred).max()
    brier = brier_score_loss(y_true, y_prob)

    # AUC-based metrics
    auc_roc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    # Risk-Coverage Curve (AURC / E-AURC)
    sorted_indices = np.argsort(-y_prob)
    sorted_labels = np.array(y_true)[sorted_indices]
    sorted_preds = np.array(y_pred)[sorted_indices]
    coverage = np.arange(1, len(y_true)+1) / len(y_true)
    risk = np.cumsum(sorted_preds != sorted_labels) / np.arange(1, len(y_true)+1)
    aurc = np.trapz(risk, coverage)
    e_aurc = np.mean(risk)

    return {
        "AUC-ROC": auc_roc,
        "PR-AUC": pr_auc,
        # "recall": recall,
        # "specificity": specificity,
        "FNR": fnr,
        "FPR": fpr,
        "ECE": ece,
        # "MCE": mce,
        "Brier": brier,
        "precision": precision,
        "accuracy": accuracy,
        "f1": f1
        # "AURC": aurc,
        # "E-AURC": e_aurc
    }

# ---------------- Main ------------------
def main():
    args = parser.parse_args()
    set_seed(args.seed)

    if args.method == 'CE' or args.method == 'FOCAL':
        base_path = "dataset={}/model={}/noise_fac={}/loss={}/epochs={}/lr={}/".format(args.dataset, args.model, args.noise_rho, args.method, args.epochs, args.lr)
        store_name = '_'.join([args.dataset, args.model, str(args.noise_rho), args.method, str(args.epochs), str(args.lr)])
    elif args.method == 'Margin_8' or args.method == 'Margin_9':
        base_path = "dataset={}/model={}/noise_fac={}/loss={}/reg_plus={}/reg_minus={}/b_plus={}/b_minus={}/alpha_plus={}/alpha_minus={}/epochs={}/lr={}/scheduler={}/".format(args.dataset, args.model, args.noise_rho, args.method, args.lambda_plus, args.lambda_minus, args.b_plus, args.b_minus, args.alpha_plus, args.alpha_minus, args.epochs, args.lr, args.scheduler)
        store_name = '_'.join([args.dataset, args.model, str(args.noise_rho), args.method, str(args.lambda_plus), str(args.lambda_minus), str(args.b_plus), str(args.b_minus), str(args.alpha_plus), str(args.alpha_minus), str(args.epochs), str(args.lr)])
    
    elif args.method == 'SVM':
        base_path = "dataset={}/model={}/noise_fac={}/loss={}/reg_l2={}/epochs={}/lr={}/".format(args.dataset, args.model, args.noise_rho, args.method, args.lambda_reg, args.epochs, args.lr)
        store_name = '_'.join([args.dataset, args.model, str(args.noise_rho), args.method, str(args.lambda_reg), str(args.epochs), str(args.lr)])
    
    elif args.method == 'Margin_10':
        base_path = "dataset={}/model={}/noise_fac={}/loss={}/reg_l2={}/reg_plus={}/reg_minus={}/b_plus={}/b_minus={}/alpha_plus={}/alpha_minus={}/epochs={}/lr={}/scheduler={}/".format(args.dataset, args.model, args.noise_rho, args.method, args.lambda_reg, args.lambda_plus, args.lambda_minus, args.b_plus, args.b_minus, args.alpha_plus, args.alpha_minus, args.epochs, args.lr, args.scheduler)
        store_name = '_'.join([args.dataset, args.model, str(args.noise_rho), args.method, str(args.lambda_reg), str(args.lambda_plus), str(args.lambda_minus), str(args.b_plus), str(args.b_minus), str(args.alpha_plus), str(args.alpha_minus), str(args.epochs), str(args.lr)])
    
    else:
        raise ValueError("Unsupported method")

    if args.finetune:
        root_log =  'log/finetune/noise/' + base_path
        root_model = 'checkpoint/finetune/noise/' + base_path
    else:
        root_log =  'log/noise/' + base_path
        root_model = 'checkpoint/noise/' + base_path

    warnings.filterwarnings("ignore")
    x, y = load_data(args)
    # x_res, y_res = imbalance_sample(x, y, args.rho)

    os.makedirs(root_model, exist_ok=True)
    model_path = os.path.join(root_model, f"{store_name}.pt")
    
    os.makedirs(root_log, exist_ok=True)
    log_file = os.path.join(root_log, 'experiment.log')

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=log_file,
                    filemode='a')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=args.seed)

    if args.noise_rho > 0:
        y_train = inject_label_noise(y_train, noise_ratio=args.noise_rho, seed=args.seed)

    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x.columns)
    

    y_true_train, y_pred_train, scores_train, y_true_test, y_pred_test, scores_test = train_model(x_train, y_train, x_test, y_test, x_train.shape[1], model_path, args)

    train_score = compute_all_metrics(y_true_train, y_pred_train, scores_train)
    test_score = compute_all_metrics(y_true_test, y_pred_test, scores_test)

    print(f"[TRAIN] dataset={args.dataset}, rho={args.noise_rho}, score={train_score}")
    print(f"[TEST] dataset={args.dataset}, rho={args.noise_rho}, score={test_score}")

    logging.info("[TRAIN] dataset=%s, rho=%s, score=%s", args.dataset, args.noise_rho, train_score)
    logging.info("[TEST] dataset=%s, rho=%s, score=%s", args.dataset, args.noise_rho, test_score)

if __name__ == "__main__":
    main()
