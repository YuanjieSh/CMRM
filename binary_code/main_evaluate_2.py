import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import os
import sys
import random
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
# from pytorch_ops import soft_rank, soft_sort

from conformal_learning.utils import evaluate_predictions, get_scores_HPS, get_scores, classwise_conformal, Marginal_conformal
from conformal_learning import black_boxes_CNN
from conformal_learning.utils import *
from conformal_learning.help import *
from conformal_learning.black_boxes_CNN import Estimate_quantile_n, Scores_RAPS_all_diff, Scores_APS_all_diff, Scores_HPS_all_diff, PinballMarginal, UniformMatchingLoss, Estimate_size_loss_RAPS, save_plot, find_scores_RAPS, find_scores_APS, find_scores_HPS, load_train_objs, base_path_for_finetune, load_checkpoint, prepare_dataloader, loss_fnc, check_path, create_final_data, create_folder, test_model, loss_cal, create_optimizers
from conformal_learning.smooth_conformal_prediction import smooth_aps_score, smooth_aps_score_all

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
import joblib

sys.path.insert(0, './')

# ---------------- Arguments ------------------
parser = argparse.ArgumentParser(description='PyTorch UAI Training')
parser.add_argument('--dataset', default='adult', help='dataset setting')
parser.add_argument('--method', default='CE', help='method setting')
parser.add_argument('--noise_rho', default=0.01, type=float, help='noise fraction')
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

parser.add_argument('--cal_alpha', type=float, default=0.1, help='calibration')
parser.add_argument('--splits', type=int, default=10, help='split times for calibration')
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

    def forward(self, x):
        return self.fc(x)

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

def margin(sigmoid_scores, label, alpha_plus, alpha_minus, device):

    tau_plus = get_pos_threshold(sigmoid_scores, label, alpha_plus, device)

    tau_minus = get_neg_threshold(sigmoid_scores, label, alpha_minus, device)

    # margin = tau_minus - tau_plus

    return tau_plus, tau_minus, tau_minus - tau_plus

# ---------------- Evaluation ------------------

def evaluate_model(x_test, y_test, input_dim, model_path, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # print(f"args.method = '{args.method}'") 

    if args.model == 'LR':
        model = LogisticRegressionModel(input_dim).to(device)
    elif args.model == 'MLP':
        model = MLP(input_dim).to(device)
    elif args.model == 'SVM':
        model = LinearSVM(input_dim).to(device)
    else:
        raise ValueError("Unsupported model")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x_test_tensor)
        # print(f"logits shape: {logits.shape}")
        # print(f"args.method = {args.method}")

        if args.method == 'SVM' or args.method == 'Margin_10':
            scores = torch.sigmoid(logits).view(-1).cpu().numpy()
            # print(f"scores shape: {scores.shape}")

            probs_pos = scores  # P(y=1)
            probs_neg = 1.0 - scores  # P(y=0)
            probs = np.stack([probs_neg, probs_pos], axis=1)  # shape: (N, 2)
            # print(f"probs shape: {probs.shape}")

            preds = (logits > 0).long().view(-1).cpu().numpy() 
        else: 
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()  # shape: [N, 2]
            scores = probs[:, 1]
            preds = logits.argmax(dim=1).detach().cpu().numpy()
        
        labels = y_test_tensor.cpu().numpy()

    return probs, scores, preds, labels

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
        root_result = 'result/finetune/noise/' + base_path
    else:
        root_log =  'log/noise/' + base_path
        root_model = 'checkpoint/noise/' + base_path
        root_result = 'result/noise/' + base_path

    os.makedirs(root_result, exist_ok=True)    
    model_path = os.path.join(root_model, f"{store_name}.pt")

    # print(model_path)

    warnings.filterwarnings("ignore")
    x, y = load_data(args)
    # x_res, y_res = imbalance_sample(x, y, args.rho)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=args.seed)
    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x.columns)

    probs, scores, preds, labels = evaluate_model(x_test, y_test, x_train.shape[1], model_path, args)
    metrics = compute_all_metrics(labels, preds, scores)

    print(metrics)

    eval_name = f"{args.dataset}_{args.model}_{args.noise_rho}_{args.method}.csv"
    pd.DataFrame([metrics]).to_csv(os.path.join(root_result, eval_name), index=False)
    
    # plot_ROC_curve(labels, scores, metrics['AUC-ROC'], args, root_result)
    # plot_PR_curve(labels, scores, metrics['PR-AUC'], args, root_result)
    
    # y_test = y_test.to_numpy() 
    labels = torch.tensor(labels, dtype=torch.long).cpu()

    marginal_results = pd.DataFrame()
    class_results = pd.DataFrame()

    for experiment in tqdm(range(args.splits)):
        n_test = len(y_test)
        idx1, idx2 = train_test_split(np.arange(n_test), train_size=0.5, random_state = experiment + 1111)

        scores_cal = 1 - probs
    
        _, _, marginal_set_matrices = Marginal_conformal(scores_cal[idx1, labels[idx1]], labels[idx1], scores_cal[idx2, :], labels[idx2], args.cal_alpha,
                          num_classes=2, default_qhat=np.inf, regularize=False, exact_coverage=False)
        
        _, _, class_set_matrices = classwise_conformal(scores_cal[idx1, labels[idx1]], labels[idx1], scores_cal[idx2, :], labels[idx2], args.cal_alpha,
                          num_classes=2, default_qhat=np.inf, regularize=False, exact_coverage=False)

        marginal_res = evaluate_predictions(marginal_set_matrices, labels[idx2], preds[idx2], coverage_on_label=True, num_of_classes=2)

        class_res = evaluate_predictions(class_set_matrices, labels[idx2], preds[idx2], coverage_on_label=True, num_of_classes=2)
              
        marginal_res['Experiment'] = str(experiment + 1)
        class_res['Experiment'] = str(experiment + 1)

        marginal_results = pd.concat([marginal_results, marginal_res])
        class_results = pd.concat([class_results, class_res])

    
    marginal_results.to_csv(root_result + '/marginal_results.csv', index = False)
    class_results.to_csv(root_result + '/class_results.csv', index = False)


if __name__ == "__main__":
    main()
