import torch
import shutil
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from collections import Counter

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
            
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

def calc_confusion_mat(val_loader, model, args):
    
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    cf = confusion_matrix(all_targets, all_preds).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt

    print('Class Accuracy : ')
    print(cls_acc)
    classes = [str(x) for x in args.cls_num_list]
    plot_confusion_matrix(all_targets, all_preds, classes)
    plt.savefig(os.path.join(args.root_log, args.store_name, 'confusion_matrix.png'))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def prepare_folders(args):
    
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            print(folder, 'folder')
            os.makedirs(folder)

# def prepare_folders(args):
    
#     folders_util = [args.root_log, args.root_model,
#                     os.path.join(args.root_model, args.store_name)]
#     for folder in folders_util:
#         if not os.path.exists(folder):
#             print('creating folder ' + folder)
#             print(folder, 'folder')
#             os.makedirs(folder)

def save_checkpoint(args, state, is_best):
    
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        #print(pred.t())
        #print(target.view(1, -1))
        # print(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_intermediate_checkpoint(args, epoch, model, optimizer, best_acc1):
    epoch_dir = "dataset={}/architecture={}/loss_type={}/noise_rho={}/train_rule={}/epochs={}/batch-size={}/lr={}/momentum={}".format(
        args.dataset, args.arch, args.loss_type, args.noise_rho, args.train_rule,
        epoch, args.batch_size, args.lr, args.momentum
    )
    save_path = os.path.join("checkpoint/noise", epoch_dir)
    
    if not os.path.exists(save_path):
        print("Creating folder:", save_path)
        os.makedirs(save_path)

    filename = os.path.join(save_path, 'ckpt.pth.tar')
    torch.save({
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
    }, filename)


def save_intermediate_checkpoint_shift(args, epoch, model, optimizer, best_acc1):
    epoch_dir = "dataset={}/architecture={}/loss_type={}/corruption={}/severity={}/train_rule={}/epochs={}/batch-size={}/lr={}/momentum={}".format(
        args.dataset, args.arch, args.loss_type, args.corruption, args.severity, args.train_rule,
        epoch, args.batch_size, args.lr, args.momentum
    )
    save_path = os.path.join("checkpoint/noise", epoch_dir)
    
    if not os.path.exists(save_path):
        print("Creating folder:", save_path)
        os.makedirs(save_path)

    filename = os.path.join(save_path, 'ckpt.pth.tar')
    torch.save({
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
    }, filename)


def get_cls_num_list(dataset, num_classes=10):
    """
    Returns a list containing the number of samples for each class in the dataset.

    Args:
        dataset: PyTorch dataset object with a `targets` attribute (e.g., CIFAR10).
        num_classes: Total number of classes in the dataset.

    Returns:
        cls_num_list: List of length `num_classes`, where cls_num_list[i] = number of samples of class i.
    """
    targets = dataset.targets if hasattr(dataset, 'targets') else [label for _, label in dataset]
    counter = Counter(targets)
    cls_num_list = [counter.get(i, 0) for i in range(num_classes)]
    return cls_num_list


