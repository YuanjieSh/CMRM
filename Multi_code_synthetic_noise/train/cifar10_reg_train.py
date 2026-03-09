import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from corrupted_cifar import corrupted_CIFAR10, corrupted_CIFAR100
from utils import *
from losses import LDAMLoss, FocalLoss, MarginRegularizedLoss, MarginRegularizedLoss_2
import csv
from torch.nn.functional import softmax
from collections import defaultdict
from sklearn.metrics import top_k_accuracy_score
import torch.nn.functional as F
import pandas as pd

sys.path.insert(0, './')

import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet32)')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--noise_rho', default=0.01, type=float, help='noise fraction')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--alpha', default=0.1, type=float,
                    metavar='alpha', help='alpha for cvar', dest='alpha')
parser.add_argument('--reg', default=0.1, type=float,
                    metavar='regular', help='regularization for cvar', dest='reg')
parser.add_argument('--temp_scale', default=1.0, type=float, help='Temperature scaling factor (T)')
best_acc1 = 0


def main():
    args = parser.parse_args()
    if args.loss_type == 'CE_reg' or args.loss_type == 'Focal_reg' or args.loss_type == 'LDAM_reg':
        base_path = "dataset={}/architecture={}/loss_type={}/noise_rho={}/reg={}/alpha={}/train_rule={}/epochs={}/batch-size={}\
        /lr={}/momentum={}/".format(args.dataset, args.arch, args.loss_type, args.noise_rho, args.reg, args.alpha, args.train_rule,\
             args.epochs, args.batch_size, args.lr, args.momentum)
        args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, str(args.noise_rho), args.exp_str, str(args.reg), str(args.alpha)])
    else:
        base_path = "dataset={}/architecture={}/loss_type={}/noise_rho={}/train_rule={}/epochs={}/batch-size={}\
        /lr={}/momentum={}/".format(args.dataset, args.arch, args.loss_type, args.noise_rho, args.train_rule,\
             args.epochs, args.batch_size, args.lr, args.momentum)
        args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, str(args.noise_rho), args.exp_str])
    args.root_log =  'log_2/noise/' + base_path
    args.root_model = 'checkpoint_2/noise/' + base_path 
    # print(args.root_model)
    # args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, args.imb_type, str(args.imb_factor), args.exp_str])
    # print(args.store_name)

    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 100 if args.dataset == 'cifar100' else 10
    use_norm = True if args.loss_type in ['LDAM', 'LDAM_reg'] else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    cudnn.benchmark = True

    # Data loading code

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset = corrupted_CIFAR10(root='./data', noise_rho=args.noise_rho, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar100':
        train_dataset = corrupted_CIFAR100(root='./data', noise_rho=args.noise_rho, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return
    cls_num_list = train_dataset.get_cls_num_list()
    noisy_ids = train_dataset.get_noisy_indices()
    # print('cls num list:')
    # print(cls_num_list)
    args.cls_num_list = cls_num_list
    
    train_sampler = None
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

        # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            # print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            print(best_acc1)
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))


            base_path = "dataset={}/architecture={}/loss_type={}/noise_rho={}/reg={}/alpha={}/train_rule={}/start_epochs={}/epochs={}/batch-size={}\
            /lr={}/momentum={}/".format(args.dataset, args.arch, args.loss_type, args.noise_rho, args.reg, args.alpha, args.train_rule,\
                args.start_epoch, args.epochs, args.batch_size, args.lr, args.momentum)
            args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, str(args.noise_rho), args.exp_str, str(args.reg), str(args.alpha)])
            args.root_log =  'log_2/noise/fine_tune/' + base_path
            args.root_model = 'checkpoint_2/noise/fine_tune/' + base_path

            prepare_folders(args)

            device = torch.device(f'cuda:{args.gpu}' if args.gpu is not None else 'cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            model.to(device)

            all_margins = []
            all_labels = []
            all_indices = []
            all_is_noisy = []

            train_loader_eval = torch.utils.data.DataLoader(
                train_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

            noisy_set = set(train_dataset.get_noisy_indices())

            with torch.no_grad():
                for inputs, targets, indices in train_loader_eval:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs)  # [B, C]

                    # compute margins: true logit - max(other logits)
                    true_logits = outputs.gather(1, targets.unsqueeze(1)).squeeze(1)
                    tmp = outputs.clone()
                    tmp.scatter_(1, targets.unsqueeze(1), float('-inf'))
                    max_other_logits, _ = tmp.max(dim=1)
                    margins = (true_logits - max_other_logits).cpu().numpy()

                    all_margins.extend(margins.tolist())
                    all_labels.extend(targets.cpu().numpy().tolist())
                    all_indices.extend(indices.tolist())
                    all_is_noisy.extend([int(idx in noisy_set) for idx in indices.tolist()])

            df_margin = pd.DataFrame({
                "index": all_indices,
                "label": all_labels,
                "margin": all_margins,
                "is_noisy": all_is_noisy
            })

            os.makedirs(args.root_model, exist_ok=True)
            save_path = os.path.join(args.root_model, 'train_margin_info.csv')
            df_margin.to_csv(save_path, index=False)
            print(f"=> margin info saved to: {save_path}")
            
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        if args.train_rule == 'None':
            train_sampler = None  
            per_cls_weights = None 
        elif args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset)
            per_cls_weights = None
        elif args.train_rule == 'Reweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Sample rule is not listed')
        
        if args.loss_type == 'CE_reg':
            base_loss = nn.CrossEntropyLoss(weight=per_cls_weights, reduction='none').cuda(args.gpu)
            criterion = MarginRegularizedLoss_2(base_loss, alpha=args.alpha, reg=args.reg)
        elif args.loss_type == 'LDAM_reg':
            base_loss = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights, reduction='none').cuda(args.gpu)
            criterion = MarginRegularizedLoss(base_loss, alpha=args.alpha, reg=args.reg)
        elif args.loss_type == 'Focal_reg':
            base_loss = FocalLoss(weight=per_cls_weights, gamma=1, reduction='none').cuda(args.gpu)
            criterion = MarginRegularizedLoss_2(base_loss, alpha=args.alpha, reg=args.reg)
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, noisy_ids, args, log_training, tf_writer)
        
        # evaluate on validation set
        acc1, val_margin_logit, val_margin_prob = validate(val_loader, model, criterion, epoch, num_classes, args, log_testing, tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'val_margin_logit': val_margin_logit,   
            'val_margin_prob':  val_margin_prob,    
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, noisy_ids, args, log, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    cl_losses = AverageMeter('cl_Loss', ':.4e')
    crm_losses = AverageMeter('crm_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to train mode
    model.train()

    masked_global_indices = []

    end = time.time()
    for i, (input, target, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        #print(f"Input shape: {input.size()}")
        #print(f"Target shape: {target.size()}")
        # compute output
        output = model(input)
        #print(f"Output shape: {output.size()}")
        cl_loss, crm_loss, loss = criterion(output, target)


        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        cl_losses.update(cl_loss.item(), input.size(0))
        crm_losses.update(crm_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if hasattr(criterion, "get_last_masked_indices"):
            masked_idx = criterion.get_last_masked_indices()
            masked_global_indices.extend([index[i].item() for i in masked_idx])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()
    
    noise_ratio = 0.0  # default
    if hasattr(train_loader.dataset, 'get_noisy_indices'):
        noisy_ids = set(train_loader.dataset.get_noisy_indices())
        masked_set = set(masked_global_indices)
        if len(masked_set) > 0:
            noisy_in_mask = masked_set & noisy_ids
            noise_ratio = len(noisy_in_mask) / len(masked_set)
            print(f"[Epoch {epoch}] Masked={len(masked_set)}, Noisy-in-mask={len(noisy_in_mask)}, Ratio={noise_ratio:.4f}")

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
    tf_writer.add_scalar('noise_ratio/masked_train', noise_ratio, epoch)

    os.makedirs(args.root_model, exist_ok=True)
    csv_path = os.path.join(args.root_model, 'train_history.csv')
    header = not os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if header:
            f.write("epoch,total_loss,cl_loss,crm_loss,noise_ratio\n")
        f.write(f"{epoch},{losses.avg:.6f},{cl_losses.avg:.6f},{crm_losses.avg:.6f},{noise_ratio:.6f}\n")

def validate(val_loader, model, criterion, epoch, num_classes, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    cl_losses = AverageMeter('cl_Loss', ':.4e')
    crm_losses = AverageMeter('crm_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    m_logit = AverageMeter('MarginLogit', ':6.4f')   # NEW
    m_prob = AverageMeter('MarginProb',  ':6.4f')   # NEW

    T = getattr(args, 'temp_scale', 1.0)

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            cl_loss, crm_loss, loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            n = input.size(0)
            losses.update(loss.item(), input.size(0))
            cl_losses.update(cl_loss.item(), input.size(0))
            crm_losses.update(crm_loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            
            # ---------- margin_logit ----------
            true_log = output.gather(1, target.view(-1,1)).squeeze(1)      # [N]
            other_max = output.masked_fill(
                torch.zeros_like(output).scatter(1, target.view(-1,1), 1).bool(),
                -1e9).max(dim=1)[0]
            margin_l = (true_log - other_max).mean()
            m_logit.update(margin_l.item(), n)

            # ---------- margin_prob (with temperature) ----------
            output_T = output / T
            probs    = torch.softmax(output_T, dim=1)
            true_p   = probs.gather(1, target.view(-1,1)).squeeze(1)
            other_p  = probs.masked_fill(
                torch.zeros_like(probs).scatter(1, target.view(-1,1), 1).bool(),
                -1e9).max(dim=1)[0]
            margin_p = (true_p - other_p).mean()
            m_prob.update(margin_p.item(), n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        tf_writer.add_scalar('loss/test_'+ flag, losses.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i):x for i, x in enumerate(cls_acc)}, epoch)
        tf_writer.add_scalar(f'margin/logit_{flag}', m_logit.avg, epoch)
        tf_writer.add_scalar(f'margin/prob_{flag}',  m_prob.avg,  epoch)

        os.makedirs(args.root_model, exist_ok=True)
        csv_path = os.path.join(args.root_model, 'val_history.csv')
        header = not os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            if header:
                f.write("epoch,total_loss,cl_loss,crm_loss,m_logit, m_prob\n")
            f.write(f"{epoch},{losses.avg:.6f},{cl_losses.avg:.6f},{crm_losses.avg:.6f},{m_logit.avg:.6f},{m_prob.avg:.6f}\n")

    return top1.avg, m_logit.avg, m_prob.avg

def adjust_learning_rate(optimizer, epoch, args):
    """
    Custom learning rate schedule based on args.start_epoch and args.loss_type.
    """
    epoch = epoch + 1  # Epoch count starts from 1

    # Warm-up for first 5 epochs
    if epoch <= 5:
        lr = args.lr * epoch / 5

    elif args.start_epoch == 150:
        if args.dataset == 'cifar100':
            if epoch > 160:
                lr = args.lr * 0.01
            else:
                lr = args.lr
        else:
            if epoch > 180:
                lr = args.lr * 0.001
            elif epoch > 160:
                lr = args.lr * 0.01
            else:
                lr = args.lr

    elif args.start_epoch == 100:
        if args.loss_type == 'Focal_reg' or args.loss_type == 'CE_reg':
            if epoch > 180:
                lr = args.lr * 0.0001
            elif epoch > 130:
                lr = args.lr * 0.01
            else:
                lr = args.lr

        elif args.loss_type == 'LDAM_reg':
            if epoch > 180:
                lr = args.lr * 0.0001
            elif epoch > 160:
                lr = args.lr * 0.01
            else:
                lr = args.lr

        else:
            lr = args.lr

    else:
        lr = args.lr  # Default: no decay

    # Apply to all parameter groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

#    if epoch == args.epochs:
#
#       top_k_accuracy_matrix = calculate_accuracy_matrix(model, val_loader, device = args.gpu, num_classes = 10)
#       print(top_k_accuracy_matrix)
#
#       log_matrix = open(os.path.join(args.root_log, args.store_name, 'log_matrix.txt'), 'w')
#       with open(os.path.join(args.root_log, args.store_name, 'log_matrix.txt'), 'w') as f:
#           f.write(str(top_k_accuracy_matrix))
#       k, acc = calculate_accuracy_2(model, val_loader, device=args.gpu, num_classes = 10)
#       # k, acc = calculate_accuracy_2(model, train_loader, device=device, num_classes = num_classes)
#       # print(k)
#       print(acc[0])
