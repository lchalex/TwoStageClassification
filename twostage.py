import argparse
import os
import os.path as osp
import random
import shutil
import time
import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from augmentation import get_augmentation
from data import TwoStageDataset, ClusterBatchSampler
from model import get_model, MultiheadResNet
from utils import AverageMeter, ProgressMeter, accuracy, adjust_learning_rate
from sklearn.metrics import accuracy_score

import logging
import pdb

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default="/srv/storage/alex/imagenet2012",
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=['resnet18', 'resnet50', 'resnet101'],
                    help='stage 2 model architecture (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--s1_epochs', default=0, type=int, metavar='N',
                    help='number of total epochs to train stage 1')
parser.add_argument('--s2_epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to train stage 2')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--s1_lr', '--s1-learning-rate', default=0.001, type=float,
                    metavar='LR', help='Stage 1 initial learning rate', dest='s1_lr')
parser.add_argument('--s2_lr', '--s2-learning-rate', default=0.01, type=float,
                    metavar='LR', help='Stage 2 initial learning rate', dest='s2_lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-5, type=float,
                    metavar='W', help='weight decay (default: 5e-5)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--debug', dest='debug', action='store_true',
                    help='debug mode')

def main():
    args = parser.parse_args()
    if args.pretrained:
        logname = "twostage_resnet50-{model}_lr_{lr2}_bs_{bs}_pretrained".format(model=args.arch[-2:], lr2=args.s2_lr, bs=args.batch_size)
    else:
        logname = "twostage_resnet50-{model}_lr_{lr2}_bs_{bs}".format(model=args.arch[-2:], lr2=args.s2_lr, bs=args.batch_size)

    if args.evaluate:
        logname = "twostage_evaluate"
    
    if args.debug:
        logname = "debug"

    path = osp.join(os.getcwd(), "records", logname)
    if not osp.isdir(path):
        os.makedirs(path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = osp.join(path, rq + 'resnet50-{}'.format(args.arch[-2:]) +'.log')

    fh = logging.FileHandler(log_name, mode='a')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    formatter = logging.Formatter("'%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s'")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # 输出到console的log等级的开关
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if torch.cuda.is_available():
        logging.info('using GPU')
        args.device = torch.device('cuda')
    else:
        logging.info('using CPU, this will be slow')
        args.device = torch.device('cpu')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    class_df = pd.read_csv(osp.join(args.data, 'class.csv'))

    if args.pretrained:
        logging.info("=> using pre-trained model '{}'".format(args.arch))
    else:
        logging.info("=> creating model '{}'".format(args.arch))

    args.num_cluster = len(class_df['cluster_id'].unique())
    args.num_classes = []
    for c in range(args.num_cluster):
        args.num_classes.append(len(class_df[class_df['cluster_id'] == c]))
    
    args.clsmapper = dict()
    for v1, v2, v3 in zip(class_df['cluster_id'], class_df['intra_cluster_id'], class_df['class_id']):
        args.clsmapper[(v1, v2)] = v3

    logging.info("num_cluster = {}".format(args.num_cluster))
    logging.info("num_classes = {}".format(str(args.num_classes)))

    cluster_model = get_model(arch="resnet50", num_class=args.num_cluster, pretrained=args.pretrained) # Use resnet50 for stage 1
    cluster_model = cluster_model.to(args.device)

    multihead_model = MultiheadResNet(arch=args.arch, num_cluster=args.num_cluster, num_classes=args.num_classes, pretrained=args.pretrained)
    multihead_model = multihead_model.to(args.device)

    train_transform, valid_transform = get_augmentation()
    train_dataset = TwoStageDataset(root=args.data, partition='train', transform=train_transform)
    valid_dataset = TwoStageDataset(root=args.data, partition='valid', transform=valid_transform)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    cluster_optimizer = torch.optim.SGD(cluster_model.parameters(), args.s1_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    cluster_scheduler = torch.optim.lr_scheduler.StepLR(cluster_optimizer, 20, gamma=0.1)

    common_params, multi_parmas = [], []
    for name, param in multihead_model.named_parameters():
        if name[:5] == 'multi':
            multi_parmas.append(param)
        else:
            common_params.append(param)

    multihead_optimizer = torch.optim.SGD(multihead_model.parameters(), args.s2_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    multihead_scheduler = torch.optim.lr_scheduler.StepLR(multihead_optimizer, 20, gamma=0.1)

    cudnn.benchmark = True

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    s1_best_acc1, s2_best_acc1 = 0, 0
    args.s1_start_epoch, args.s2_start_epoch = 0, 0

    if args.resume:
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        cluster_model = get_model(arch=checkpoint['stage1']['arch'], num_class=args.num_cluster, pretrained=False)
        cluster_model.load_state_dict(checkpoint['stage1']['state_dict'])
        cluster_model = cluster_model.to(args.device)
        if 'epoch' in checkpoint['stage1']:
            args.s1_start_epoch = checkpoint['stage1']['epoch']
        if 'best_acc1' in checkpoint['stage1']:
            s1_best_acc1 = checkpoint['stage1']['best_acc1']
        if 'optimizer' in checkpoint['stage1']:
            cluster_optimizer.load_state_dict(checkpoint['stage1']['optimizer'])
        if 'scheduler' in checkpoint['stage1']:
            cluster_scheduler.load_state_dict(checkpoint['stage1']['scheduler'])

        if 'arch' in checkpoint['stage2']:
            multihead_model = MultiheadResNet(arch=checkpoint['stage2']['arch'], num_cluster=args.num_cluster, num_classes=args.num_classes, pretrained=False)
            multihead_model.load_state_dict(checkpoint['stage2']['state_dict'])
            multihead_model = multihead_model.to(args.device)
            multihead_optimizer = torch.optim.SGD(multihead_model.parameters(), args.s2_lr, momentum=args.momentum, weight_decay=args.weight_decay)
            multihead_scheduler = torch.optim.lr_scheduler.StepLR(multihead_optimizer, 20, gamma=0.1)
        if 'epoch' in checkpoint['stage2']:
            args.s2_start_epoch = checkpoint['stage2']['epoch']
        if 'best_acc1' in checkpoint['stage2']:
            s2_best_acc1 = checkpoint['stage2']['best_acc1']
        if 'optimizer' in checkpoint['stage2']:
            multihead_optimizer.load_state_dict(checkpoint['stage2']['optimizer'])
        if 'scheduler' in checkpoint['stage2']:
            multihead_scheduler.load_state_dict(checkpoint['stage2']['scheduler'])

    if args.evaluate:
        validate_stage2(valid_loader, cluster_model, multihead_model, criterion, args)
        return

    writer = SummaryWriter(osp.join(os.getcwd(), 'runs', logname))
    writer.add_text('args', str(args))

    # stage 1
    for epoch in range(args.s1_start_epoch, args.s1_epochs):
        logging.info('Stage 1 - Epoch {}'.format(epoch + 1))

        # train for one epoch
        train_acc1, train_acc5, train_loss = train_stage1(train_loader, cluster_model, criterion, cluster_optimizer, cluster_scheduler, args)
        writer.add_scalar('stage1/train/acc1', train_acc1, epoch)
        writer.add_scalar('stage1/train/acc5', train_acc5, epoch)
        writer.add_scalar('stage1/train/loss', train_loss, epoch)
        writer.add_scalar('stage1/train/lr', cluster_optimizer.param_groups[0]['lr'], epoch)
        # evaluate on validation set
        val_acc1, val_acc5, val_loss = validate_stage1(valid_loader, cluster_model, criterion, args)
        writer.add_scalar('stage1/valid/acc1', val_acc1, epoch)
        writer.add_scalar('stage1/valid/acc5', val_acc5, epoch)
        writer.add_scalar('stage1/valid/loss', val_loss, epoch)
        # remember best acc@1 and save checkpoint

        is_best = val_acc1 > s1_best_acc1
        s1_best_acc1 = max(val_acc1, s1_best_acc1)

        save_checkpoint({
            'stage1': {
                'epoch': epoch + 1,
                'arch': "resnet50",
                'state_dict': cluster_model.state_dict(),
                'best_acc1': s1_best_acc1,
                'optimizer' : cluster_optimizer.state_dict(),
                'scheduler': cluster_scheduler.state_dict(),
            },
            'stage2': {
                'epoch': 0,
            }
        }, epoch, is_best, path, 'checkpoint_s1_{}.pth.tar'.format(str(epoch + 1).zfill(4)))

        if epoch + 1 == args.s1_epochs and osp.isfile(osp.join(path, 'model_best.pth.tar')):
            checkpoint = torch.load(osp.join(path, 'model_best.pth.tar'))
            cluster_model.load_state_dict(checkpoint['stage1']['state_dict'])
            logging.info('--------  Stage 1 Evaluation  --------')
            validate_stage1(valid_loader, cluster_model, criterion, args)

    # stage 2
    for epoch in range(args.s2_start_epoch, args.s2_epochs):
        logging.info('Stage 2 - Epoch {}'.format(epoch + 1))
        bsampler = ClusterBatchSampler(train_dataset.cluster, num_cluster_per_batch=2, batch_size=args.batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=bsampler, num_workers=args.workers, pin_memory=True)

        # train for one epoch
        train_acc1, train_loss = train_stage2(train_loader, multihead_model, criterion, multihead_optimizer, multihead_scheduler, args)
        writer.add_scalar('stage2/train/acc1', train_acc1, epoch)
        writer.add_scalar('stage2/train/loss', train_loss, epoch)
        writer.add_scalar('stage2/train/lr', multihead_optimizer.param_groups[0]['lr'], epoch)
        # evaluate on validation set
        val_acc1 = validate_stage2(valid_loader, cluster_model, multihead_model, criterion, args)
        writer.add_scalar('stage2/valid/acc1', val_acc1, epoch)
        # remember best acc@1 and save checkpoint

        is_best = val_acc1 > s2_best_acc1
        s2_best_acc1 = max(val_acc1, s2_best_acc1)

        save_checkpoint({
            'stage1': {
                'epoch': args.s1_epochs,
                'arch': "resnet50",
                'state_dict': cluster_model.state_dict(),
                'best_acc1': s1_best_acc1,
                'optimizer' : cluster_optimizer.state_dict(),
                'scheduler': cluster_scheduler.state_dict(),
            },
            'stage2': {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': multihead_model.state_dict(),
                'best_acc1': s2_best_acc1,
                'optimizer' : multihead_optimizer.state_dict(),
                'scheduler': multihead_scheduler.state_dict(),
            }
        }, epoch, is_best, path, 'checkpoint_s2_{}.pth.tar'.format(str(epoch + 1).zfill(4)))
    
    checkpoint = torch.load(osp.join(path, 'model_best.pth.tar'))
    cluster_model.load_state_dict(checkpoint['stage1']['state_dict'])
    multihead_model.load_state_dict(checkpoint['stage2']['state_dict'])
    logging.info('--------  Stage 2 Evaluation  --------')
    validate_stage2(valid_loader, cluster_model, multihead_model, criterion, args)

def train_stage1(train_loader, cluster_model, criterion, cluster_optimizer, cluster_scheduler, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    cluster_model.train()

    end = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, (target, _, _)) in pbar:
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(args.device)
        target = target.to(args.device)

        # compute output
        output = cluster_model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        # compute gradient and do SGD step
        cluster_optimizer.zero_grad()
        loss.backward()
        cluster_optimizer.step()

        pbar.set_postfix(**{'lr': cluster_optimizer.param_groups[0]['lr'], 'loss': losses.avg, 'top1': top1.avg, 'top5': top5.avg})

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    cluster_scheduler.step()
    return top1.avg, top5.avg, losses.avg

def validate_stage1(valid_loader, cluster_model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    cluster_model.eval()

    with torch.no_grad():
        end = time.time()
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for i, (images, (target, _, _)) in pbar:
            images = images.to(args.device)
            target = target.to(args.device)

            # compute output
            output = cluster_model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            pbar.set_postfix(**{'loss': losses.avg, 'top1': top1.avg, 'top5': top5.avg})

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

def train_stage2(train_loader, multihead_model, criterion, multihead_optimizer, multihead_scheduler, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    # switch to train mode
    multihead_model.train()

    end = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, (cluster, target, _)) in pbar:
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(args.device)
        cluster = cluster.to(args.device)
        target = target.to(args.device)

        # compute output
        output, indices = multihead_model(images, cluster)
        loss = torch.zeros(1).to(args.device)
        acc1 = torch.zeros(1).to(args.device)
        for cluid in range(args.num_cluster):
            if indices[cluid] is not None:
                ctarget = target[indices[cluid]]
                loss += criterion(output[cluid], ctarget) * ctarget.size(0)
                acc1 += accuracy(output[cluid], ctarget, topk=(1, ))[0] * ctarget.size(0)

        loss /= images.size(0)
        acc1 /= images.size(0)

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # compute gradient and do SGD step
        multihead_optimizer.zero_grad()
        loss.backward()
        multihead_optimizer.step()

        pbar.set_postfix(**{'lr': multihead_optimizer.param_groups[0]['lr'], 'loss': losses.avg, 'top1': top1.avg})

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    multihead_scheduler.step()
    return top1.avg, losses.avg

def validate_stage2(valid_loader, cluster_model, multihead_model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    cluster_model.eval()
    multihead_model.eval()

    end = time.time()
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for i, (images, (_, _, target)) in pbar:
        images = images.to(args.device)

        pred = []
        for j in range(images.size(0)):
            pred.append(predict(images[[j]], cluster_model, multihead_model, args.clsmapper))
        
        pred = torch.tensor(pred).type(target.type())
        acc1 = accuracy_score(target, pred) * 100

        # measure accuracy
        top1.update(acc1, images.size(0))

        pbar.set_postfix(**{'top1': top1.avg})

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logging.info(' * Acc@1 {top1.avg:.3f}'
            .format(top1=top1))

    return top1.avg

def predict(image, cluster_model, multihead_model, clsmapper):
    assert image.size(0) == 1, "Predict image must be single batch size"

    cluster_model.eval()
    multihead_model.eval()
    with torch.no_grad():
        output = cluster_model(image)
        _, cluster_pred = torch.max(output, 1)
        output, _ = multihead_model(image, cluster_pred)
        output = output[cluster_pred.item()]
        _, intra_pred = torch.max(output, 1)
    
    return clsmapper[(cluster_pred.item(), intra_pred.item())]

def set_common_layer_grad(model, value):
    for name, param in model.named_parameters():
        if name[:5] != 'multi':
            param.requires_grad = value

def set_multi_layer_grad(model, value):
    for name, param in model.named_parameters():
        if name[:5] == 'multi':
            param.requires_grad = value

def save_checkpoint(state, epoch, is_best, path, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, osp.join(path, filename))
        torch.save(state, osp.join(path, 'model_best.pth.tar'))
        return True
    
    return False

if __name__ == '__main__':
    main()
