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
from data import TwoStageDataset
from model import AuxResNet
from utils import AverageMeter, ProgressMeter, adjust_learning_rate, accuracy

import logging
import pdb

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default="input",
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=['resnet18', 'resnet50', 'resnet101'],
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-5, type=float,
                    metavar='W', help='weight decay (default: 5e-5)',
                    dest='weight_decay')
parser.add_argument('--loss_ratio', default=0.25, type=float,
                    help='number of epochs to decay loss ratio')
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
        logname = "auxiliary_ratio{ratio}_{arch}_lr_{lr}_bs_{bs}_pretrained".format(ratio=args.loss_ratio, arch=args.arch, lr=args.lr, bs=args.batch_size)
    else:
        logname = "auxiliary_ratio{ratio}_{arch}_lr_{lr}_bs_{bs}".format(ratio=args.loss_ratio, arch=args.arch, lr=args.lr, bs=args.batch_size)

    if args.evaluate:
        logname = "auxiliary_evaluate"

    if args.debug:
        logname = "debug"

    path = osp.join(os.getcwd(), "records", logname)
    if not osp.isdir(path):
        os.makedirs(path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = osp.join(path, rq + args.arch +'.log')

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
    args.num_class = len(class_df)

    logging.info("num_cluster = {}".format(args.num_cluster))
    logging.info("num_class = {}".format(args.num_class))

    model = AuxResNet(arch=args.arch, num_cluster=args.num_cluster, num_class=args.num_class, pretrained=args.pretrained)
    model = model.to(args.device)
    train_transform, valid_transform = get_augmentation()
    train_dataset = TwoStageDataset(root=args.data, partition='train', transform=train_transform)
    valid_dataset = TwoStageDataset(root=args.data, partition='valid', transform=valid_transform)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    best_acc1 = 0
    args.start_epoch = 0

    if args.resume:
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model = AuxResNet(arch=checkpoint['arch'], num_cluster=args.num_cluster, num_class=args.num_class, pretrained=False)
        model.to(args.device)
        model.load_state_dict(checkpoint['state_dict'])
        best_acc1 = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])


    if args.evaluate:
        validate(valid_loader, model, criterion, args)
        time.sleep(1)
        return 

    writer = SummaryWriter(osp.join(os.getcwd(), 'runs', logname))
    writer.add_text('args', str(args))

    for epoch in range(args.start_epoch, args.epochs):
        logging.info('Epoch {}'.format(epoch + 1))
        adjust_learning_rate(optimizer, epoch, 20, args)
        
        # train for one epoch
        train_acc1, train_acc5, train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        writer.add_scalar('train/acc1', train_acc1, epoch)
        writer.add_scalar('train/acc5', train_acc5, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/loss_ratio', args.loss_ratio, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        # evaluate on validation set
        val_acc1, val_acc5, val_loss = validate(valid_loader, model, criterion, args)
        writer.add_scalar('valid/acc1', val_acc1, epoch)
        writer.add_scalar('valid/acc5', val_acc5, epoch)
        writer.add_scalar('valid/loss', val_loss, epoch)
        # remember best acc@1 and save checkpoint

        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, epoch, is_best, path, 'checkpoint_{}.pth.tar'.format(str(epoch + 1).zfill(4)))
    

    checkpoint = torch.load(osp.join(path, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    logging.info('--------  Evaluation  --------')
    validate(valid_loader, model, criterion, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    aux_losses = AverageMeter('AuxLoss', ':.4e')
    out_losses = AverageMeter('OutLoss', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()
    loss_ratio = args.loss_ratio

    end = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, (cluster, _, target)) in pbar:
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(args.device)
        cluster = cluster.to(args.device)
        target = target.to(args.device)

        # compute output
        output, aux = model(images)
        aux_loss = criterion(aux, cluster)
        out_loss = criterion(output, target)
        loss = aux_loss * loss_ratio + out_loss * (1 - loss_ratio)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        aux_losses.update(aux_loss.item(), images.size(0))
        out_losses.update(out_loss.item(), images.size(0))
        losses.update(loss.item(), images.size(0))

        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(**{
            'lr': optimizer.param_groups[0]['lr'], 
            'aux_loss': aux_losses.avg, 
            'out_loss': out_losses.avg, 
            'loss': losses.avg, 
            'top1': top1.avg, 
            'top5': top5.avg
        })

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg, top5.avg, losses.avg

def validate(valid_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for i, (images, (_, _, target)) in pbar:
            images = images.to(args.device)
            target = target.to(args.device)

            # compute output
            output, _ = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            _, pred = torch.max(output, 1)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            pbar.set_postfix(**{
                'loss': losses.avg, 
                'top1': top1.avg, 
                'top5': top5.avg
            })

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # TODO: this should also be done with the ProgressMeter
        logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, epoch, is_best, path, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, osp.join(path, filename))
        torch.save(state, osp.join(path, 'model_best.pth.tar'))
        return True
    
    return False

if __name__ == '__main__':
    main()
