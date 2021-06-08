from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import math
import time
import argparse
import shutil

from dataloaders import get_test_dataloaders
from adaptive_inference import dynamic_evaluate
import models
from op_counter import measure_model
from utils import train, validate, save_checkpoint, load_checkpoint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str)
parser.add_argument('--data-root', default='../data', type=str, help='root path of dataset or where it should be downloaded')
parser.add_argument('--load-path', type=str, help='Path to load saved model from (.pth)')
parser.add_argument('--mode', default='gray', type=str, help="Perturbation mode. Options: 'gray' (default), 'noise', 'blur', 'color'")
parser.add_argument('--pert_std', default=0.0, type=float, help="Value of standard deviation for Gaussian distribution if mode == noise or blur")
parser.add_argument('--depth', default='s', type=str, help="size of backbone. Options: 's' (default), 'm', 'l'")
global_args = parser.parse_args()

class MSDNETConfig():
    """
    Description:
    Configuration for MSDNET. This will be important during training. 
    
    Important changes can be selection of data='cifar10' or 'cifar100'
    Epochs can be changed. Default should be 300
    Refer to https://github.com/kalviny/MSDNet-PyTorch/blob/master/README.md
    """
    # global args
    tag = global_args.tag
    data_root = global_args.data_root
    load_path = global_args.load_path
    mode = global_args.mode
    depth = global_args.depth
    pert_std = global_args.pert_std

    # medium and large model (for backbone comparison)
    if depth == 'm' or depth == 'l':
        arch='msdnetmod' 
        base=4
        batch_size=64
        bnFactor='1-2-4' 
        bottleneck=True 
        data='cifar10'
        data_root=data_root
        decay_rate=0.1
        epochs=200
        evalmode='dynamic'
        evaluate_from=load_path
        gpu='0'
        grFactor='1-2-4' 
        growthRate=6
        lr=0.1
        lr_type='multistep'
        momentum=0.9
        nBlocks=7
        nChannels=16
        optimizer='sgd'
        print_freq=10
        prune='max' 
        reduction=0.5
        resume=False
        save='./model_files/test/{}_{}'.format(tag,time.strftime('%Y.%m.%d_%H.%M.%S')) # change the save path to a new path
        seed=0
        start_epoch=0
        step=2
        stepmode='even' 
        use_valid=False
        weight_decay=0.0001 
        workers=4

    # smallest model (from paper)
    elif depth == 's':
        arch='msdnetmod' 
        base=3
        batch_size=64
        bnFactor='1-1-1' # changed
        bottleneck=False 
        data='cifar10'
        data_root=data_root
        decay_rate=0.1
        epochs=300                       # Change the epochs to 300
        evalmode='dynamic'
        evaluate_from=load_path
        gpu='0'
        grFactor='1-1-1' # changed
        growthRate=6
        lr=0.1
        lr_type='multistep'
        momentum=0.9
        nBlocks=7 # changed
        nChannels=8 # changed
        optimizer='sgd'
        print_freq=10
        prune='max' 
        reduction=0.5
        resume=False
        save='./model_files/test/{}_{}'.format(tag, time.strftime('%Y.%m.%d_%H.%M.%S')) # change the save path to a new path
        seed=0
        start_epoch=0
        step=2
        stepmode='even' 
        use_valid=False
        weight_decay=0.0001 
        workers=4

    else:
        raise ValueError


def run_training():

    global args
    best_prec1, best_epoch = 0.0, 0

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.data.startswith('cifar'):
        IM_SIZE = 32
    else:
        IM_SIZE = 224

    model = getattr(models, args.arch)(args)
    n_flops, n_params = measure_model(model, IM_SIZE, IM_SIZE)    
    torch.save(n_flops, os.path.join(args.save, 'flops.pth'))
    del(model)
        
        
    model = getattr(models, args.arch)(args)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    train_loader, val_loader, test_loader = get_test_dataloaders(args)

    if args.evalmode is not None:
        state_dict = torch.load(args.evaluate_from)['state_dict']
        model.load_state_dict(state_dict)

        if args.evalmode == 'anytime':
            validate(test_loader, model, criterion)
        else:
            dynamic_evaluate(model, test_loader, val_loader, args)
        return

    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_prec1'
              '\tval_prec1\ttrain_prec5\tval_prec5']

    for epoch in range(args.start_epoch, args.epochs):

        train_loss, train_prec1, train_prec5, lr = train(args, train_loader, model, criterion, optimizer, epoch)

        val_loss, val_prec1, val_prec5 = validate(args, val_loader, model, criterion)

        scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6)
                      .format(epoch, lr, train_loss, val_loss,
                              train_prec1, val_prec1, train_prec5, val_prec5))

        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = val_prec1
            best_epoch = epoch
            print('Best var_prec1 {}'.format(best_prec1))

        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best, model_filename, scores)

    print('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))

    ### Test the final model

    print('********** Final prediction results **********')
    validate(test_loader, model, criterion)

    print('finished {} epochs'.format(epoch))
    return True


if __name__ == "__main__":
    args = MSDNETConfig()

    if args.mode == 'noise':
        # noise_stds = list(np.arange(0.0, 0.11, 0.01))
        # args.noise = noise_stds[args.sweep_step - 1]
        args.noise = args.pert_std
        args.save = args.save + '_noise' + str(args.noise)
    elif args.mode == 'blur':
        # blur_stds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0]
        # args.blur = blur_stds[args.sweep_step - 1]
        args.blur = args.pert_std
        args.save = args.save + '_blur' + str(args.blur)
    elif args.mode == 'color':
        args.save = args.save + '_color'
    elif args.mode == 'gray':
        args.save = args.save + '_gray'

    print(args)
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.grFactor = list(map(int, args.grFactor.split('-')))
    args.bnFactor = list(map(int, args.bnFactor.split('-')))
    args.nScales = len(args.grFactor)

    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']

    if args.data == 'cifar10':
        args.num_classes = 10
    elif args.data == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 1000

    torch.manual_seed(args.seed)

    run_training()