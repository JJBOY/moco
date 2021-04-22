"""
python ssl_train.py MoCo /data/qinxin/CIFAR10 CIFAR10 -a resnet18 --lr 0.06 -b 512 -ms 4096 -url tcp://localhost:10001
python ssl_train.py MoCo /data/qinxin/CIFAR10 CIFAR10 -a resnet18 --lr 0.06 -b 512 -ms 4096 --symmetric -url tcp://localhost:10002
python ssl_train.py MoCo /data/qinxin/CIFAR10 CIFAR10 -a resnet18 --lr 0.06 -b 512 -ms 4096 --symmetric --projector -url tcp://localhost:10003
python ssl_train.py MoCo /data/qinxin/CIFAR10 CIFAR10 -a resnet18 --lr 0.06 -b 512 -ms 4096 --symmetric --projector --predictor -url tcp://localhost:10004

"""

import argparse

def parse_opt():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('method', choices=['MoCo', 'SimSiam'], default='MoCo',
                        help='self-supervised training method to use')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('dataset', help='dataset to train', default='CIFAR10')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        help='model architecture: ' +
                             ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('-url', '--dist-url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('-nmd', '--multiprocessing-distributed', action='store_false',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('-fd', '--feat-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('-ms', '--memory-size', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('-em', '--encoder-momentum', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('-st', '--softmax-temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('-ncos', '--cos', action='store_false',
                        help='use cosine lr schedule')
    parser.add_argument('--symmetric', action='store_true',
                        help='use symmetric simsiam loss')
    parser.add_argument('--projector', action='store_true',
                        help='use a 3 layers MLP instead of a 2 layers MLP in MoCov2')
    parser.add_argument('--predictor', action='store_true',
                        help='use predictor for asymmetric siamese networks')
    parser.add_argument('--sync_bn', action='store_true',
                        help='use synchronization batch normalization')
    args = parser.parse_args()
    return args
