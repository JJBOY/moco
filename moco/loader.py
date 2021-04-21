# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch
import torch.utils.data
import torch.utils.data.distributed
from torchvision.datasets import CIFAR10, CIFAR100


def get_data_loader(data_path, data_type='ImageNet', batch_size=256, workers=8, distributed=False):
    if data_type == 'ImageNet':
        traindir = os.path.join(data_path, 'train')
        testdir = os.path.join(data_path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

        train_dataset = datasets.ImageFolder(
            traindir,
            TwoCropsTransform(transforms.Compose(augmentation)))
        memory_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(test_transform))
        test_dataset = datasets.ImageFolder(
            testdir,
            transforms.Compose(test_transform))

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler, drop_last=True)
        memory_loader = torch.utils.data.DataLoader(
            memory_dataset, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, sampler=train_sampler)

    elif 'CIFAR' in data_type:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        if data_type == 'CIFAR10':
            train_dataset = CIFAR10(root=data_path, train=True, transform=TwoCropsTransform(train_transform),
                                    download=True)
            memory_dataset = CIFAR10(root=data_path, train=True, transform=test_transform, download=True)
            test_dataset = CIFAR10(root=data_path, train=False, transform=test_transform, download=True)
        elif data_type == 'CIFAR100':
            train_dataset = CIFAR100(root=data_path, train=True, transform=TwoCropsTransform(train_transform),
                                     download=True)
            memory_dataset = CIFAR100(root=data_path, train=True, transform=test_transform, download=True)
            test_dataset = CIFAR100(root=data_path, train=False, transform=test_transform, download=True)
        else:
            raise ValueError("only support CIFAR10 and CIFAR100 datasets...")

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=(train_sampler is None), num_workers=workers,
                                                   pin_memory=True, drop_last=True)

        memory_loader = torch.utils.data.DataLoader(memory_dataset, batch_size=batch_size, shuffle=False,
                                                    num_workers=workers, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=workers,
                                                  pin_memory=True)
    else:
        raise ValueError("only support datasets: ImageNet, CIFAR10 and CIFAR100...")

    return train_loader, train_sampler, memory_loader, test_loader


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
