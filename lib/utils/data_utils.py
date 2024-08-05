# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import os
import numpy as np

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import listDataset
from PascalLoader import DataLoader
import torchvision

def get_dataset(dataset_name, batch_size, n_worker, data_root='data/imagenet', for_inception=False):
    print('==> Preparing data..')
    if dataset_name == 'imagenet':
        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        assert os.path.exists(traindir), traindir + ' not found'
        assert os.path.exists(valdir), valdir + ' not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                traindir, transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size, shuffle=True,
            num_workers=n_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=n_worker, pin_memory=True)

        n_class = 1000
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
            # transforms.Scale(256),
            # transforms.CenterCrop(227),
            transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        # DataLoader initialize
        train_data = DataLoader(data_root, 'trainval', transform=train_transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=n_worker)

        val_data = DataLoader(data_root, 'test', transform=val_transform, random_crops=10)
        val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=n_worker)
        n_class = 21
    return train_loader, val_loader, n_class


def get_split_train_dataset(dataset_name, batch_size, n_worker, val_size, train_size=None, random_seed=1,
                            data_root='data/imagenet', for_inception=False, shuffle=True):
    """
    generate train and val data loader
    Args:
        dataset_name: imagenet or others
        batch_size: 60 on default, size of each sampled batch
        n_worker: 4 on defalult
        val_size: for imagenet, 10000
        train_size: for imagenet, 20000
        random_seed: 2019,
        data_root: data set path
        for_inception: false
        shuffle: True, is shuffle befure sample

    Returns:
        train_loader: dataloader to get shuffled, sampled dataset for generating FM
        val_loader： dataloader to get shuffled, sampled dataset for evaluation
        n_class： class for classification job, in detections should be the according GT?
    """
    if shuffle:
        index_sampler = SubsetRandomSampler
    else:
        # use the same order
        class SubsetSequentialSampler(SubsetRandomSampler):
            def __iter__(self):
                return (self.indices[i] for i in torch.arange(len(self.indices)).int())
        index_sampler = SubsetSequentialSampler

    print('==> Preparing data..')
    if dataset_name == 'imagenet':

        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')# todo: change back to val!
        assert os.path.exists(traindir), traindir + ' not found'
        assert os.path.exists(valdir), valdir + ' not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        test_transform = transforms.Compose([
                transforms.Resize(int(input_size/0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])

        trainset = datasets.ImageFolder(traindir, train_transform)
        valset = datasets.ImageFolder(valdir, test_transform)
        # use real val
        n_val = len(valset)
        indices = list(range(n_val))
        # shuffle the indices
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        assert val_size <= n_val
        indices = list(range(n_val))
        np.random.shuffle(indices)
        _, val_idx = indices[val_size:], indices[:val_size]
        train_idx = list(range(len(trainset)))  # all trainset

        #train_sampler = index_sampler(train_idx)
        _, train_idx = indices[train_size:], indices[:train_size]
        val_sampler = index_sampler(val_idx)
        train_sampler = index_sampler(train_idx)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=True, drop_last=True) # add drop last
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=True)
        n_class = 1000
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
            # transforms.Scale(256),
            # transforms.CenterCrop(227),
            transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        # DataLoader initialize
        pascal_path = data_root
        train_data = DataLoader(pascal_path, 'trainval', transform=train_transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=n_worker)

        val_data = DataLoader(pascal_path, 'test', transform=val_transform, random_crops=10)
        val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=n_worker)
        n_class = 21

    return train_loader, val_loader, n_class

def get_data_loaders(data_root, batch_size, n_data_worker, use_cuda=True):
    """
    data_root：数据集路径
    batch_size：一个batch的图片个数，默认为32
    num_workers：取数据线程数，默认为4
    use_cuda：是否使用了cuda，默认为True
    """
    train_list = os.path.join(data_root, 'train_list.txt')
    val_list = os.path.join(data_root, 'val_list.txt')
    assert os.path.exists(train_list), train_list + ' not found'
    assert os.path.exists(val_list), val_list + ' not found'
    kwargs = {'num_workers': n_data_worker, 'pin_memory': True} if use_cuda else {}
    train_dataset = listDataset(train_list, shape=(320, 160),
                           shuffle=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
                           ]),
                           train=True,
                           seen=0,
                           batch_size=batch_size,
                           num_workers=n_data_worker)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    val_dataset = listDataset(val_list, shape=(320, 160),
                              shuffle=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]), ]),
                              train=False,
                              batch_size=batch_size,
                              num_workers=n_data_worker)

    # 为了后续采样，这里shuffle设置为True，每次取都随机
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, val_loader


def get_voc_data_loaders(pascal_path,batch, num_workers, random_crops):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        # transforms.Scale(256),
        # transforms.CenterCrop(227),
        transforms.RandomResizedCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_data = DataLoader(pascal_path, 'trainval', transform=train_transform)
    N = len(train_data.names)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch,
                                               shuffle=True,
                                               num_workers=num_workers)

    val_data = DataLoader(pascal_path, 'test', transform=val_transform, random_crops=random_crops)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=batch,
                                             shuffle=False,
                                             num_workers=num_workers)
    return train_loader, val_loader, N

def get_mnist_data_loaders(batch):
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                          ]))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                         ]))
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=2)
    return trainloader, testloader

def get_cifar_data_loaders(batch):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=2)
    return trainloader, testloader