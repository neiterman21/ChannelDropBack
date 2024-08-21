import logging

import torch
import os
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.dataset == 'imagenet':
        imagenet_res = 384
        num_workers = 32
        path = '/home/evgenyn/project/imagenet/'
        traindir = os.path.join(path, 'train')
        valdir = os.path.join(path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_dataset = torchvision.datasets.ImageFolder(traindir,transform =  transforms.Compose([
                                                                    
                                                                    #transforms.RandomCrop(imagenet_res, padding=4),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.RandomRotation(15),
                                                                    transforms.Resize((imagenet_res,imagenet_res)),
                                                                    transforms.ToTensor(),
                                                                    normalize]))
        #train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_dataset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=num_workers, pin_memory=True,shuffle=True)
        val_dateset = torchvision.datasets.ImageFolder(valdir,transform =  transforms.Compose([transforms.Resize((imagenet_res,imagenet_res)),
                                                        transforms.ToTensor(),
                                                        normalize]))
        #val_sampler =torch.utils.data.distributed.DistributedSampler(dataset=val_dateset, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_dateset,
                                                    batch_size=args.train_batch_size, shuffle=True,
                                                    num_workers=num_workers, pin_memory=False)
        return train_loader , val_loader

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
