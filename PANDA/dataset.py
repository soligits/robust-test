from torchvision.datasets import CIFAR10, MNIST, SVHN, FashionMNIST
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from glob import glob

transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_gray = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
#######################
#  Define Transform   #
#######################

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)



def get_dataloader(dataset='cifar', normal_class_indx = 0, batch_size=8):
    if dataset == 'cifar10':
        return get_CIFAR10(normal_class_indx, batch_size)
    elif dataset == 'mnist':
        return get_MNIST(normal_class_indx, batch_size)
    elif dataset == 'fashion':
        return get_FASHION_MNIST(normal_class_indx, batch_size)
    elif dataset == 'svhn':
        return get_SVHN(normal_class_indx, batch_size)
    else:
        raise Exception("Dataset is not supported yet. ")


def get_CIFAR10(normal_class_indx, batch_size):
    trainset = CIFAR10(root=os.path.join('~', 'cifar10'), train=True, download=True, transform=transform_color)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    trainset.targets  = [0 for t in trainset.targets]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = CIFAR10(root=os.path.join('~/', 'cifar10'), train=False, download=True, transform=transform_color)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader



def get_MNIST(normal_class_indx, batch_size):
    
    trainset = MNIST(root=os.path.join('~', 'mnist'), train=True, download=True, transform=transform_gray)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    trainset.targets  = [0 for t in trainset.targets]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = MNIST(root=os.path.join('~', 'mnist'), train=False, download=True, transform=transform_gray)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_FASHION_MNIST(normal_class_indx, batch_size):

    trainset = FashionMNIST(root=os.path.join('~', 'fashion-mnist'), train=True, download=True, transform=transform_gray)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    trainset.targets  = [0 for t in trainset.targets]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = FashionMNIST(root=os.path.join('~', 'fashion-mnist'), train=False, download=True, transform=transform_gray)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_SVHN(normal_class_indx, batch_size):

    trainset = SVHN(root=os.path.join('~', 'SVHN'), split='train', download=True, transform=transform_color)
    trainset.data = trainset.data[np.array(trainset.labels) == normal_class_indx]
    trainset.labels  = [0 for t in trainset.labels]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = SVHN(root=os.path.join('~', 'SVHN'), split='test', download=True, transform=transform_color)
    testset.labels  = [int(t!=normal_class_indx) for t in testset.labels]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

