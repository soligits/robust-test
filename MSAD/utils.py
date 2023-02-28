import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import torchvision.models as models
import torch.nn.functional as F
from PIL import ImageFilter
import random
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from torchvision.datasets import CIFAR10, MNIST, SVHN, FashionMNIST
import os
from torch.utils.data import Dataset
from PIL import Image
from glob import glob

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])

transform_resnet18 = transforms.Compose([
    transforms.Resize(224, interpolation=BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor()])


transform_color_bw = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.Grayscale(num_output_channels=3),
                                      transforms.ToTensor()])

transform_resnet18_bw = transforms.Compose([
    transforms.Resize(224, interpolation=BICUBIC),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()])


moco_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

moco_transform_bw = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

mu = torch.tensor(mean).view(3,1,1).cuda()
std = torch.tensor(std).view(3,1,1).cuda()

class Transform:
    def __init__(self, bw=False):
        self.moco_transform = moco_transform_bw if bw else moco_transform
    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        return x_1, x_2


class Model(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.norm = lambda x: ( x - mu ) / std
        if backbone == 152:
            self.backbone = models.resnet152(pretrained=True)
        else:
            self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        freeze_parameters(self.backbone, backbone, train_fc=False)

    def forward(self, x):
        x = self.norm(x)
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n

def freeze_parameters(model, backbone, train_fc=False):
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False
    if backbone == 152:
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False
        for p in model.layer1.parameters():
            p.requires_grad = False
        for p in model.layer2.parameters():
            p.requires_grad = False


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def get_loaders(dataset, label_class, batch_size, path, backbone):

    if dataset == 'cifar10':
        return get_CIFAR10(label_class, batch_size, path, backbone)
    elif dataset == 'mnist':
        return get_MNIST(label_class, batch_size, path, backbone)
    elif dataset == 'fashion':
        return get_FASHION_MNIST(label_class, batch_size, path, backbone)
    elif dataset == 'svhn':
        return get_SVHN(label_class, batch_size, path, backbone)
    elif dataset == 'mvtec':
        return get_MVTEC(label_class, batch_size, path, backbone)
    else:
        raise Exception("Dataset is not supported yet. ")
        exit()
        

mvtec_labels = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
                'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
                'wood', 'zipper']


def get_CIFAR10(normal_class_indx, batch_size, path, backbone):
    transform = transform_color if backbone == 152 else transform_resnet18

    trainset = CIFAR10(root=path, train=True, download=True, transform=transform)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    trainset.targets  = [0 for t in trainset.targets]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = CIFAR10(root=path, train=False, download=True, transform=transform)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    trainset_msad = CIFAR10(root=path, train=True, download=True, transform=Transform())
    trainset_msad.data = trainset_msad.data[np.array(trainset_msad.targets) == normal_class_indx]
    trainset_msad.targets  = [0 for t in trainset_msad.targets]
    train_loader_msad = torch.utils.data.DataLoader(trainset_msad, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader, test_loader, train_loader_msad



def get_MNIST(normal_class_indx, batch_size, path, backbone):
    transform = transform_color_bw if backbone == 152 else transform_resnet18_bw

    trainset = MNIST(root=path, train=True, download=True, transform=transform)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    trainset.targets  = [0 for t in trainset.targets]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = MNIST(root=path, train=False, download=True, transform=transform)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    trainset_msad = MNIST(root=path, train=True, download=True, transform=Transform(bw=True))
    trainset_msad.data = trainset_msad.data[np.array(trainset_msad.targets) == normal_class_indx]
    trainset_msad.targets  = [0 for t in trainset_msad.targets]
    train_loader_msad = torch.utils.data.DataLoader(trainset_msad, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader, test_loader, train_loader_msad


def get_FASHION_MNIST(normal_class_indx, batch_size, path, backbone):
    transform = transform_color_bw if backbone == 152 else transform_resnet18_bw

    trainset = FashionMNIST(root=path, train=True, download=True, transform=transform)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    trainset.targets  = [0 for t in trainset.targets]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = FashionMNIST(root=path, train=False, download=True, transform=transform)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    trainset_msad = MNIST(root=path, train=True, download=True, transform=Transform(bw=True))
    trainset_msad.data = trainset_msad.data[np.array(trainset_msad.targets) == normal_class_indx]
    trainset_msad.targets  = [0 for t in trainset_msad.targets]
    train_loader_msad = torch.utils.data.DataLoader(trainset_msad, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader, test_loader, train_loader_msad

def get_SVHN(normal_class_indx, batch_size, path, backbone):
    transform = transform_color if backbone == 152 else transform_resnet18

    trainset = SVHN(root=path, split='train', download=True, transform=transform)
    trainset.data = trainset.data[np.array(trainset.labels) == normal_class_indx]
    trainset.labels  = [0 for t in trainset.labels]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = SVHN(root=path, split='test', download=True, transform=transform)
    testset.labels  = [int(t!=normal_class_indx) for t in testset.labels]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    trainset_msad = SVHN(root=path, split='train', download=True, transform=Transform())
    trainset_msad.data = trainset_msad.data[np.array(trainset_msad.labels) == normal_class_indx]
    trainset_msad.labels  = [0 for t in trainset_msad.labels]
    train_loader_msad = torch.utils.data.DataLoader(trainset_msad, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader, test_loader, train_loader_msad


class MVTecDataset(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, normal=True):
        self.transform = transform
        if train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
          image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
          normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
          anomaly_image_files = list(set(image_files) - set(normal_image_files))
          self.image_files = image_files

        self.image_files.sort(key=lambda y: y.lower())
        self.train = train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1

        return image, target

    def __len__(self):
        return len(self.image_files)


def get_MVTEC(normal_class_indx, batch_size, path, backbone):
    normal_class = mvtec_labels[normal_class_indx]
    transform = transform_color if backbone == 152 else transform_resnet18

    trainset = MVTecDataset(path, normal_class, transform, train=True)
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size)

    testset = MVTecDataset(path, normal_class, transform, train=False)
    test_loader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=batch_size)  

    trainset_msad = MVTecDataset(path, normal_class, Transform(), train=True)
    train_loader_msad = torch.utils.data.DataLoader(trainset_msad, shuffle=True, batch_size=batch_size)

    return train_loader, test_loader, train_loader_msad