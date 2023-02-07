import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
from losses import CompactnessLoss, EWCLoss
import utils
from copy import deepcopy
from tqdm import tqdm
from KNN import KnnFGSM, KnnPGD

# KnnPGD.PGD_KNN(model, mean_train.to(device), eps=attack.eps, steps=attack.steps)
# test_features += model.get_feature_vector(inputs).detach().cpu().numpy().tolist()
# test_labels_normal += labels.detach().cpu().numpy().tolist()
# adv_inputs, labels, _, __ = test_attack(inputs, labels)
# adv_test_features += model.get_feature_vector(adv_inputs).detach().cpu().numpy().tolist()

def train_model(model, train_loader, test_loader, device, args, ewc_loss):
    model.eval()
    adv_auc, auc, feature_space = get_score(model, device, train_loader, test_loader, args.attack_type)
    print('Epoch: {}, AUROC is: {}, ADV AUROC is: {}'.format(0, auc, adv_auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader, optimizer, criterion, device, args.ewc, ewc_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        adv_auc, auc, feature_space = get_score(model, device, train_loader, test_loader, args.attack_type)
        print('Epoch: {}, AUROC is: {}, ADV AUROC is: {}'.format(epoch + 1, auc, adv_auc))


def run_epoch(model, train_loader, optimizer, criterion, device, ewc, ewc_loss):
    running_loss = 0.0
    for i, (imgs, _) in enumerate(train_loader):

        images = imgs.to(device)

        optimizer.zero_grad()

        _, features = model(images)

        loss = criterion(features)

        if ewc:
            loss += ewc_loss(model)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / (i + 1)


def get_score(model, device, train_loader, test_loader, attack_type):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()

    mean_train = torch.mean(torch.Tensor(train_feature_space), axis=0)

    test_feature_space = []
    test_adversarial_feature_space = []

    test_attack = None
    if attack_type == 'PGD':
        test_attack = KnnPGD.PGD_KNN(model, mean_train.to(device), eps=2/255, steps=10)
    else:
        test_attack = KnnFGSM.FGSM_KNN(model, mean_train.to(device), eps=2/255)

    with torch.no_grad():
        for (imgs, _) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            test_feature_space.append(features)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = test_loader.dataset.targets

    for (imgs, labels) in tqdm(test_loader, desc='Test set adversarial feature extracting'):
        imgs = imgs.to(device)
        labels = labels.to(device)
        adv_imgs = test_attack(imgs, labels)
        _, adv_features = model(imgs)
        test_adversarial_feature_space.append(adv_features)
    
    test_adversarial_feature_space = torch.cat(test_adversarial_feature_space, dim=0).contiguous().cpu().numpy()

    distances = utils.knn_score(train_feature_space, test_feature_space)
    adv_distances = utils.knn_score(train_feature_space, test_adversarial_feature_space)

    auc = roc_auc_score(test_labels, distances)
    adv_auc = roc_auc_score(test_labels, adv_distances)

    return adv_auc, auc, train_feature_space

def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = utils.get_resnet_model(resnet_type=args.resnet_type)
    model = model.to(device)

    ewc_loss = None

    # Freezing Pre-trained model for EWC
    if args.ewc:
        frozen_model = deepcopy(model).to(device)
        frozen_model.eval()
        utils.freeze_model(frozen_model)
        fisher = torch.load(args.diag_path)
        ewc_loss = EWCLoss(frozen_model, fisher)

    utils.freeze_parameters(model)
    train_loader, test_loader = utils.get_loaders(dataset=args.dataset, path=args.dataset_path, label_class=args.label, batch_size=args.batch_size)
    train_model(model, train_loader, test_loader, device, args, ewc_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--dataset_path', default='~/cifar10', type=str)
    parser.add_argument('--diag_path', default='./data/fisher_diagonal.pth', help='fim diagonal path')
    parser.add_argument('--ewc', action='store_true', help='Train with EWC')
    parser.add_argument('--epochs', default=15, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
    parser.add_argument('--resnet_type', default=152, type=int, help='which resnet to use')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--attack_type', default='PGD', type=str)

    args = parser.parse_args()

    main(args)
