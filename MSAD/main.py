import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
import utils
from tqdm import tqdm
import torch.nn.functional as F
from KNN import KnnFGSM, KnnPGD, KnnAdvancedPGD
import gc
import logging
import sys
import os

global Logger
Logger = None

def log(msg):
    global Logger
    Logger.write(f'{msg}\n')
    print(msg)

def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    temp = 0.25
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def train_model(model, train_loader, test_loader, train_loader_1, device, args):
    model.eval()
    auc, feature_space = get_score(model, device, train_loader, test_loader)
    log('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    if args.angular:
        center = F.normalize(center, dim=-1)
    center = center.to(device)
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader_1, optimizer, center, device, args.angular)
        log('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        auc, _ = get_score(model, device, train_loader, test_loader)
        log('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
    

    for test_attack in args.test_attacks:
        log(f'\nStarting the test for {test_attack} ...\n')
        adv_auc, adv_auc_in, adv_auc_out, feature_space = get_adv_score(model, device, train_loader, test_loader, test_attack, eval(args.eps))
        log(f'{test_attack} ADV AUROC is: {adv_auc}')
        log(f'IN: {test_attack} ADV AUROC is: {adv_auc_in}')
        log(f'OUT: {test_attack} ADV AUROC is: {adv_auc_out}')

def run_epoch(model, train_loader, optimizer, center, device, is_angular):
    total_loss, total_num = 0.0, 0
    for ((img1, img2), _) in tqdm(train_loader, desc='Train...'):

        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        out_1 = model(img1)
        out_2 = model(img2)
        out_1 = out_1 - center
        out_2 = out_2 - center

        loss = contrastive_loss(out_1, out_2)

        if is_angular:
            loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())

        loss.backward()

        optimizer.step()

        total_num += img1.size(0)
        total_loss += loss.item() * img1.size(0)

    return total_loss / (total_num)


def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    test_labels = []
    with torch.no_grad():
        for (imgs, labels) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            test_feature_space.append(features)
            test_labels.append(labels)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space

def get_adv_score(model, device, train_loader, test_loader, attack_type, eps):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            train_feature_space.append(features.detach().cpu())
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()

    mean_train = torch.mean(torch.Tensor(train_feature_space), axis=0)

    gc.collect()
    torch.cuda.empty_cache()


    test_attack = None
    if attack_type.startswith('PGDA'):
        steps = int(attack_type.split('-')[1])
        test_attack = KnnAdvancedPGD.PGD_KNN_ADVANCED(model, train_feature_space, eps=eps, steps=steps, alpha = (2.5 * eps) / steps)
    elif attack_type.startswith('PGD'):
        steps = int(attack_type.split('-')[1])
        test_attack = KnnPGD.PGD_KNN(model, mean_train.to(device), eps=eps, steps=steps, alpha = (2.5 * eps) / steps)
    else:
        test_attack = KnnFGSM.FGSM_KNN(model, mean_train.to(device), eps=eps)

    test_adversarial_feature_space = []
    test_adversarial_feature_space_in = []
    test_adversarial_feature_space_out = []
    adv_test_labels = []

    for (imgs, labels) in tqdm(test_loader, desc='Test set adversarial feature extracting'):
        imgs = imgs.to(device)
        labels = labels.to(device)
        adv_imgs, adv_imgs_in, adv_imgs_out, labels= test_attack(imgs, labels)

        adv_test_labels += labels.cpu().numpy().tolist()
        del imgs, labels

        adv_features = model(adv_imgs)
        test_adversarial_feature_space.append(adv_features.detach().cpu())
        del adv_features, adv_imgs

        adv_features_in = model(adv_imgs_in)
        test_adversarial_feature_space_in.append(adv_features_in.detach().cpu())
        del adv_features_in, adv_imgs_in

        adv_features_out = model(adv_imgs_out)
        test_adversarial_feature_space_out.append(adv_features_out.detach().cpu())
        del adv_features_out, adv_imgs_out

        torch.cuda.empty_cache()
            
    test_adversarial_feature_space = torch.cat(test_adversarial_feature_space, dim=0).contiguous().detach().cpu().numpy()
    test_adversarial_feature_space_in = torch.cat(test_adversarial_feature_space_in, dim=0).contiguous().detach().cpu().numpy()
    test_adversarial_feature_space_out = torch.cat(test_adversarial_feature_space_out, dim=0).contiguous().detach().cpu().numpy()

    adv_distances = utils.knn_score(train_feature_space, test_adversarial_feature_space)
    adv_distances_in = utils.knn_score(train_feature_space, test_adversarial_feature_space_in)
    adv_distances_out = utils.knn_score(train_feature_space, test_adversarial_feature_space_out)

    adv_auc = roc_auc_score(adv_test_labels, adv_distances)
    adv_auc_in = roc_auc_score(adv_test_labels, adv_distances_in)
    adv_auc_out = roc_auc_score(adv_test_labels, adv_distances_out)

    del test_adversarial_feature_space, test_adversarial_feature_space_in, test_adversarial_feature_space_out, adv_distances, adv_distances_in, adv_distances_out, adv_test_labels
    gc.collect()
    torch.cuda.empty_cache()

    return adv_auc, adv_auc_in, adv_auc_out, train_feature_space

def main(args):
    log('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(device)
    model = utils.Model(str(args.backbone), args.model_path)
    model = model.to(device)

    train_loader, test_loader, train_loader_1 = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, backbone=args.backbone, path=args.dataset_path)
    train_model(model, train_loader, test_loader, train_loader_1, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--dataset_path', default='~/cifar10', type=str)
    parser.add_argument('--model_path', default='./pretrained_models/resnet50.ckpt', type=str)
    parser.add_argument('--epochs', default=20, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate.')
    parser.add_argument('--eps', type=str, default='2/255', help='The esp for attack.')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--backbone',
                    choices=['resnet18_linf_eps0.5', 'resnet18_linf_eps1.0', 'resnet18_linf_eps2.0', 'resnet18_linf_eps4.0', 'resnet18_linf_eps8.0',
                             'resnet50_linf_eps0.5', 'resnet50_linf_eps1.0', 'resnet50_linf_eps2.0', 'resnet50_linf_eps4.0', 'resnet50_linf_eps8.0',
                             'wide_resnet50_linf_eps0.5', 'wide_resnet50_linf_eps1.0', 'wide_resnet50_linf_eps2.0', 'wide_resnet50_linf_eps4.0', 'wide_resnet50_linf_eps8.0',
                             '18', '50', '152'],
                    default='18',
                    type=str,
                    help='ResNet Backbone')

    parser.add_argument('--test_attacks', help='Desired Attacks for adversarial test', nargs='+')
    parser.add_argument('--angular', action='store_true', help='Train with angular center loss')
    args = parser.parse_args()

    if not os.path.exists('./Results/'):
        os.makedirs('./Results/')

    Logger = open(f"./Results/MSAD-{args.dataset}-{args.label}-epochs{args.epochs}-ResNet{args.backbone}.txt", "a", encoding='utf-8')

    main(args)


