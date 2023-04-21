import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
import utils
import torchattacks
import torch
from tqdm import tqdm


def train_model(model, train_loader, outliers_loader, test_loader, device, epochs, lr):
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    ce = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = run_epoch(
            model, train_loader, outliers_loader, optimizer, ce, device
        )
        print("Epoch: {}, Loss: {}".format(epoch + 1, running_loss))
        auc = get_score(model, test_loader, device)
        print("Epoch: {}, AUROC is: {}".format(epoch + 1, auc))


def run_epoch(model, train_loader, outliers_loader, optimizer, ce, device):
    running_loss = 0.0
    for i, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)

        out_imgs, _ = next(iter(outliers_loader))

        outlier_im = out_imgs.to(device)

        optimizer.zero_grad()

        pred = model(imgs)
        outlier_pred = model(outlier_im)

        batch_1 = pred.size()[0]
        batch_2 = outlier_pred.size()[0]

        labels = torch.zeros(size=(batch_1 + batch_2,), device=device, dtype=torch.long)
        labels[batch_1:] = torch.ones(size=(batch_2,))

        loss = ce(torch.cat([pred, outlier_pred]), labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / (i + 1)


def get_score(model, test_loader, device):
    is_train = model.training
    model.eval()

    soft = torch.nn.Softmax(dim=1)
    anomaly_scores = []
    preds = []
    test_labels = []

    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            torch.cuda.empty_cache()
            for i, (data, target) in enumerate(tepoch):
                data, target = data.to(device), target.to(device)
                target = target.type(torch.LongTensor).cuda()
                output = model(data)

                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                preds += predictions.detach().cpu().numpy().tolist()

                probs = soft(output).squeeze()
                anomaly_scores += probs[:, -1].detach().cpu().numpy().tolist()

                target = target == 1

                test_labels += target.detach().cpu().numpy().tolist()

    auc = roc_auc_score(test_labels, anomaly_scores)

    if is_train:
        model.train()
    else:
        model.eval()

    return auc


def get_score_adversarial(model, test_loader, test_attack, device):
    is_train = model.training
    model.eval()

    soft = torch.nn.Softmax(dim=1)
    anomaly_scores = []
    preds = []
    test_labels = []

    with tqdm(test_loader, unit="batch") as tepoch:
        torch.cuda.empty_cache()
        for i, (data, target) in enumerate(tepoch):
            data, target = data.to(device), target.to(device)

            adv_data = test_attack(data, target)
            output = model(adv_data)

            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            preds += predictions.detach().cpu().numpy().tolist()

            probs = soft(output).squeeze()
            anomaly_scores += probs[:, -1].detach().cpu().numpy().tolist()

            target = target == 1

            test_labels += target.detach().cpu().numpy().tolist()

    auc = roc_auc_score(test_labels, anomaly_scores)

    if is_train:
        model.train()
    else:
        model.eval()

    return auc


def main(args):
    print("Dataset: {}, Label: {}, LR: {}".format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = utils.get_resnet_model(resnet_type=args.resnet_type)

    # Change last layer
    model.fc = torch.nn.Linear(args.latent_dim_size, 2)

    model = model.to(device)
    utils.freeze_parameters(model, train_fc=True)

    train_loader, test_loader = utils.get_loaders(
        dataset=args.dataset,
        label_class=args.label,
        batch_size=args.batch_size,
        path="./",
    )
    outliers_loader = utils.get_outliers_loader(args.batch_size)

    train_model(
        model, train_loader, outliers_loader, test_loader, device, args.epochs, args.lr
    )

    print(get_score(model, test_loader=test_loader, device=device))
    utils.unfreeze_parameters(model)
    test_attack = torchattacks.PGD(model, steps=10, eps=8 / 255, alpha=2 / 255)
    print(get_score_adversarial(model, test_loader, test_attack, device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument(
        "--outlier", default="tiny", choices=["tiny", "SVHN", "MNIST", "gaussian_noise"]
    )
    parser.add_argument(
        "--epochs", default=50, type=int, metavar="epochs", help="number of epochs"
    )
    parser.add_argument("--label", default=0, type=int, help="The normal class")
    parser.add_argument(
        "--lr", type=float, default=1e-1, help="The initial learning rate."
    )
    parser.add_argument(
        "--resnet_type", default=152, type=int, help="which resnet to use"
    )
    parser.add_argument("--latent_dim_size", default=2048, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    args = parser.parse_args()

    main(args)
