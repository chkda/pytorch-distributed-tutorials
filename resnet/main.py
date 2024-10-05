import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as transforms

import argparse
import os
import random
import numpy as np

def set_random_seeds(seed=0):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def evaluate(model, device, test_loader):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc


def main():

    defaults = {
        "num_epochs":10000,
        "batch_size":256,
        "lr":0.01,
        "seed":0,
        "model_dir":"saved_models",
        "filename":"resnet_distributed.pth"
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int, help="Local rank. necessary for using torch.distributed.launch")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs", default=defaults["num_epochs"])
    parser.add_argument("--batch-size", type=int, help="Training batch size", default=defaults["batch_size"])
    parser.add_argument("--learning_rate", type=int, help="Learning rate", default=defaults["lr"])
    parser.add_argument("--seed", type=int, help="Random seed for training", default=defaults["seed"])
    parser.add_argument("--model_dir", type=str, help="Model directory to store saved models", default=defaults["model_dir"])
    parser.add_argument("--model_filename", type=str, help="Model filename to be saved", default=defaults["model_filename"])
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")

    argv = parser.parse_args()
    local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    lr = argv.learning_rate
    seed = argv.seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename
    resume = argv.resume

    model_filepath = os.path.join(model_dir, model_filename)
    set_random_seeds(seed)

    torch.distributed.init_process_group(backend="nccl")

    model = torchvision.models.resnet18(pretrained=False)

    device = torch.device("cuda:{}".format(local_rank))
    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)


    if resume:
        map_location = {"cuda:0":"cuda:{}".format(local_rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root="data", train=True, download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="data", train=False, download=False, transform=transform)

    train_sampler = DistributedSampler(dataset=train_set)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=8)

    test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=8)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

    for epoch in range(num_epochs):

        print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))

        if epoch % 10 == 0:
            if local_rank == 0:
                acc = evaluate(model=ddp_model, device=device, test_loader)
                torch.save(ddp_model.state_dict(), model_filepath)
                print("-" * 75)
                print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
                print("-" * 75)

        ddp_model.train()
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
    