#
# File:   mnist.py
# Author: Artem Suprun
# Date:   06/05/2026
# Description: A group of functions that help handle importing and 
#  modifying the data for training and testing.
#


from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataset(transform=None):
    if transform is not None:
        transform = transform
    else:
        transform = transforms.ToTensor()
    train_set = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    test_set = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return train_set, test_set


def create_mnist_loader(train_dataset, test_dataset, batch_size=128, shuffle=True):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_mnist_loader(batch_size=128, shuffle=True, transform=None):
    train_set, test_set = get_mnist_dataset(transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

