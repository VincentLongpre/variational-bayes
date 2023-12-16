import torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, IterableDataset

def mnist_dataloaders(data_root, batch_size, image_size=32):
    normalize = transforms.Normalize((0.5,), (0.5,))
    transform = transforms.Compose((
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize))

    train = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_root, train=False, download=True, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    return train_dataloader, test_dataloader

class ThresholdTransform(object):
  def __init__(self, thr_255):
    self.thr = thr_255 / 255.

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)

def binary_mnist_dataloaders(data_root, batch_size, image_size=32):
    transform = transforms.Compose((
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        ThresholdTransform(thr_255=127)))

    train = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_root, train=False, download=True, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    return train_dataloader, test_dataloader

class IterDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()

def generator():
    while True:
        yield F.one_hot(torch.tensor([np.random.randint(0,4)]), 4).float().view(2,2)

def toy_dataloader(batch_size):
    dataset = IterDataset(generator)
    dataloader = iter(DataLoader(dataset, batch_size=batch_size))
    return dataloader
