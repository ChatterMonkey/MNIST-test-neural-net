import torch
import torchvision
from variables import *

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/Users/mayabasu/pytorchdatasets', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=train_batch_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/Users/mayabasu/pytorchdatasets', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=test_batch_size, shuffle=False)
