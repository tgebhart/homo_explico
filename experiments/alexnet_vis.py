import os
import torch
import matplotlib.pyplot as plt

from homo_explico.models.alexnet import AlexNet
from homo_explico.functions.deep_dream import DeepDream

from torchvision import datasets, transforms


def run():

    model = AlexNet()
    model.load_state_dict(torch.load('/home/tgebhart/projects/homo_explico/logdir/models/alexnet_cifar.pt'))

    test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../../data/cifar', train=False, download=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])), batch_size=1, shuffle=False)

    d, t = None, None
    ioi = 1
    iii = 0
    for data, target in test_loader:
        if iii == ioi:
            d = data
            t = target
            break
        iii += 1

    dd = DeepDream(model, d)
    dd.dream(percentile=90)


if __name__ == "__main__":
    run()
