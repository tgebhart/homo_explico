import os
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import dionysus as dion
import networkx as nx

import numpy as np
import pandas as pd

from persistence_dropout.functions.filtration import conv_filtration, linear_filtration, conv_layer_as_matrix, conv_filtration_static, linear_filtration_static, conv_filtration_fast, linear_filtration_fast


class CFF(nn.Module):
    def __init__(self, filters=5, kernel_size=5, fc1_size=50):
        super(CFF, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.fc1_size = fc1_size
        self.stride = 1
        self.activation = 'relu'
        self.conv1 = nn.Conv2d(1, self.filters, kernel_size=self.kernel_size, bias=False, stride=1)
        self.fc1 = nn.Linear(((28-self.kernel_size+1)**2)*self.filters, self.fc1_size, bias=False)
        self.fc2 = nn.Linear(self.fc1_size, 10, bias=False)
        self.layer_types = ['convolution', 'fully-connected', 'fully-connected']
        self.input_size = 28

    def forward(self, x, muls=None, hiddens=False):

        h1_m = F.relu(self.conv1(x))
        h1 = h1_m.view(-1, (28-self.kernel_size+1)**2*self.filters)
        h2 = F.relu(self.fc1(h1))
        y = self.fc2(h2)
        if hiddens:
            return F.log_softmax(y, dim=1), [h1, h2, y]
        return F.log_softmax(y, dim=1)


    def hidden_forward(self,x):
        h1_m = F.relu(self.conv1(x))
        h1 = h1_m.view(-1, (28-self.kernel_size+1)**2*self.filters)
        h2 = F.relu(self.fc1(h1))
        y = self.fc2(h2)
        return F.log_softmax(y, dim=1), [h1, h2, y]


    def save_string(self, dataset):
        return "cff_relu_{}.pt".format(dataset)

    def layerwise_ids(self, input_size=28*28):
        l1_size = (28-self.kernel_size+1)**2*self.filters
        l1_end = input_size+l1_size
        l2_end = l1_end+self.fc1_size
        l3_end = l2_end + 10
        return [range(input_size), range(input_size, l1_end), range(l1_end, l2_end), range(l2_end, l3_end)]

    def compute_static_filtration(self, x, hiddens, percentile=None):
        f = dion.Filtration()

        h1_id_start = x.cpu().detach().numpy().reshape(-1).shape[0]
        f, h1_births = conv_filtration_static(f, x[0], self.conv1.weight.data[:,0,:,:], 0, h1_id_start, percentile=percentile)

        h2_id_start = h1_id_start + hiddens[0].cpu().detach().numpy().shape[0]
        f, h2_births = linear_filtration_static(f, hiddens[0], self.fc1, h1_births, h1_id_start, h2_id_start, percentile=percentile, last=False)

        h3_id_start = h2_id_start + hiddens[1].cpu().detach().numpy().shape[0]
        f = linear_filtration_static(f, hiddens[1], self.fc2, h2_births, h2_id_start, h3_id_start, percentile=percentile, last=True)

        # print('filtration size', len(f))
        f.sort(reverse=True)
        return f


    def compute_induced_filtration(self, x, hiddens, percentile=None, mat=None):

        if mat is None:
            mat = conv_layer_as_matrix(self.conv1.weight.data[:,0,:,:], x[0], self.stride)

        h1_id_start = x.cpu().detach().numpy().reshape(-1).shape[0]
        m1, h0_births, h1_births, percentile_1 = conv_filtration_fast(x[0], mat, 0, h1_id_start, percentile=percentile)
        enums = m1
        enums += [([i], h0_births[i]) for i in np.argwhere(h0_births > percentile_1)]

        h2_id_start = h1_id_start + hiddens[0].cpu().detach().numpy().shape[0]
        m2, h1_births_2, h2_births, percentile_2 = linear_filtration_fast(hiddens[0], self.fc1, h1_id_start, h2_id_start, percentile=percentile)
        enums += m2

        max1 = np.maximum.reduce([h1_births, h1_births_2])
        comp_percentile = percentile_1 if percentile_1 < percentile_2 else percentile_2
        enums += [([i+h1_id_start], max1[i]) for i in np.argwhere(max1 > comp_percentile)]

        h3_id_start = h2_id_start + hiddens[1].cpu().detach().numpy().shape[0]
        m3, h2_births_2, h3_births, percentile_3 = linear_filtration_fast(hiddens[1], self.fc2, h2_id_start, h3_id_start, percentile=percentile)
        enums += m3

        max2 = np.maximum.reduce([h2_births, h2_births_2])
        comp_percentile = percentile_2 if percentile_2 < percentile_3 else percentile_3
        enums += [([i+h2_id_start], max2[i]) for i in np.argwhere(max2 > comp_percentile)]

        enums += [([i+h3_id_start], h3_births[i]) for i in np.argwhere(h3_births > percentile_3)]

        f = dion.Filtration(enums)
        f.sort(reverse=True)

        return f

    def compute_layer_mask(self, x, hiddens, subgraph=0, percentile=0):
        mat = conv_layer_as_matrix(self.conv1.weight.data[:,0,:,:], x[0][0], self.stride)
        f = self.compute_induced_filtration(x, hiddens, percentile=percentile, mat=mat)
        m = dion.homology_persistence(f)
        dgms = dion.init_diagrams(m,f)
        subgraphs = {}

        # compute ones tensors of same hidden dimensions
        muls = []

        for h in hiddens:
            muls.append(torch.zeros(h.shape))

        fac = 1.0

        for i,c in enumerate(m):
            if len(c) == 2:
                if f[c[0].index][0] in subgraphs:
                    subgraphs[f[c[0].index][0]].add_edge(f[c[0].index][0],f[c[1].index][0],weight=f[i].data)
                else:
                    eaten = False
                    for k, v in subgraphs.items():
                        if v.has_node(f[c[0].index][0]):
                            v.add_edge(f[c[0].index][0], f[c[1].index][0], weight=f[i].data)
                            eaten = True
                            break
                    if not eaten:
                        g = nx.Graph()
                        g.add_edge(f[c[0].index][0], f[c[1].index][0], weight=f[i].data)
                        subgraphs[f[c[0].index][0]] = g

        #  I don't think we need this composition. Disjoint subgraphs are fine.
        # subgraph = nx.compose_all([subgraphs[k] for k in list(subgraphs.keys())[:thru]])

        k = list(subgraphs.keys())[subgraph]
        # lifetimes = np.empty(thru)
        # t = 0
        # for pt in dgms[0]:
        #     if pt.death < float('inf'):
        #         lifetimes[t] = pt.birth - pt.death
        #         t += 1
        #     if t >= thru:
        #         break
        # max_lifetime = max(lifetimes)
        # min_lifetime = min(lifetimes)
        ids = self.layerwise_ids()
        layer_types = self.layer_types

        for e in subgraphs[k].edges(data=True):
            for l in range(len(ids)-1):
                if e[0] in ids[l] and e[1] in ids[l+1]:
                    muls[l][e[1]-ids[l+1][0]] = fac

        return muls



def train(args, model, device, train_loader, optimizer, epoch, percentile=0.0):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('-m', '--model-directory', type=str, required=True,
                        help='location to store trained model')
    parser.add_argument('-d', '--diagram-directory', type=str, required=False,
                        help='location to store homology info')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--up-to', type=int, default=500, metavar='N',
                        help='How many testing exmaples for creating diagrams')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-da', '--dataset', type=str, required=True,
                        help='which dataset to train on (mnist or fashionmnist)')
    parser.add_argument('-p', '--percentile', type=float, default=0,
                        help='Filtration threshold percentile')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])), batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])), batch_size=args.test_batch_size, shuffle=False, **kwargs)


    if args.dataset == 'fashion':
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data/fashion', train=True, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])), batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data/fashion', train=False, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])), batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = CFF().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    res_df = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, percentile=args.percentile)
        test(args, model, device, test_loader)


    save_path = os.path.join(args.model_directory, model.save_string(args.dataset))
    torch.save(model.state_dict(), save_path)

    if args.diagram_directory is not None and args.create_diagrams:
        create_diagrams(args, model)

if __name__ == '__main__':
    main()
