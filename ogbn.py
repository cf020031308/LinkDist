# Usage: python3 ogbn.py [arxiv|mag|products]

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from ogb.nodeproppred import Evaluator, NodePropPredDataset

g_data, *gcard = sys.argv[1:]
gcard = int((gcard or [0])[0])

runs = 10
iterations = 1000
batch_size = 64 * 1024
hid = 256
n_layers = 3

gpu = lambda x: x
if torch.cuda.is_available() and gcard >= 0:
    dev = torch.device('cuda:%d' % gcard)
    gpu = lambda x: x.to(dev)


def optimize(params, lr=0.01):
    if run == 0:
        print('params:', sum(p.numel() for p in params))
    return optim.Adam(params, lr=lr)


FC = (
    lambda din, dout: gpu(nn.Sequential(
        nn.BatchNorm1d(din),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(din, dout)))
) if g_data == 'arxiv' else (
    lambda din, dout: gpu(nn.Sequential(
        nn.ReLU(),
        nn.Linear(din, dout)))
)


class LinkDist(nn.Module):
    def __init__(self, din, hid, dout, n_layers=3):
        super(self.__class__, self).__init__()
        self.inlayer = gpu(nn.Linear(din, hid))
        self.layers = nn.ModuleList()
        for _ in range(n_layers - 2):
            self.layers.append(FC(hid, hid))
        self.outlayer = FC(hid, dout)
        self.inflayer = FC(hid, dout)

    def forward(self, x):
        x = self.inlayer(x)
        for layer in self.layers:
            x = layer(x)
        return self.outlayer(x), self.inflayer(x)


dataname = 'ogbn-%s' % g_data
dataset = NodePropPredDataset(name=dataname)
train_idx, valid_idx, test_idx = map(
    dataset.get_idx_split().get, 'train valid test'.split())
if g_data == 'mag':
    train_idx = train_idx['paper']
    valid_idx = valid_idx['paper']
    test_idx = test_idx['paper']
c = dataset.num_classes
g, labels = dataset[0]
if g_data == 'mag':
    labels = labels['paper']
    g['edge_index'] = g['edge_index_dict'][('paper', 'cites', 'paper')]
    g['node_feat'] = g['node_feat_dict']['paper']
labels = torch.from_numpy(labels)
Y = gpu(labels.clone().squeeze(-1))
Y[valid_idx] = -1
Y[test_idx] = -1
src, dst = torch.from_numpy(g['edge_index'])
e = src.shape[0]
X = gpu(torch.from_numpy(g['node_feat']))
n, d = X.shape

train_nprob = train_idx.shape[0] / n
train_eprob = ((Y[src] >= 0).sum() + (Y[dst] >= 0).sum()).item() / (2 * e)
alpha = 1 - train_eprob
beta = 0.05
beta1 = beta * train_nprob / (train_nprob + train_eprob)
beta2 = beta - beta1

evaluator = Evaluator(name=dataname)
best_metrics = []
smax = lambda x: torch.softmax(x, dim=-1)
nidx = torch.arange(n)
eidx = torch.arange(e)
for run in range(runs):
    torch.manual_seed(run)
    linkdist = LinkDist(d, hid, c, n_layers)
    opt = optimize([*linkdist.parameters()])
    metrics = []
    for iteration in range(1, 1 + iterations):
        linkdist.train()
        opt.zero_grad()
        pidx = nidx[torch.randint(0, n, (batch_size, ))]
        perm = eidx[torch.randint(0, e, (batch_size, ))]
        psrc = src[perm]
        pdst = dst[perm]
        z, s = linkdist(X[pidx])
        z1, s1 = linkdist(X[psrc])
        z2, s2 = linkdist(X[pdst])
        loss = alpha * (
            F.mse_loss(z1, s2) + F.mse_loss(z2, s1)
            - 0.5 * (
                F.mse_loss(smax(z1), smax(s))
                + F.mse_loss(smax(z2), smax(s))
                + F.mse_loss(smax(z), smax(s1))
                + F.mse_loss(smax(z), smax(s2))))
        m = Y[psrc] >= 0
        if m.any().item():
            target = Y[psrc][m]
            loss = loss + (
                F.cross_entropy(z1[m], target)
                + F.cross_entropy(s2[m], target)
                - beta1 * F.cross_entropy(s[m], target))
        m = Y[pdst] >= 0
        if m.any().item():
            target = Y[pdst][m]
            loss = loss + (
                F.cross_entropy(z2[m], target)
                + F.cross_entropy(s1[m], target)
                - beta1 * F.cross_entropy(s[m], target))
        m = Y[pidx] >= 0
        if m.any().item():
            target = Y[pidx][m]
            loss = loss + (
                2 * F.cross_entropy(z[m], target)
                - beta2 * (
                    F.cross_entropy(s1[m], target)
                    + F.cross_entropy(s2[m], target)))
        loss.backward()
        opt.step()
        if iteration % 5:
            continue
        with torch.no_grad():
            linkdist.eval()
            Z = []
            for perm in DataLoader(range(n), batch_size=batch_size):
                z, _ = linkdist(X[perm])
                Z.append(z)
            Z = torch.cat(Z, dim=0)
        Z = Z.max(dim=1, keepdim=True).indices
        metric = [
            evaluator.eval({'y_pred': Z[idx], 'y_true': labels[idx]})['acc']
            for idx in (train_idx, valid_idx, test_idx)]
        # print(run, iteration, *metric)
        metrics.append(metric)
    metrics = torch.tensor(metrics)
    best_metrics.append(metrics[metrics.max(dim=0).indices[1]].tolist())
    print(run, 'best:', best_metrics[-1])
best_metrics = torch.tensor(best_metrics)
print('data:', dataname)
for metric in zip(
        'train valid test'.split(),
        best_metrics.mean(dim=0),
        best_metrics.std(dim=0)):
    print('%s: %.4f±%.4f' % metric)
