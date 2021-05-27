import sys
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl


if len(sys.argv) > 2:
    g_method, g_data, *gcard = sys.argv[1:]
    gcard.append('0')
else:
    g_method = 'lpa'
    g_data = 'cora'
    gcard = [0]
epochs = 200
batch_size = 1024
hid = 256

gcard = int(gcard[0])
gpu = lambda x: x
if torch.cuda.is_available() and gcard >= 0:
    dev = torch.device('cuda:%d' % gcard)
    gpu = lambda x: x.to(dev)


def optimize(params, lr=0.01):
    if run == 0:
        print('params:', sum(p.numel() for p in params))
    return optim.Adam(params, lr=lr)


def speye(n):
    return torch.sparse_coo_tensor(
        torch.arange(n).view(1, -1).repeat(2, 1), [1] * n)


def spnorm(A, eps=1e-5):
    D = (torch.sparse.sum(A, dim=1).to_dense() + eps) ** -1
    indices = A._indices()
    return gpu(torch.sparse_coo_tensor(indices, D[indices[0]], size=A.size()))


def FC(din, dout):
    return gpu(nn.Sequential(
        nn.BatchNorm1d(din),
        nn.LayerNorm(din),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(din, dout)))


class MLP(nn.Module):
    def __init__(self, din, hid, dout, n_layers=3, A=None):
        super(self.__class__, self).__init__()
        self.A = A
        self.layers = nn.ModuleList()
        self.layers.append(gpu(nn.Linear(din, hid)))
        for _ in range(n_layers - 2):
            self.layers.append(FC(hid, hid))
        self.layers.append(FC(hid, dout))

    def forward(self, x):
        for layer in self.layers:
            if self.A is not None:
                x = self.A @ x
            x = layer(x)
        return x


GCN = MLP


class LinkDist(nn.Module):
    def __init__(self, din, hid, dout, n_layers=3):
        super(self.__class__, self).__init__()
        self.mlp = MLP(din, hid, hid, n_layers=n_layers - 1)
        self.out = FC(hid, dout)
        self.inf = FC(hid, dout)

    def forward(self, x):
        x = self.mlp(x)
        return self.out(x), self.inf(x)


if g_data == 'ppi':
    graphs = {}
    stat = defaultdict(lambda: 0)
    modes = ('train', 'valid', 'test')
    for mode in modes:
        graphs[mode] = []
        for graph in dgl.data.PPIDataset(mode=mode):
            X = node_features = gpu(graph.ndata['feat'])
            Y = node_labels = gpu(graph.ndata['label'])
            src, dst = graph.edges()
            n_nodes = node_features.shape[0]
            n_edges = src.shape[0]
            n_features = node_features.shape[1]
            n_labels = node_labels.shape[1]
            if 'linkdist' in g_method:
                A = spnorm(graph.adj())
            else:
                A = spnorm(graph.adj() + speye(n_nodes), eps=0)
            graphs[mode].append((
                X, Y, src, dst, n_nodes, n_edges, n_features, n_labels, A))
            stat['nodes'] += n_nodes
            stat['edges'] += (n_edges - (src == dst).sum().item())
            is_bidir = ((dst == src[0]) & (src == dst[0])).any().item()
    print('nodes: %d' % stat['nodes'])
    print('features: %d' % n_features)
    print('classes: %d' % n_labels)
    print('edges: %d' % (stat['edges'] / (1 + is_bidir)))
    degree = stat['edges'] * (2 - is_bidir) / stat['nodes']
    print('degree: %.2f' % degree)
else:
    graph = (
        dgl.data.CoraGraphDataset() if g_data == 'cora'
        else dgl.data.CiteseerGraphDataset() if g_data == 'citeseer'
        else dgl.data.PubmedGraphDataset() if g_data == 'pubmed'
        else dgl.data.CoraFullDataset() if g_data == 'corafull'
        else dgl.data.CoauthorCSDataset() if g_data == 'coauthor-cs'
        else dgl.data.CoauthorPhysicsDataset() if g_data == 'coauthor-phy'
        else dgl.data.RedditDataset() if g_data == 'reddit'
        else dgl.data.AmazonCoBuyComputerDataset()
        if g_data == 'amazon-com'
        else dgl.data.AmazonCoBuyPhotoDataset() if g_data == 'amazon-photo'
        else None
    )[0]
    X = node_features = gpu(graph.ndata['feat'])
    Y = node_labels = gpu(graph.ndata['label'])
    n_nodes = node_features.shape[0]
    nrange = torch.arange(n_nodes)
    n_features = node_features.shape[1]
    n_labels = int(Y.max().item() + 1)
    src, dst = graph.edges()
    n_edges = src.shape[0]
    is_bidir = ((dst == src[0]) & (src == dst[0])).any().item()
    print('BiDirection: %s' % is_bidir)
    print('nodes: %d' % n_nodes)
    print('features: %d' % n_features)
    print('classes: %d' % n_labels)
    print('edges: %d' % (
        (n_edges - (src == dst).sum().item()) / (1 + is_bidir)))
    degree = n_edges * (2 - is_bidir) / n_nodes
    print('degree: %.2f' % degree)


class Stat(object):
    def __init__(self, name=''):
        self.name = name
        self.accs = []
        self.times = []
        self.best_accs = []
        self.best_times = []

    def __call__(self, logits, startfrom=0):
        self.accs.append([
            ((logits[mask].max(dim=1).indices == Y[mask]).sum()
             / gpu(mask).sum().float()).item()
            for mask in (train_mask, valid_mask, test_mask)
        ])
        self.times.append(time.time() - self.tick)
        return self.accs[-1][0]

    def evppi(self, rets):
        self.accs.append([
            torch.cat([
                ((torch.sigmoid(logits) > 0.5) == Y.bool())
                .float().mean(dim=1)
                for logits, Y in ret
            ], dim=0).mean()
            for ret in rets])
        self.times.append(time.time() - self.tick)
        return self.accs[-1][0]

    def start_run(self):
        self.tick = time.time()

    def end_run(self):
        self.accs = torch.tensor(self.accs)
        print('best:', self.accs.max(dim=0).values)
        idx = self.accs.max(dim=0).indices[1]
        self.best_accs.append((idx, self.accs[idx, 2]))
        self.best_times.append(self.times[idx])
        self.accs = []
        self.times = []
        print('best:', self.best_accs[-1])

    def end_all(self):
        conv = 1.0 + torch.tensor([idx for idx, _ in self.best_accs])
        acc = 100 * torch.tensor([acc for _, acc in self.best_accs])
        tm = torch.tensor(self.best_times)
        print(self.name)
        print('time:%.3f±%.3f' % (tm.mean().item(), tm.std().item()))
        print('conv:%.3f±%.3f' % (conv.mean().item(), conv.std().item()))
        print('acc:%.2f±%.2f' % (acc.mean().item(), acc.std().item()))


evaluate = Stat(name='data: %s, method: %s' % (g_data, g_method))
for run in range(10):
    torch.manual_seed(run)
    if g_data != 'ppi':
        if 'train_mask' in graph.ndata:
            train_mask = graph.ndata['train_mask']
            valid_mask = graph.ndata['val_mask']
            test_mask = graph.ndata['test_mask']
        else:
            # split dataset like Cora, Citeseer and Pubmed
            train_mask = torch.zeros(n_nodes, dtype=bool)
            for y in range(n_labels):
                label_mask = (graph.ndata['label'] == y)
                train_mask[
                    nrange[label_mask][torch.randperm(label_mask.sum())[:20]]
                ] = True
            print(node_labels[train_mask].float().histc(n_labels))
            valid_mask = ~train_mask
            valid_mask[
                nrange[valid_mask][torch.randperm(valid_mask.sum())[500:]]
            ] = False
            test_mask = ~(train_mask | valid_mask)
            test_mask[
                nrange[test_mask][torch.randperm(test_mask.sum())[1000:]]
            ] = False
            print(train_mask.sum(), valid_mask.sum(), test_mask.sum())
        train_idx = nrange[train_mask]
        known_idx = nrange[~(valid_mask | test_mask)]
        E = speye(n_nodes)
        if '-trans' in g_method:
            A = [spnorm(graph.adj() + E, eps=0)] * 2
        else:
            # Inductive Settings
            src, dst = graph.edges()
            flt = ~(
                valid_mask[src] | test_mask[src]
                | valid_mask[dst] | test_mask[dst])
            src = src[flt]
            dst = dst[flt]
            n_edges = src.shape[0]
            A = torch.sparse_coo_tensor(
                torch.cat((
                    torch.cat((src, dst), dim=0).unsqueeze(0),
                    torch.cat((dst, src), dim=0).unsqueeze(0)), dim=0),
                values=torch.ones(2 * n_edges),
                size=(n_nodes, n_nodes))
            A = (spnorm(A + E), spnorm(graph.adj() + E, eps=0))
        if 'linkdist' in g_method:
            A = spnorm(graph.adj())
    evaluate.start_run()
    if g_method in ('mlp', 'mlp-trans'):
        mlp = MLP(n_features, hid, n_labels)
        opt = optimize([*mlp.parameters()])
        for epoch in range(1, 1 + epochs):
            mlp.train()
            if g_data == 'ppi':
                for X, Y, src, dst, n, e, d, c, A in graphs['train']:
                    for perm in DataLoader(
                            range(n), batch_size=batch_size, shuffle=True):
                        opt.zero_grad()
                        F.binary_cross_entropy_with_logits(
                            mlp(X[perm]), Y[perm]).backward()
                        opt.step()
                with torch.no_grad():
                    mlp.eval()
                    rets = []
                    for mode in modes:
                        ret = []
                        for X, Y, src, dst, n, e, d, c, A in graphs[mode]:
                            ret.append((mlp(X), Y))
                        rets.append(ret)
                    evaluate.evppi(rets)
            else:
                for perm in DataLoader(
                        train_idx, batch_size=batch_size, shuffle=True):
                    opt.zero_grad()
                    F.cross_entropy(mlp(X[perm]), Y[perm]).backward()
                    opt.step()
                with torch.no_grad():
                    mlp.eval()
                    evaluate(mlp(X))
    elif g_method in ('gcn', 'gcn-trans'):
        gcn = GCN(n_features, hid, n_labels)
        opt = optimize([*gcn.parameters()])
        for epoch in range(1, 1 + epochs):
            gcn.train()
            if g_data == 'ppi':
                for X, Y, src, dst, n, e, d, c, A in graphs['train']:
                    gcn.A = A
                    opt.zero_grad()
                    F.binary_cross_entropy_with_logits(gcn(X), Y).backward()
                    opt.step()
                with torch.no_grad():
                    gcn.eval()
                    rets = []
                    for mode in modes:
                        ret = []
                        for X, Y, src, dst, n, e, d, c, A in graphs[mode]:
                            gcn.A = A
                            ret.append((gcn(X), Y))
                        rets.append(ret)
                    evaluate.evppi(rets)
            else:
                gcn.A = A[0]
                opt.zero_grad()
                F.cross_entropy(gcn(X)[train_mask], Y[train_mask]).backward()
                opt.step()
                with torch.no_grad():
                    gcn.eval()
                    gcn.A = A[1]
                    evaluate(gcn(X))
    elif g_method in ('gcn2mlp', 'gcn-trans2mlp'):
        gcn = GCN(n_features, hid, n_labels)
        opt = optimize([*gcn.parameters()])
        best_acc = 0
        ev2 = Stat()
        ev2.start_run()
        for epoch in range(1, 1 + epochs):
            gcn.train()
            if g_data == 'ppi':
                for X, Y, src, dst, n, e, d, c, A in graphs['train']:
                    gcn.A = A
                    opt.zero_grad()
                    F.binary_cross_entropy_with_logits(gcn(X), Y).backward()
                    opt.step()
                with torch.no_grad():
                    gcn.eval()
                    rets = []
                    for mode in modes:
                        ret = []
                        for X, Y, src, dst, n, e, d, c, A in graphs[mode]:
                            gcn.A = A
                            ret.append((gcn(X), Y))
                        rets.append(ret)
                    ev2.evppi(rets)
                acc = ev2.accs[-1][1]
                if acc > best_acc:
                    best_acc = acc
                    probs = [
                        [torch.sigmoid(logits) for logits, Y in ret]
                        for ret in rets]
            else:
                opt.zero_grad()
                gcn.A = A[0]
                F.cross_entropy(gcn(X)[train_mask], Y[train_mask]).backward()
                opt.step()
                with torch.no_grad():
                    gcn.eval()
                    gcn.A = A[1]
                    logits = gcn(X)
                    ev2(logits)
                acc = ev2.accs[-1][1]
                if acc > best_acc:
                    best_acc = acc
                    probs = torch.softmax(logits.detach(), dim=-1)
        mlp = MLP(n_features, hid, n_labels)
        opt = optimize([*mlp.parameters()])
        for epoch in range(1, 1 + epochs):
            mlp.train()
            if g_data == 'ppi':
                for (X, Y, src, dst, n, e, d, c, A), prob in zip(
                        graphs['train'], probs[0]):
                    for perm in DataLoader(
                            range(n), batch_size=batch_size, shuffle=True):
                        opt.zero_grad()
                        F.binary_cross_entropy_with_logits(
                            mlp(X[perm]), prob[perm]).backward()
                        opt.step()
                with torch.no_grad():
                    mlp.eval()
                    rets = []
                    for mode in modes:
                        ret = []
                        for X, Y, src, dst, n, e, d, c, A in graphs[mode]:
                            ret.append((mlp(X), Y))
                        rets.append(ret)
                    evaluate.evppi(rets)
            else:
                for perm in DataLoader(
                        known_idx, batch_size=batch_size, shuffle=True):
                    opt.zero_grad()
                    F.kl_div(
                        F.log_softmax(mlp(X[perm]), dim=-1), probs[perm]
                    ).backward()
                    opt.step()
                with torch.no_grad():
                    mlp.eval()
                    evaluate(mlp(X))
    elif g_method.startswith('linkdist'):
        linkdist = LinkDist(n_features, hid, n_labels, n_layers=3)
        opt = optimize([*linkdist.parameters()])
        if '-trans' in g_method:
            src, dst = graph.edges()
            n_edges = src.shape[0]
        if g_data == 'ppi':
            params = []
            for X, Y, src, dst, n, e, d, c, A in graphs['train']:
                alpha = 0
                label_ndist = Y.sum(dim=0)
                label_edist = (Y[src] + Y[dst]).sum(dim=0)
                weight = c * F.normalize(
                    label_ndist / label_edist, p=1, dim=0)
                params.append((alpha, weight))
        else:
            alpha = 1 - ((
                train_mask[src].sum() + train_mask[dst].sum()
            ) / (2 * n_edges)).item()
            label_ndist = Y[
                torch.arange(n_nodes)[train_mask]].float().histc(n_labels)
            label_edist = (
                Y[src[train_mask[src]]].float().histc(n_labels)
                + Y[dst[train_mask[dst]]].float().histc(n_labels))
            weight = n_labels * F.normalize(
                label_ndist / label_edist, p=1, dim=0)
        for epoch in range(1, 2 + int(epochs // degree)):
            linkdist.train()
            if g_data == 'ppi':
                for (X, Y, src, dst, n, e, d, c, A), (alpha, weight) in zip(
                        graphs['train'], params):
                    for perm in DataLoader(
                            range(e), batch_size=batch_size, shuffle=True):
                        opt.zero_grad()
                        psrc = src[perm]
                        pdst = dst[perm]
                        y1, z1 = linkdist(X[psrc])
                        y2, z2 = linkdist(X[pdst])
                        loss = alpha * (
                            F.mse_loss(y1, z2) + F.mse_loss(y2, z1)
                        ) + F.binary_cross_entropy_with_logits(
                            y1, Y[psrc], weight=weight
                        ) + F.binary_cross_entropy_with_logits(
                            z2, Y[psrc], weight=weight
                        ) + F.binary_cross_entropy_with_logits(
                            y2, Y[pdst], weight=weight
                        ) + F.binary_cross_entropy_with_logits(
                            z1, Y[pdst], weight=weight)
                        loss.backward()
                        opt.step()
                with torch.no_grad():
                    linkdist.eval()
                    rets = []
                    for mode in modes:
                        ret = []
                        for X, Y, src, dst, n, e, d, c, A in graphs[mode]:
                            Z, S = linkdist(X)
                            if '-nomp' in g_method:
                                ret.append((Z, Y))
                            else:
                                ret.append((Z + alpha * (A @ S), Y))
                        rets.append(ret)
                    evaluate.evppi(rets)
            else:
                for perm in DataLoader(
                        range(n_edges), batch_size=batch_size, shuffle=True):
                    opt.zero_grad()
                    psrc = src[perm]
                    pdst = dst[perm]
                    y1, z1 = linkdist(X[psrc])
                    y2, z2 = linkdist(X[pdst])
                    loss = alpha * (F.mse_loss(y1, z2) + F.mse_loss(y2, z1))
                    m = train_mask[psrc]
                    if m.any().item():
                        target = Y[psrc][m]
                        loss = loss + (
                            F.cross_entropy(y1[m], target, weight=weight)
                            + F.cross_entropy(z2[m], target, weight=weight))
                    m = train_mask[pdst]
                    if m.any().item():
                        target = Y[pdst][m]
                        loss = loss + (
                            F.cross_entropy(y2[m], target, weight=weight)
                            + F.cross_entropy(z1[m], target, weight=weight))
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    linkdist.eval()
                    Z, S = linkdist(X)
                    if '-nomp' in g_method:
                        evaluate(Z)
                    else:
                        evaluate(
                            F.log_softmax(Z, dim=-1)
                            + alpha * (A @ F.log_softmax(S, dim=-1)))
    else:
        # Label Propagation
        alpha = 0.4
        Z = gpu(torch.zeros(n_nodes, n_labels))
        train_probs = gpu(F.one_hot(Y, n_labels)[train_mask].float())
        for _ in range(50):
            Z[train_mask] = train_probs
            Z = (1 - alpha) * Z + alpha * (A[1] @ Z)
            evaluate(Z)
    evaluate.end_run()
evaluate.end_all()
