# Drawn from https://gist.github.com/rocknrollnerd/c5af642cf217971d93f499e8f70fcb72 (in Theano)
# This is implemented in PyTorch
# Author : Anirudh Vemula

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm


def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2)


def log_gaussian_logsigma(x, mu, logsigma):
    return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)


def plot_weights(model):

    d_weights = model.get_weights()

    list_db = []

    for w_mu, w_rho in d_weights["w"]:
        w_mu = w_mu.data.cpu().numpy()
        w_rho = w_rho.data.cpu().numpy()
        db = np.log(np.abs(w_mu) / np.log(1.0 + np.exp(w_rho)))
        list_db += np.ravel(db).tolist()

    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2,1)
    ax = plt.subplot(gs[0])
    ax.hist(list_db, bins=100)

    hist, bin_edges = np.histogram(list_db, bins=100, normed=True)
    dx = bin_edges[1] - bin_edges[0]
    F1 = np.cumsum(hist) * dx
    ax = plt.subplot(gs[1])
    ax.plot(bin_edges[1:], F1)

    plt.savefig("weight_dist.png")
    plt.clf()
    plt.close()


class MLPLayer(nn.Module):
    def __init__(self, n_input, n_output, sigma_prior):
        super(MLPLayer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.sigma_prior = sigma_prior
        self.W_mu = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 0.01))
        self.W_logsigma = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 0.01))
        self.b_mu = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
        self.b_logsigma = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
        self.lpw = 0
        self.lqw = 0

    def forward(self, X, infer=False):
        if infer:
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_output)
            return output

        epsilon_W, epsilon_b = self.get_random()
        W = self.W_mu + torch.log(1 + torch.exp(self.W_logsigma)) * epsilon_W
        b = self.b_mu + torch.log(1 + torch.exp(self.b_logsigma)) * epsilon_b
        output = torch.mm(X, W) + b.expand(X.size()[0], self.n_output)
        self.lpw = log_gaussian(W, 0, self.sigma_prior).sum() + log_gaussian(b, 0, self.sigma_prior).sum()
        self.lqw = log_gaussian_logsigma(W, self.W_mu, self.W_logsigma).sum() + \
            log_gaussian_logsigma(b, self.b_mu, self.b_logsigma).sum()
        return output

    def get_weights(self):

        return (self.W_mu, torch.log(1 + torch.exp(self.W_logsigma))), (self.b_mu, torch.log(1 + torch.exp(self.b_logsigma)))

    def get_random(self):
        return Variable(torch.Tensor(self.n_input, self.n_output).normal_(0, self.sigma_prior).cuda()), Variable(torch.Tensor(self.n_output).normal_(0, self.sigma_prior).cuda())


class MLP(nn.Module):
    def __init__(self, n_input, sigma_prior):
        super(MLP, self).__init__()
        self.l1 = MLPLayer(n_input, 200, sigma_prior)
        self.l1_relu = nn.ReLU()
        self.l2 = MLPLayer(200, 200, sigma_prior)
        self.l2_relu = nn.ReLU()
        self.l3 = MLPLayer(200, 10, sigma_prior)
        self.l3_softmax = nn.Softmax()

    def forward(self, X, infer=False):
        output = self.l1_relu(self.l1(X, infer))
        output = self.l2_relu(self.l2(output, infer))
        output = self.l3_softmax(self.l3(output, infer))
        return output

    def get_lpw_lqw(self):
        lpw = self.l1.lpw + self.l2.lpw + self.l3.lpw
        lqw = self.l1.lqw + self.l2.lqw + self.l3.lqw
        return lpw, lqw

    def get_weights(self):

        d_weights = {"w": [], "b": []}
        for layer in [self.l1, self.l2, self.l3]:
            weights = layer.get_weights()
            d_weights["w"].append(weights[0])
            d_weights["b"].append(weights[1])

        return d_weights


def forward_pass_samples(X, y):
    s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.
    for _ in range(n_samples):
        output = net(X)
        sample_log_pw, sample_log_qw = net.get_lpw_lqw()
        sample_log_likelihood = log_gaussian(y, output, sigma_prior).sum()
        # sample_log_likelihood = (torch.log(output) * y).sum()
        s_log_pw += sample_log_pw
        s_log_qw += sample_log_qw
        s_log_likelihood += sample_log_likelihood

    return s_log_pw / n_samples, s_log_qw / n_samples, s_log_likelihood / n_samples


def criterion(l_pw, l_qw, l_likelihood):
    return ((1. / n_batches) * (l_qw - l_pw) - l_likelihood).sum() / float(batch_size)


N = 5000

mnist = np.load("mnist.npz")
train_data, train_target = mnist['X_train'][:].astype(np.float32), mnist["Y_train"][:]

# Reshape and normalize
train_data = train_data.reshape(-1, 784) / 255.
train_target = train_target.astype(np.int64)

test_data, test_target = mnist['X_test'][:].astype(np.float32), mnist["Y_test"][:]

# Reshape and normalize
test_data = test_data.reshape(-1, 784) / 255.
test_target = test_target.astype(np.int64)

train_target = np.float32(preprocessing.OneHotEncoder(sparse=False).fit_transform(train_target.reshape(-1, 1)))

n_input = train_data.shape[1]
M = train_data.shape[0]
sigma_prior = float(np.exp(-3))
n_samples = 1
learning_rate = 0.001
n_epochs = 100

# Initialize network
net = MLP(n_input, sigma_prior)
net = net.cuda()

# building the objective
# remember, we're evaluating by samples
log_pw, log_qw, log_likelihood = 0., 0., 0.
batch_size = 100
n_batches = M / float(batch_size)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

n_train_batches = int(train_data.shape[0] / float(batch_size))

for e in range(n_epochs):
    errs = []
    for b in tqdm(range(n_train_batches)):
        net.zero_grad()
        X = Variable(torch.Tensor(train_data[b * batch_size: (b + 1) * batch_size]).cuda())
        y = Variable(torch.Tensor(train_target[b * batch_size: (b + 1) * batch_size]).cuda())

        log_pw, log_qw, log_likelihood = forward_pass_samples(X, y)
        loss = criterion(log_pw, log_qw, log_likelihood)
        errs.append(loss.data.cpu().numpy())
        loss.backward()
        optimizer.step()

    X = Variable(torch.Tensor(test_data).cuda(), volatile=True)
    pred = net(X, infer=True)
    _, out = torch.max(pred, 1)
    acc = np.count_nonzero(np.squeeze(out.data.cpu().numpy()) == np.int32(
        test_target.ravel())) / float(test_data.shape[0])

    print('epoch', e, 'loss', np.mean(errs), 'acc', acc)
    plot_weights(net)