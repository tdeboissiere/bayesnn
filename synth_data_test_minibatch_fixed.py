
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module, Linear
from torch.nn.parameter import Parameter

import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm


# def log_gaussian_posterior(x, mu, sigma):

#     expr = -0.5 * torch.pow(x - mu, 2) / torch.pow(sigma, 2) - torch.log(sigma)

#     return expr


def log_gaussian_logsigma(x, mu, logsigma):
    return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)


def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2)


def log_gaussian_mixedprior(x, sigma_wide=1., sigma_narrow=2E-3, frac=0.5):

    expr_wide = float(frac) * torch.exp(-0.5 * torch.pow(x / float(sigma_wide), 2)) / float(sigma_wide)
    expr_narrow = float((1 - frac)) * torch.exp(-0.5 * torch.pow(x / float(sigma_narrow), 2)) / float(sigma_narrow)

    expr = torch.log(expr_wide + expr_narrow)

    return expr


class BayesLinear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # stdv = 1. / np.sqrt(in_features)
        stdv = 0.0001

        self.w_mu = Parameter(torch.Tensor(out_features, in_features).uniform_(-stdv, stdv))
        self.w_rho = Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -2))
        if bias:
            self.b_mu = Parameter(torch.Tensor(out_features).uniform_(-0.01, 0.01))
            self.b_rho = Parameter(torch.Tensor(out_features).uniform_(-5, -2))
        else:
            self.register_parameter('b_mu', None)
            self.register_parameter('b_rho', None)

        self.log_q = None
        self.log_p = None
        self.kl = None

    def get_weights(self):

        return (self.w_mu, self.w_rho), (self.b_mu, self.b_rho)

    def forward(self, input):

        # Sample weights and biases from normal distribution
        w_epsilon = torch.normal(mean=0, std=torch.ones(self.w_mu.size()))
        b_epsilon = torch.normal(mean=0, std=torch.ones(self.b_mu.size()))

        w_epsilon = Variable(w_epsilon, requires_grad=False)
        b_epsilon = Variable(b_epsilon, requires_grad=False)

        w_sigma = torch.log(1.0 + torch.exp(self.w_rho))
        b_sigma = torch.log(1.0 + torch.exp(self.b_rho))

        weight = self.w_mu + w_sigma * w_epsilon
        bias = self.b_mu + b_sigma * b_epsilon

        # self.log_p = log_gaussian(weight, 0., float(1)).sum()
        # self.log_p += log_gaussian(bias, 0., float(1)).sum()

        self.log_p = log_gaussian_mixedprior(weight).sum()
        self.log_p += log_gaussian_mixedprior(bias).sum()

        # Approximation ? TODO CHECK
        self.log_q = log_gaussian_logsigma(weight, self.w_mu, w_sigma).sum()
        self.log_q += log_gaussian_logsigma(bias, self.b_mu, b_sigma).sum()

        self.kl = self.log_q - self.log_p

        return F.linear(input, weight, bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class MLP(Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(MLP, self).__init__()

        # Params
        self.hidden_dim = hidden_dim

        # Layers / nn objects
        self.dense1 = Linear(input_size, hidden_dim)
        self.dense2 = Linear(hidden_dim, hidden_dim)
        self.dense3 = Linear(hidden_dim, hidden_dim)
        self.dense4 = Linear(hidden_dim, output_size)

    def forward(self, x):

        x = F.leaky_relu(self.dense1(x))
        x = F.leaky_relu(self.dense2(x))
        x = F.leaky_relu(self.dense3(x))
        x = self.dense4(x)

        return x


class BayesMLP(Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(BayesMLP, self).__init__()

        # Params
        self.hidden_dim = hidden_dim

        # Layers / nn objects
        self.dense1 = BayesLinear(input_size, hidden_dim)
        self.dense2 = BayesLinear(hidden_dim, hidden_dim)
        self.dense3 = BayesLinear(hidden_dim, hidden_dim)
        self.dense4 = BayesLinear(hidden_dim, output_size)

    def get_weights(self):

        d_weights = {"w": [], "b": []}
        for layer in [self.dense1, self.dense2, self.dense3]:
            weights = layer.get_weights()
            d_weights["w"].append(weights[0])
            d_weights["b"].append(weights[1])

        return d_weights

    def forward(self, x):

        x = F.leaky_relu(self.dense1(x))
        x = F.leaky_relu(self.dense2(x))
        x = F.leaky_relu(self.dense3(x))
        x = self.dense4(x)

        self.kl = self.dense1.kl.sum() + self.dense2.kl.sum() + self.dense3.kl.sum() + self.dense4.kl.sum()

        return x


def plot_weights(model):

    d_weights = model.get_weights()

    list_db = []
    list_w = []

    for w_mu, w_rho in d_weights["w"]:
        w_mu = w_mu.data.cpu().numpy()
        w_rho = w_rho.data.cpu().numpy()
        db = np.log(np.abs(w_mu) / np.log(1.0 + np.exp(w_rho)))
        list_db += np.ravel(db).tolist()
        list_w += np.ravel(w_mu).tolist()

    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3,1)

    ax = plt.subplot(gs[2])
    ax.hist(list_w, bins=100)

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


def get_metrics(model, X, Y):

    x = np.linspace(-1, 1.2, 1000).reshape(-1, 1)
    x_var = Variable(torch.FloatTensor(x))

    if isinstance(model, BayesMLP):
        # Make several predictions
        list_preds = []
        for i in range(100):
            list_preds.append(model(x_var).data.cpu().numpy())
        preds = np.stack(list_preds)

        plt.scatter(X, Y, s=5, color="C0")
        plt.plot(x, np.median(preds, 0), color="C1")

        upper = np.mean(preds, 0) + np.std(preds, 0)
        lower = np.mean(preds, 0) - np.std(preds, 0)

        plt.fill_between(x[:, 0], lower[:, 0], upper[:, 0], color='C1', alpha=0.4)

        upper = np.mean(preds, 0) + 1.64 * np.std(preds, 0)
        lower = np.mean(preds, 0) - 1.64 * np.std(preds, 0)

        plt.fill_between(x[:, 0], lower[:, 0], upper[:, 0], color='C1', alpha=0.2)
        plt.ylim([-0.4, 1.])
        plt.xlim([-0.2, 1.2])
        plt.show()

        plt.show()

    else:
        preds = model(x_var).data.cpu().numpy()

        plt.scatter(X, Y, s=5, color="C0")
        plt.plot(x, preds, color="C1")
        plt.ylim([-0.4, 1.])
        plt.show()


def train_synth():

    X_train = np.random.uniform(0, 0.6, 500).reshape(-1, 1)
    eps = np.random.normal(0, 0.04, X_train.shape[0]).reshape(-1, 1)
    # Y_train = X_train + 0.3 * np.sin(2 * np.pi * (X_train + eps)) + 0.3 * np.sin(4 * np.pi * (X_train + eps)) + eps
    Y_train = X_train + 0.3 * np.sin(2 * np.pi * (X_train + eps)) + 0.3 * np.sin(4 * np.pi * (X_train)) + eps

    # plt.scatter(X_train, Y_train, s=5, color="C0")
    # plt.show()

    # Create a list of batches
    num_elem = X_train.shape[0]
    batch_size = 32
    num_batches = num_elem / batch_size
    list_batches = np.array_split(np.arange(num_elem), num_batches)

    # Load model
    model = BayesMLP(1, 32, 1)
    print(model)
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1E-3)

    desc_str = ""
    for epoch in tqdm(range(40)):

        list_likelihood_loss = []
        list_kl_loss = []

        for batch_idxs in tqdm(list_batches, desc=desc_str):

            # Reset gradients
            optim.zero_grad()

            # Load batch
            start, end = batch_idxs[0], batch_idxs[-1] + 1
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]
            X_batch, Y_batch = torch.FloatTensor(X_batch), torch.FloatTensor(Y_batch)
            X_var, Y_var = Variable(X_batch), Variable(Y_batch)

            # Forward pass
            out = model(X_var)

            # Backward pass
            likelihood_loss = criterion(out, Y_var)
            kl_loss = model.kl / (X_train.shape[0])
            loss = likelihood_loss  #+ 0.001 * kl_loss

            loss.backward()
            optim.step()

            list_likelihood_loss.append(likelihood_loss.data.cpu().numpy()[0])
            list_kl_loss.append(kl_loss.data.cpu().numpy()[0])
            # list_kl_loss.append(0)

        desc_str = "Likelihood %.3g --  KL %.3g" % (np.mean(list_likelihood_loss), np.mean(list_kl_loss))

        if epoch % 10 == 0:

            plot_weights(model)

    print("\n")
    print("Likelihood", np.mean(list_likelihood_loss), "KL", np.mean(list_kl_loss))
    get_metrics(model, X_train, Y_train)
    print("\n")


if __name__ == '__main__':

    train_synth()
