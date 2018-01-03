
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module, Linear
from torch.nn.parameter import Parameter
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm
import numpy as np
from tqdm import tqdm
from sklearn import metrics


def log_gaussian_posterior(x, mu, sigma):

    expr = -0.5 * np.log(2 * np.pi) - torch.log(sigma) - 0.5 * torch.pow(x - mu, 2) / torch.pow(sigma, 2)

    return expr


def log_gaussian_prior(x, sigma=1):

    sigma = Variable(sigma * torch.ones(x.size()).cuda(), requires_grad=False)
    expr = -0.5 * torch.pow(x, 2) / torch.pow(sigma, 2) - torch.log(sigma)

    return expr


# def log_gaussian(x):
#     # From theano script

#     mu = torch.ones(x.size())
#     sigma = np.exp(-3) * torch.ones(x.size())

#     mu = Variable(mu.cuda(), requires_grad=False)
#     sigma = Variable(sigma.cuda(), requires_grad=False)

#     return - torch.log(sigma) - 0.5 * torch.pow((x - mu) / sigma, 2)


# def log_gaussian_logsigma(x, mu, logsigma):

#     # From theano script
#     return - logsigma / 2. - (x - mu) ** 2 / (2. * torch.exp(logsigma))


def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2)


def log_gaussian_logsigma(x, mu, logsigma):
    return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)


def log_gaussian_mixedprior(x, sigma_wide=1, sigma_narrow=2E-3, frac=0.5):

    sigma_wide = Variable(sigma_wide * torch.ones(x.size()).cuda(), requires_grad=False)
    sigma_narrow = Variable(sigma_narrow * torch.ones(x.size()).cuda(), requires_grad=False)

    expr_wide = frac * torch.exp(-0.5 * torch.pow(x, 2) / torch.pow(sigma_wide, 2)) / sigma_wide
    expr_narrow = (1 - frac) * torch.exp(-0.5 * torch.pow(x, 2) / torch.pow(sigma_narrow, 2)) / sigma_wide

    expr = torch.log(expr_wide + expr_narrow)

    return expr


class BayesLinear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # self.w_mu = Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.01))
        # self.w_rho = Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.01))
        # self.b_mu = Parameter(torch.Tensor(out_features).uniform_(-0.01, 0.01))
        # self.b_rho = Parameter(torch.Tensor(out_features).uniform_(-0.01, 0.01))

        self.w_mu = Parameter(torch.Tensor(out_features, in_features))
        self.w_rho = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.b_mu = Parameter(torch.Tensor(out_features))
            self.b_rho = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('b_mu', None)
            self.register_parameter('b_rho', None)

        self.log_q = None
        self.log_p = None
        self.kl = None

        self.reset_parameters()

    def reset_parameters(self):

        # Initialize weights
        # stdv = 1. / np.sqrt(self.w_mu.size(1))
        self.w_mu.data.normal_(0., 0.05)
        self.w_rho.data.normal_(0., 0.05)
        # self.w_mu.data.uniform_(-stdv, stdv)
        # self.w_rho.data.fill_(-3)

        # Initialize biases
        if self.bias is not None:
            self.b_mu.data.normal_(0., 0.05)
            self.b_rho.data.normal_(0., 0.05)
            # self.b_mu.data.uniform_(-stdv, stdv)
            # self.b_rho.data.fill_(-3)

        # self.w_mu.data.normal_(0, 0.01)
        # self.w_rho.data.normal_(0, 0.01)

        # if self.bias is not None:
            # self.b_mu.data.uniform_(-0.01, 0.01)
            # self.b_rho.data.normal_(0, 0.01)

    def get_weights(self):

        return (self.w_mu, self.w_rho), (self.b_mu, self.b_rho)

    def forward(self, input):

        # Sample weights and biases from normal distribution
        w_epsilon = torch.normal(mean=0, std=np.exp(-3) * torch.ones(self.w_mu.size())).cuda()
        b_epsilon = torch.normal(mean=0, std=np.exp(-3) * torch.ones(self.b_mu.size())).cuda()

        w_epsilon = Variable(w_epsilon, requires_grad=False)
        b_epsilon = Variable(b_epsilon, requires_grad=False)

        w_sigma = torch.log(1.0 + torch.exp(self.w_rho)).cuda()
        b_sigma = torch.log(1.0 + torch.exp(self.b_rho)).cuda()

        weight = self.w_mu + w_sigma * w_epsilon
        bias = self.b_mu + b_sigma * b_epsilon

        self.log_p = log_gaussian(weight, 0, float(np.exp(-3))).sum()
        self.log_p += log_gaussian(bias, 0, float(np.exp(-3))).sum()

        # Approximation ? TODO CHECK
        self.log_q = log_gaussian_logsigma(weight, self.w_mu, self.w_rho).sum()
        self.log_q += log_gaussian_logsigma(bias, self.b_mu, self.b_rho).sum()

        # self.log_p = log_gaussian_prior(weight).sum()
        # self.log_p += log_gaussian_prior(bias).sum()

        # self.log_p = log_gaussian_mixedprior(weight).sum()
        # self.log_p += log_gaussian_mixedprior(bias).sum()

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
        self.dense3 = Linear(hidden_dim, output_size)

    def forward(self, x):

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)

        return x


class BayesMLP(Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(BayesMLP, self).__init__()

        # Params
        self.hidden_dim = hidden_dim

        # Layers / nn objects
        self.dense1 = BayesLinear(input_size, hidden_dim)
        self.dense2 = BayesLinear(hidden_dim, hidden_dim)
        self.dense3 = BayesLinear(hidden_dim, output_size)

    def get_weights(self):

        d_weights = {"w": [], "b": []}
        for layer in [self.dense1, self.dense2, self.dense3]:
            weights = layer.get_weights()
            d_weights["w"].append(weights[0])
            d_weights["b"].append(weights[1])

        return d_weights

    def forward(self, x):

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)

        self.kl = self.dense1.kl.sum() + self.dense2.kl.sum() + self.dense3.kl.sum()

        return x


def get_metrics(model, X, Y):

    # Create a list of batches
    num_elem = X.shape[0]
    batch_size = 128
    num_batches = num_elem / batch_size
    list_batches = np.array_split(np.arange(num_elem), num_batches)

    list_preds = []

    for batch_idxs in tqdm(list_batches):

        # Load batch
        start, end = batch_idxs[0], batch_idxs[-1] + 1
        X_batch = X[start:end]
        X_batch = torch.FloatTensor(X_batch).cuda()
        X_var = Variable(X_batch)

        Y_out = F.softmax(model(X_var)).data.cpu().numpy()

        list_preds.append(Y_out)

    Y_preds = np.concatenate(list_preds)
    Y_pred_class = np.argmax(Y_preds, 1)
    acc = metrics.accuracy_score(Y, Y_pred_class)

    print("Validation accuracy", acc)


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


def train_mnist():

    mnist = np.load("mnist.npz")
    X_train, Y_train = mnist['X_train'][:], mnist["Y_train"][:]
    print(X_train.shape, Y_train.shape)

    # Reshape and normalize
    X_train = X_train.reshape(-1, 784) / 255.
    Y_train = Y_train.astype(np.int64)

    X_test, Y_test = mnist['X_test'], mnist["Y_test"]
    print(X_test.shape, Y_test.shape)

    # Reshape and normalize
    X_test = X_test.reshape(-1, 784) / 255.
    Y_test = Y_test.astype(np.int64)

    # Create a list of batches
    num_elem = X_train.shape[0]
    batch_size = 100
    num_batches = num_elem / batch_size
    list_train_batches = np.array_split(np.arange(num_elem), num_batches)

    num_elem = X_test.shape[0]
    batch_size = 100
    num_batches = num_elem / batch_size
    list_test_batches = np.array_split(np.arange(num_elem), num_batches)

    # Load model
    model = BayesMLP(784, 200, 10).cuda()
    # model = MLP(784, 200, 10).cuda()
    print(model)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1E-3)

    desc_str = ""
    for epoch in tqdm(range(100)):

        list_likelihood_loss = []
        list_acc_loss = []
        list_kl_loss = []

        for batch_idxs in tqdm(list_train_batches[:], desc=desc_str):

            # Reset gradients
            optim.zero_grad()

            # Load batch
            start, end = batch_idxs[0], batch_idxs[-1] + 1
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]
            X_batch, Y_batch = torch.FloatTensor(X_batch).cuda(), torch.LongTensor(Y_batch).cuda()
            X_var, Y_var = Variable(X_batch), Variable(Y_batch)

            # Forward pass
            out = model(X_var)
            predictions = out.max(1)[1].type_as(Y_var)
            correct = predictions.eq(Y_var)
            acc_loss = 100. * correct.data.cpu().sum() / len(batch_idxs)

            # Backward pass
            likelihood_loss = criterion(out, Y_var)

            if isinstance(model, BayesMLP):
                kl_loss = model.kl / (X_train.shape[0])
                loss = likelihood_loss + kl_loss
            else:
                loss = likelihood_loss
            loss.backward()
            optim.step()

            list_likelihood_loss.append(likelihood_loss.data.cpu().numpy()[0])
            list_acc_loss.append(acc_loss)
            if isinstance(model, BayesMLP):
                list_kl_loss.append(kl_loss.data.cpu().numpy()[0])
            else:
                list_kl_loss.append(0)

        # Get test acc
        list_out = []
        for batch_idxs in list_test_batches:

            # Load batch
            start, end = batch_idxs[0], batch_idxs[-1] + 1
            X_batch = X_test[start:end]
            X_batch = torch.FloatTensor(X_batch).cuda()
            X_var = Variable(X_batch, volatile=True)

            # Forward pass
            out = model(X_var).data.cpu().numpy()
            out = np.argmax(out, 1)
            list_out.append(out)

        Y_test_pred = np.concatenate(list_out)
        correct = Y_test_pred == Y_test
        acc_loss = 100. * np.sum(correct) / len(correct)

        desc_str = "Likelihood %.3g -- Acc: %.3g --  KL %.3g" % (np.mean(list_likelihood_loss),
                                                                 acc_loss,
                                                                 np.mean(list_kl_loss))

        plot_weights(model)

    print("\n")
    print("Likelihood", np.mean(list_likelihood_loss), "KL", np.mean(list_kl_loss))
    get_metrics(model, X_train, Y_train)
    print("\n")


if __name__ == '__main__':

    train_mnist()
