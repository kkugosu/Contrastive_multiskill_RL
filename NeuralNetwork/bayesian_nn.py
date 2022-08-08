import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("device", device)


class CustomActivationF:

    def __init__(self, rate=1):
        self.rate = rate

    def log_act(self, a):
        """
        logistic activation function
        """
        i = 0
        while i < len(a):
            j = 0
            while j < len(a[i]):
                if a[i][j] > 0:
                    a[i][j] = torch.log(a[i][j] + self.rate)
                else:
                    a[i][j] = - torch.log(self.rate - a[i][j])
                j = j + 1
            i = i + 1
        return a


class BayesianLinear(nn.Module):

    def __init__(self, i_s, o_s):
        """
        param i_s : input_size
        param o_s : output_size
        """
        super().__init__()
        self.freeze = 0
        self.i_s = i_s
        self.o_s = o_s
        self.w = nn.Parameter(
            torch.zeros(self.i_s, self.o_s, dtype=torch.float32, requires_grad=True)
        )
        self.b = nn.Parameter(
            torch.zeros(1, self.o_s, dtype=torch.float32, requires_grad=True)
        )
        self.w_prior = nn.Parameter(
            torch.zeros(self.i_s, self.o_s, dtype=torch.float32, requires_grad=False)
        )
        self.b_prior = nn.Parameter(
            torch.zeros(1, self.o_s, dtype=torch.float32, requires_grad=False)
        )

    @staticmethod
    def _rep(mu):
        return mu + torch.randn_like(mu) * 0.1

    def make_freeze(self, freeze):
        self.freeze = freeze
        return None

    def kld_loss(self):
        sum1 = torch.sum(torch.square(self.w - self.w_prior))
        sum2 = torch.sum(torch.square(self.b - self.b_prior))
        return sum1 + sum2

    def forward(self, x):
        if self.freeze == 0:
            b = self._rep(self.b)
            w = self._rep(self.w)
        else:
            b = self.b
            w = self.w
        x = torch.matmul(x, w) + b
        # self._update_prior(self.w1_prior, self.w2_prior, self.b_prior, rate)
        # if we want to move prior, we can just subtract _prior term at the upper line
        return x


class BayesianModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.b_linear1 = BayesianLinear(input_size, hidden_size)
        self.b_linear2 = BayesianLinear(hidden_size, hidden_size)
        self.b_linear3 = BayesianLinear(hidden_size, output_size)
        self.layer = nn.Sequential(
            self.b_linear1,
            nn.Tanh(),
            self.b_linear2,
            nn.LeakyReLU(0.1),
            self.b_linear3
        )

    def set_freeze(self, freeze):
        self.b_linear1.make_freeze(freeze)
        self.b_linear2.make_freeze(freeze)
        self.b_linear3.make_freeze(freeze)
        return None

    def forward(self, x):
        result = self.layer(x)
        # self._update_prior(self.w1_prior, self.w2_prior, self.b_prior, rate)
        # if we want to move prior, we can just subtract _prior term at the upper line
        return result

    def kld_loss(self):
        l1 = self.b_linear1.kld_loss()
        l2 = self.b_linear2.kld_loss()
        l3 = self.b_linear3.kld_loss()
        return l1 + l2 + l3
