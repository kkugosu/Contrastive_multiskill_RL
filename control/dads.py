import torch
from NeuralNetwork import basic_nn
import numpy as np
from control import BASE
from torch import nn


class DADS(BASE.BaseControl):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.cont_name = "dads"
        self.discriminator = basic_nn.ProbNN(self.s_l * self.sk_n, self.s_l * self.sk_n, self.s_l).to(self.device)
        # state + skill -> state
        self.optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=self.l_r)
        self.criterion = nn.MSELoss(reduction='mean')

    def reward(self, *trajectory):
        n_p_s, n_a, n_s, n_r, n_d, sk_idx = np.squeeze(trajectory)
        t_p_s = torch.from_numpy(n_p_s).to(self.device).type(torch.float32)
        t_s = torch.from_numpy(n_s).to(self.device).type(torch.float32)
        sk_idx = torch.from_numpy(sk_idx).to(self.device).type(torch.int64)

        tmp = torch.zeros((len(t_p_s), self.s_l * self.sk_n))
        i = 0
        while i < len(t_p_s):
            tmp[i][sk_idx[i] * self.s_l: (sk_idx[i] + 1) * self.s_l] = t_p_s[i]
            i = i + 1
        # how can i use scatter in here?
        tmp = tmp.type(torch.float32).to(self.device)
        main_prob = torch.exp(-torch.sum(torch.square(t_s - self.discriminator(tmp)), dim=-1))

        tmp = torch.zeros((len(t_p_s), self.sk_n, self.s_l * self.sk_n))
        i = 0
        while i < len(n_p_s):
            j = 0
            while j < self.sk_n:
                tmp[i][j * self.s_l:(j + 1) * self.s_l] = n_p_s[i]
                j = j + 1
            i = i + 1
        tmp = tmp.type(torch.float32).to(self.device)
        sub_prob = torch.exp(-torch.sum(torch.square(t_s - self.discriminator(tmp)), dim=-1))
        contrast_prob = torch.sum(sub_prob, dim=-1)

        re = main_prob/contrast_prob
        re = re.squeeze()
        return re

    def update(self, memory_iter, *trajectory):
        i = 0
        loss1 = None
        loss2_ary = None
        while i < memory_iter:
            i = i + 1
            loss2_ary = self.policy.update(1, trajectory)
            out = self.reward(trajectory)
            loss1 = - torch.sum(torch.log(out))
            self.optimizer.zero_grad()
            loss1.backward()
            for param in self.discriminator.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        loss_ary = torch.cat((loss2_ary, loss1.unsqueeze(0)), -1)
        return loss_ary

    def load_model(self, path):
        self.discriminator.load_state_dict(torch.load(path + self.cont_name))
        self.policy.load_model(path)

    def save_model(self, path):
        torch.save(self.discriminator.state_dict(), path + self.cont_name)
        models = self.policy.save_model(path)
        return (self.discriminator,) + models
