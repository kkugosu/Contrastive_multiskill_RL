from utils import converter
import torch
from NeuralNetwork import basic_nn
import numpy as np
import math
from control import BASE
from torch import nn


class EDL(BASE.BaseControl):
    """
    l_r : learning rate
    s_l : state length
    policy : policy
    skill_num : skill num
    device : device
    """
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.cont_name = "edl"
        self.encoder = basic_nn.ProbNN(self.s_l, self.s_l + self.skills, self.skills)
        self.decoder = basic_nn.ProbNN(self.skills, self.s_l + self.skills, self.s_l)
        self.optimizer_e = torch.optim.SGD(self.encoder.parameters(), lr=self.l_r)
        self.optimizer_d = torch.optim.SGD(self.decoder.parameters(), lr=self.l_r)
        self.initial_state = None
        self.criterion = nn.MSELoss(reduction='mean')

    def reward(self, state_1, state_2, skill, done):
        return self.decoder(state_1, skill)

    @staticmethod
    def _rep(mu):
        return mu + torch.randn_like(mu) * 0.1

    def train_encoding(self, *trajectory):
        n_p_s, n_a, n_s, n_r, n_d, skill_idx = np.squeeze(trajectory)
        skill = self.encoder(n_p_s)
        skill = self._rep(skill)
        output = self.decoder(skill, n_p_s)
        self.optimizer_e.zero_grad()
        self.optimizer_d.zero_grad()
        output.sum().backward()
        for param in self.encoder.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.decoder.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_e.step()
        self.optimizer_d.step()
        return output.sum()

    def update(self, memory_iter, *trajectory):
        i = 0
        loss1 = None
        loss2_ary = None
        while i < memory_iter:
            i = i + 1
            loss1 = self.train_encoding(trajectory)
            loss2_ary = self.policy.update(1, trajectory)
        loss_ary = torch.cat((loss2_ary, loss1.unsqueeze(0)), -1)
        return loss_ary

    def load_model(self, path):
        self.encoder.load_state_dict(torch.load(path + self.cont_name))
        self.decoder.load_state_dict(torch.load(path + self.cont_name))
        self.policy.load_model(path)

    def save_model(self, path):
        torch.save(self.encoder.state_dict(), path + self.cont_name)
        torch.save(self.decoder.state_dict(), path + self.cont_name)
        models = self.policy.save_model(path)
        return (self.encoder, self.decoder) + models

