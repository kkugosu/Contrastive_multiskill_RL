
from utils import converter
import torch
from NeuralNetwork import basic_nn
import numpy as np
import math
from control import BASE
from torch import nn


class DADS(BASE.BaseControl):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.cont_name = "dads"
        self.discriminator = basic_nn.ProbNN(self.s_l*self.skills, self.s_l * self.skills, self.s_l).to(self.device)
        self.optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=self.l_r)
        self.criterion = nn.MSELoss(reduction='mean')

    def reward(self, *trajectory):
        n_p_s, n_a, n_s, n_r, n_d, skill_idx = np.squeeze(trajectory)
        skill_idx = torch.from_numpy(skill_idx).to(self.device).type(torch.int64)
        skill_idx = skill_idx.unsqueeze(-1)
        tmp_n_p_o = np.zeros((len(n_p_s), self.s_l * self.skills))
        # batch, statelen, skilllen
        i = 0
        while i < len(n_p_s):
            tmp_n_p_o[i][skill_idx[i] * self.s_l:(skill_idx[i] + 1) * self.s_l] = n_p_s[i]
            i = i + 1
        # how can i use scatter in here?
        n_p_o = tmp_n_p_o
        t_p_o = torch.from_numpy(n_p_o).type(torch.float32).to(self.device)
        main_prob = torch.exp(-self.criterion(n_s, self.discriminator(t_p_o)))
        tmp_n_p_o = np.zeros((len(n_p_s), self.skills, self.s_l * self.skills))
        i = 0
        while i < len(n_p_s):
            j = 0
            while j < self.skills:
                tmp_n_p_o[i][j * self.s_l:(j + 1) * self.s_l] = n_p_s[i]
                j = j + 1
            i = i + 1
        n_p_o = tmp_n_p_o
        t_p_o = torch.from_numpy(n_p_o).type(torch.float32).to(self.device)
        sub_prob = torch.exp(-self.criterion(n_s, self.discriminator(t_p_o)))
        contrast_prob = torch.sum(sub_prob)
        re = main_prob/contrast_prob
        return re
        # we need to revise later

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
