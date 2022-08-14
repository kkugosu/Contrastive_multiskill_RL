from policy import BASE, act
import gym
import torch
import numpy as np
import sys
from torch import nn
from NeuralNetwork import basic_nn
from utils import buffer
import random
GAMMA = 0.98


class PGPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.upd_policy = basic_nn.ProbNN(self.s_l, self.h_s, self.a_index_l).to(self.device)
        self.optimizer = torch.optim.SGD(self.upd_policy.parameters(), lr=self.l_r)

    def action(self):
        with torch.no_grad():
            probability = self.upd_policy(t_p_o)

        t_a_index = torch.multinomial(probability, 1)
        n_a = self.converter.index2act(t_a_index.squeeze(-1), 1)
        return n_a

    def update(self, trajectary):
        i = 0
        while i < self.m_i:
            # print(i)
            n_p_s, n_a, n_s, n_r, n_d = trajectary # next(iter(self.dataloader))
            t_p_s = torch.tensor(n_p_s, dtype=torch.float32).to(self.device)
            t_a_index = self.converter.act2index(n_a, self.b_s).unsqueeze(axis=-1)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)

            t_p_weight = torch.gather(self.upd_policy(t_p_s), 1, t_a_index)
            weight = torch.log(t_p_weight)
            p_values = torch.transpose(t_r.unsqueeze(-1), 0, 1)
            loss = -torch.matmul(p_values, weight)/self.b_s

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.upd_policy.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            i = i + 1

        return loss
