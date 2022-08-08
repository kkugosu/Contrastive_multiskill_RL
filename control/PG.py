from control import BASE, policy
import gym
import torch
import numpy as np
import sys
from torch import nn
from NeuralNetwork import NN
from utils import buffer
import random

GAMMA = 0.98


class PGPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.updatedPG = NN.ProbNN(self.o_s, self.h_s, self.a_index_s).to(self.device)
        self.policy = policy.Policy(self.cont, self.updatedPG, self.converter)
        self.buffer = buffer.Simulate(self.env, self.policy, step_size=self.e_trace, done_penalty=self.d_p)
        self.optimizer = torch.optim.SGD(self.updatedPG.parameters(), lr=self.lr)

    def get_policy(self):
        return self.policy

    def training(self, load=int(0)):

        if int(load) == 1:
            print("loading")
            self.updatedPG.load_state_dict(torch.load(self.PARAM_PATH + '/1.pth'))
            print("loading complete")
        else:
            pass
        i = 0
        while i < self.t_i:
            i = i + 1
            self.buffer.renewal_memory(self.ca, self.data, self.dataloader)
            loss = self.train_per_buff()
            print(i)
            print("loss = ", loss)
            if loss[0][0] > 20:
                break
            self.writer.add_scalar("pg/loss", loss, i)
            self.writer.add_scalar("performance", self.buffer.get_performance(), i)
            torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + '/1.pth')

        for param in self.updatedPG.parameters():
            print("----------pg--------------")
            print(param)

        self.env.close()
        self.writer.flush()
        self.writer.close()

    def train_per_buff(self):
        i = 0
        while i < self.m_i:
            # print(i)
            n_p_o, n_a, n_o, n_r, n_d = next(iter(self.dataloader))
            t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(self.device)
            t_a_index = self.converter.act2index(n_a, self.b_s).unsqueeze(axis=-1)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)

            t_p_weight = torch.gather(self.updatedPG(t_p_o), 1, t_a_index)
            weight = torch.log(t_p_weight)
            p_values = torch.transpose(t_r.unsqueeze(-1), 0, 1)
            loss = -torch.matmul(p_values, weight)/self.b_s

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.updatedPG.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            i = i + 1

        return loss
