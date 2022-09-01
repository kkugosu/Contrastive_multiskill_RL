
from utils import converter
import torch
from NeuralNetwork import basic_nn
import numpy as np
import math
from torch import nn
from control import BASE


class VALOR(BASE.BaseControl):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.cont_name = "valor"
        self.discriminator = basic_nn.ProbNN(self.s_l, self.s_l * self.skills, self.skills).to(self.device)
        self.optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=self.l_r)
        self.bid_lstm = nn.LSTM(input_size=self.s_l, hidden_size=self.s_l, bidirectional=True)

    def reward(self, *trajectory):
        # paring first
        return torch.log(self.discriminator(s_k)[skill]) - math.log((1/self.skills))

    def get_index_pair(self, n_p_s, n_d, skill_idx):
        return pair, skill_idx

    def update(self, memory_iter, *trajectory):
        i = 0
        loss1 = None
        loss2_ary = None
        while i < memory_iter:
            i = i + 1
            loss2_ary = self.policy.update(1, trajectory)
            n_p_s, n_a, n_s, n_r, n_d, skill_idx = np.squeeze(trajectory)
            pair, skill_idx = self.get_index_pair(n_p_s, n_d, skill_idx)
            output_state = self.bid_lstm(pair)
            skill_idx = torch.from_numpy(skill_idx).to(self.device).type(torch.int64)
            skill_idx = skill_idx.unsqueeze(-1)
            out = torch.gather(self.discriminator(output_state), 1, skill_idx)
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

