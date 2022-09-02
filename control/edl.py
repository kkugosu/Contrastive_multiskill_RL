import torch
from NeuralNetwork import basic_nn
import numpy as np
from control import BASE
from torch import nn


class EDL(BASE.BaseControl):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.cont_name = "edl"
        self.encoder = basic_nn.ProbNN(self.s_l, self.s_l + self.sk_n, self.sk_n)
        self.decoder = basic_nn.ProbNN(self.sk_n, self.s_l + self.sk_n, self.s_l)
        self.optimizer_e = torch.optim.SGD(self.encoder.parameters(), lr=self.l_r)
        self.optimizer_d = torch.optim.SGD(self.decoder.parameters(), lr=self.l_r)
        self.criterion = nn.MSELoss(reduction='mean')

    def reward(self, *trajectory):
        n_p_s, n_a, n_s, n_r, n_d, skill_idx = np.squeeze(trajectory)
        return self.decoder(n_p_s, skill_idx)

    def encoder_decoder_training(self, *trajectory):
        n_p_s, n_a, n_s, n_r, n_d, skill_idx = np.squeeze(trajectory)
        skill = self.encoder(n_p_s)
        skill = skill + torch.randn_like(skill) * 0.1
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
            loss1 = self.encoder_decoder_training(trajectory)
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

