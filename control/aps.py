import torch
from NeuralNetwork import basic_nn
import numpy as np
from control import BASE
from torch import nn


class APS(BASE.BaseControl):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.cont_name = "aps"
        self.key = basic_nn.ValueNN(self.s_l, self.s_l, self.sk_n).to(self.device)
        self.query = basic_nn.ValueNN(self.s_l, self.s_l, self.sk_n).to(self.device)
        self.discriminator = basic_nn.ProbNN((self.a_l + self.s_l)*self.sk_n, self.s_l + self.a_l, 1).to(self.device)
        # state + action + skill -> reward
        self.key_optimizer = torch.optim.SGD(self.key.parameters(), lr=self.l_r)
        self.query_optimizer = torch.optim.SGD(self.query.parameters(), lr=self.l_r)
        self.optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=self.l_r)
        self.criterion = nn.MSELoss(reduction='mean')

    def encoder_decoder_training(self, *trajectory):
        n_p_s, n_a, n_s, n_r, n_d, skill_idx = np.squeeze(trajectory)
        t_p_s = torch.from_numpy(n_p_s).to(self.device).type(torch.float32)

        base_batch_batch_matrix = (torch.transpose(self.key(t_p_s), -1, -2) @
                                   self.query(t_p_s)).exp()

        output = torch.gather(base_batch_batch_matrix, 1, torch.from_numpy(np.arange(1000)).squeeze(-1).T)
        bellow = base_batch_batch_matrix.sum(-1) - output
        output = output/bellow
        loss = -torch.sum(output)
        self.key_optimizer.zero_grad()
        self.query_optimizer.zero_grad()
        loss.backward()
        for param in self.key.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.query.parameters():
            param.grad.data.clamp_(-1, 1)
        self.key_optimizer.step()
        self.query_optimizer.step()

    def reward(self, *trajectory):
        # as far as gain more advantage
        n_p_s, n_a, n_s, n_r, n_d, sk_idx = np.squeeze(trajectory)
        t_p_s = torch.from_numpy(n_p_s).to(self.device).type(torch.float32)
        distance_mat = torch.square(self.key(t_p_s) - torch.transpose(self.query(t_p_s), -1, -2))
        with torch.no_grad():
            sorted_mat = torch.sort(distance_mat)
        t_a = torch.from_numpy(n_a).to(self.device).type(torch.float32)
        sk_idx = torch.from_numpy(sk_idx).to(self.device).type(torch.int64)
        sa_len = self.s_l + self.a_l

        tmp = torch.zeros((len(t_p_s), sa_len * self.sk_n))
        i = 0
        while i < len(t_p_s):
            tmp[i][sk_idx[i] * sa_len: (sk_idx[i] + 1) * sa_len] = torch.cat((t_p_s[i], t_a[i]), -1)
            i = i + 1
        # how can i use scatter in here?
        tmp = tmp.type(torch.float32).to(self.device)
        main_value = self.discriminator(tmp).squeeze()

        tmp = torch.zeros((len(t_p_s), self.sk_n, (self.s_l + self.a_l) * self.sk_n))
        i = 0
        while i < len(n_p_s):
            j = 0
            while j < self.sk_n:
                tmp[i][j][j * sa_len: (j + 1) * sa_len] = torch.cat((t_p_s[i], t_a[i]), -1)
                j = j + 1
            i = i + 1
        tmp = tmp.type(torch.float32).to(self.device)
        sub_prob = self.discriminator(tmp).squeeze()
        contrast_value = torch.sum(sub_prob, dim=-1)

        re = torch.log(main_value/contrast_value)
        return (sorted_mat[-10:-1].sum(-1) + re).squeeze()

    def update(self, memory_iter, *trajectory):
        i = 0
        loss1 = None
        loss2_ary = None
        self.encoder_decoder_training(trajectory)
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
