from policy import BASE, act
import torch
import numpy as np
from NeuralNetwork import basic_nn
GAMMA = 0.98


class PGPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.upd_policy = basic_nn.ProbNN(self.s_l*self.sk_n, self.s_l*self.sk_n, self.a_index_l).to(self.device)
        self.optimizer = torch.optim.SGD(self.upd_policy.parameters(), lr=self.l_r)

    def action(self, t_p_s):
        with torch.no_grad():
            probability = self.upd_policy(t_p_s)

        t_a_index = torch.multinomial(probability, 1)
        n_a = self.converter.index2act(t_a_index.squeeze(-1), 1)
        return n_a

    def update(self, *trajectory):
        i = 0
        loss = 0
        while i < self.m_i:
            # print(i)
            n_p_s, n_a, n_s, n_r, n_d = trajectory[0] # next(iter(self.dataloader))
            t_p_s = torch.tensor(n_p_s, dtype=torch.float32).to(self.device)
            t_a_index = self.converter.act2index(n_a).unsqueeze(axis=-1)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)

            t_p_weight = torch.gather(self.upd_policy(t_p_s), 1, t_a_index)
            weight = torch.log(t_p_weight)
            p_values = torch.transpose(t_r.unsqueeze(-1), 0, 1)
            loss = -torch.matmul(p_values, weight)

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.upd_policy.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            i = i + 1
        return loss

    def load_model(self, path):
        self.upd_policy.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.upd_policy, path)
        return self.upd_policy
