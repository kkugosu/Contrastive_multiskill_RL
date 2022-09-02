from policy import BASE
import torch
import numpy as np
from NeuralNetwork import basic_nn
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GAMMA = 0.98


class PGPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.upd_policy = basic_nn.ProbNN(self.s_l*self.sk_n, self.s_l*self.sk_n, self.a_index_l).to(self.device)
        self.optimizer = torch.optim.SGD(self.upd_policy.parameters(), lr=self.l_r)
        self.policy_name = "PG"

    def action(self, t_s, per_one=1):
        with torch.no_grad():
            probability = self.upd_policy(t_s)

        t_a_index = torch.multinomial(probability, 1)
        if per_one == 0:
            n_a = self.converter.index2act(t_a_index.squeeze(-1), per_one)
        else:
            n_a = self.converter.index2act(t_a_index.squeeze(-1))
        return n_a

    def update(self, memory_iter=0, *trajectory):
        i = 0
        loss = 0
        if memory_iter != 0:
            self.m_i = memory_iter
        else:
            self.m_i = 1
        while i < self.m_i:
            # print(i)

            n_p_s, n_a, n_s, n_r, n_d, sk_idx = np.squeeze(trajectory) # next(iter(self.dataloader))
            n_p_s = self.skill_state_converter(n_p_s, sk_idx)
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
        return loss.squeeze().unsqueeze(-1)

    def load_model(self, path):
        self.upd_policy.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.upd_policy.state_dict(), path)
        return self.upd_policy,
