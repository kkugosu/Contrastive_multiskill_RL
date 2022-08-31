
from utils import converter
import torch
from NeuralNetwork import basic_nn
import numpy as np
import math
from control import BASE


class CIC(BASE.BaseControl):
    """
    l_r : learning rate
    s_l : state length
    policy : policy
    skill_num : skill num
    device : device
    """
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.cont_name = "cic"
        self.key = basic_nn.ValueNN(self.s_l, self.s_l, self.skills).to(self.device)
        self.query = basic_nn.ValueNN(self.s_l, self.s_l, self.skills).to(self.device)
        self.discriminator = basic_nn.ProbNN(2*self.s_l*self.skills, self.s_l * self.skills, self.skills).to(self.device)
        self.optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=self.l_r)
        self.key_optimizer = torch.optim.SGD(self.key.parameters(), lr=self.l_r)
        self.query_optimizer = torch.optim.SGD(self.query.parameters(), lr=self.l_r)
        self.initial_state = None

    def reward(self, state_1, state_2, skill, done):
        # state1 + state2 -> skills

        return self.discriminator(s_k_1 + s_k_2)[skill]
        # we need to revise later

    def state_encoding(self, *trajectory):
        n_p_s, n_a, n_s, n_r, n_d, skill_idx = np.squeeze(trajectory)
        base_batch_batch_matrix = torch.matmul(self.key(n_p_s + n_s).T, self.query(all_skills)).exp()
        output = torch.gather(base_batch_batch_matrix, 1, skill_idx)
        bellow = base_batch_batch_matrix.sum(-1) - output
        # size = batchsize*1
        output = output/bellow
        loss = -output/len(n_p_s)
        self.key_optimizer.zero_grad()
        self.query_optimizer.zero_grad()
        loss.backward()
        for param in self.key.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.query.parameters():
            param.grad.data.clamp_(-1, 1)
        self.key_optimizer.step()
        self.query_optimizer.step()

    def state_penalty(self, *trajectory):
        # as far as gain more advantage
        n_p_s, n_a, n_s, n_r, n_d, skill_idx = np.squeeze(trajectory)
        distance_mat = torch.square(self.key(n_p_s + n_s) - self.key(n_p_s + n_s).T)
        sorted_mat = torch.sort(distance_mat)
        return sorted_mat[-10:-1].sum(-1)

    def update(self, memory_iter, *trajectory):
        i = 0
        loss1 = None
        loss2_ary = None
        self.state_encoding(trajectory)
        while i < memory_iter:
            i = i + 1
            loss2_ary = self.policy.update(1, trajectory)
            n_p_s, n_a, n_s, n_r, n_d, skill_idx = np.squeeze(trajectory)
            skill_idx = torch.from_numpy(skill_idx).to(self.device).type(torch.int64)
            t_p_s = torch.from_numpy(n_p_s).to(self.device).type(torch.float32)
            skill_idx = skill_idx.unsqueeze(-1)
            out = self.reward(t_p_o, n_s, skill_idx)
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
