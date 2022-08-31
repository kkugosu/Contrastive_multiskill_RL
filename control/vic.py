import gym
from torch.utils.tensorboard import SummaryWriter
from utils import converter
from utils import dataset, dataloader
import torch
from NeuralNetwork import basic_nn
from policy import gps, AC, DDPG, PG, PPO, SAC, TRPO
import numpy as np
import math
from control import BASE


class VIC(BASE.BaseControl):
    """
    l_r : learning rate
    s_l : state length
    policy : policy
    skill_num : skill num
    device : device
    """
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.cont_name = "vic"
        self.discriminator = basic_nn.ProbNN(self.skills + self.s_l, self.s_l * self.skills, self.skills).to(
            self.device)
        self.optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=self.l_r)
        self.initial_state = None

    def reward(self, state_1, state_2, skill, done):
        if n_d == 0:
            return 0
        else:
            return torch.log(self.discriminator(s_k + self.initial_state)[skill])

    def state_entropy(self, s_k, n_d):
        entropy = - torch.dot(self.discriminator(s_k + s_k), torch.log(self.discriminator(s_k + s_k)))
        out = torch.gather(entropy, 1, n_d)
        return out

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
            skill_idx = torch.from_numpy(skill_idx).to(self.device).type(torch.int64)
            t_p_s = torch.from_numpy(n_p_s).to(self.device).type(torch.float32)
            skill_idx = skill_idx.unsqueeze(-1)
            out = self.discriminator(pair)[skill_idx]
            out = out - self.state_entropy(t_p_s, n_d)
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
