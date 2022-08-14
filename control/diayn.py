import gym
from torch.utils.tensorboard import SummaryWriter
from utils import converter
from utils import dataset, dataloader
import torch
from NeuralNetwork import basic_nn
from policy import gps, AC, DDPG, PG, PPO, SAC, TRPO
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DIAYN:
    """
    b_s batch_size
    ca capacity
    o_s observation space
    a_s action space
    h_s hidden space
    lr learning rate
    t_i training iteration
    cont policy
    env_n environment name
    """

    def __init__(self,
                 l_r,
                 s_l,
                 policy,
                 ):
        self.l_r = l_r
        self.s_l = s_l
        self.policy = policy
        self.device = DEVICE

        self.skills = 10

        self.discriminator = basic_nn.ProbNN(self.s_l * self.skills, self.s_l * self.skills, self.skills).to(self.device)
        self.optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=self.l_r)

    def reward(self, s_k, skill):
        return self.discriminator(s_k)[skill] - (1/100)

    def update(self, trajectary):
        self.policy.update(trajectary)
        n_p_s, n_a, n_s, n_r, n_d, skill_idx = trajectary
        loss = - self.discriminator(n_p_s)[skill_idx] + (1/100)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.discriminator.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

