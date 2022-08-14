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
                 b_s,
                 h_s,
                 s_l,
                 l_r,
                 cont,
                 ):
        self.b_s = b_s
        self.h_s = h_s
        self.l_r = l_r
        self.s_l = s_l
        self.cont = cont
        self.device = DEVICE

        self.skills = 10

        self.discriminator = basic_nn.ValueNN(self.s_l * self.skills, self.s_l * self.skills, self.skills).to(self.device)

    def reward(self, state, skill):
        sa = torch.cat((state, skill), 0)
        return self.discriminator(sa)[skill] - (1/100)

    def loss(self):
        return -reward()

    def update(self):
        self.policy.update()
