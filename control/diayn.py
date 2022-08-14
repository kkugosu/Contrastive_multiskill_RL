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
                 ca,
                 h_s,
                 lr,
                 t_i,
                 m_i,
                 cont,
                 env_n,
                 d_p
                 ):
        self.b_s = b_s
        self.ca = ca
        self.h_s = h_s
        self.lr = lr
        self.t_i = t_i
        self.m_i = m_i
        self.cont = cont
        self.env_n = env_n
        self.device = DEVICE
        self.d_p = d_p

        self.skills = np.empty((10,), dtype='object')
        i = 0
        while i < len(self.skills):
            self.skills[i] = PG.PGPolicy()
            i = i + 1
        self.discriminator = basic_nn.ValueNN(self.o_s + len(self.skills), len(self.skills), len(self.skills)).to(self.device)

    def reward(self, state, skill):
        sa = torch.cat((state, skill), 0)
        return self.discriminator(sa)[skill] - (1/100)

    def loss(self):
        return -reward()

    def update(self):
        self.policy.update()