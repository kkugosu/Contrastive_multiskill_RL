import gym
from torch.utils.tensorboard import SummaryWriter
from utils import converter
from utils import dataset, dataloader
import torch
from NeuralNetwork import basic_nn
from policy import gps, AC, DDPG, PG, PPO, SAC, TRPO
import numpy as np
import math
from abc import *


class BaseControl(metaclass=ABCMeta):
    """
    l_r : learning rate
    s_l : state length
    policy : policy
    skill_num : skill num
    device : device
    """

    def __init__(self,
                 l_r,
                 s_l,
                 a_l,
                 policy,
                 skill_num,
                 device
                 ):
        self.l_r = l_r
        self.s_l = s_l
        self.a_l = a_l
        self.policy = policy
        self.device = device
        self.skills = skill_num
        self.cont_name = "base"
        self.tmp_state = None

    @abstractmethod
    def reward(self, state_1, state_2, skill, done):
        pass

    def set_initial_state(self, state):
        self.tmp_state = state

    @abstractmethod
    def update(self, memory_iter, *trajectory):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    def name(self):
        return self.cont_name

    def get_policy(self):
        return self.policy
