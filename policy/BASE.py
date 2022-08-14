import gym
from utils import converter
from utils import dataset, dataloader
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class BasePolicy:
    """
    b_s batch_size
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
                 sk_n,
                 t_i,
                 m_i,
                 s_l,
                 a_l,
                 a_index_l,
                 converter
                 ):
        self.l_r = l_r
        self.sk_n = sk_n
        self.t_i = t_i
        self.m_i = m_i
        self.s_l = s_l
        self.a_l = a_l
        self.a_index_l = a_index_l
        self.device = DEVICE
        self.converter = converter





