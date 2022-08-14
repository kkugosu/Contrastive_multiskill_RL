import gym
from torch.utils.tensorboard import SummaryWriter
from utils import converter
from utils import dataset, dataloader
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class BasePolicy:
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
                 m_c,
                 h_s,
                 l_r,
                 t_i,
                 m_i,
                 policy,
                 env_n,
                 e_trace,
                 precision,
                 d_p
                 ):
        self.b_s = b_s
        self.m_c = m_c
        self.h_s = h_s
        self.l_r = l_r
        self.t_i = t_i
        self.m_i = m_i
        self.policy = policy
        self.env_n = env_n
        self.e_trace = e_trace
        self.precision = precision
        self.device = DEVICE
        self.d_p = d_p

        if self.env_n == "cart":
            self.env = gym.make('CartPole-v1')
        elif self.env_n == "hope":
            self.env = gym.make('Hopper-v3')
        else:
            self.env = gym.make('Hopper-v3')

        self.s_l = len(self.env.observation_space.sample())
        print("state_space = ", self.env.observation_space)
        print("STATE_SIZE(input) = ", self.s_l)

        if self.env_n == "cart":
            self.a_l = 2
            self.a_index_l = 2
        else:
            self.a_l = len(self.env.action_space.sample())
            self.a_index_l = self.precision ** self.a_l
        print("action_space = ", self.env.action_space)
        print("ACTION_SIZE(output) = ", self.a_l)
        print("ACTION_INDEX_SIZE(output) = ", self.a_index_l)

        self.converter = converter.IndexAct(self.env_n, self.a_l, self.precision)

