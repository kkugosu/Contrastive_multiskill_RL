from abc import *


class BaseControl(metaclass=ABCMeta):
    """
    l_r : learning rate
    s_l : state length
    a_l : action length
    policy : policy
    sk_n : skill num
    device : device
    """

    def __init__(self,
                 l_r,
                 s_l,
                 a_l,
                 policy,
                 sk_n,
                 device
                 ):
        self.l_r = l_r
        self.s_l = s_l
        self.a_l = a_l
        self.policy = policy
        self.sk_n = sk_n
        self.device = device
        self.cont_name = "base"

    @abstractmethod
    def reward(self, state_1, state_2, skill, done):
        pass

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
