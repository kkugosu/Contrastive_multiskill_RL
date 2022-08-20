import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class BasePolicy:
    """
    l_r learning rate
    sk_n = skill num
    t_i training iteration
    m_i memory iteration
    s_l state length
    a_l action length
    a_index_l action index length
    policy_name
    converter
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
                 _converter,
                 device
                 ):
        self.l_r = l_r
        self.sk_n = sk_n
        self.t_i = t_i
        self.m_i = m_i
        self.s_l = s_l
        self.a_l = a_l
        self.a_index_l = a_index_l
        self.converter = _converter
        self.device = device





