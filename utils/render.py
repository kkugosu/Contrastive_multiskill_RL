import torch
from policy import BASE
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np


class Render:
    def __init__(self, policy, index, skill_num, env):
        self.policy = policy
        self.index = index
        self.skill_num = skill_num
        self.env = env

    def rend(self):
        n_p_o = self.env.reset()
        tmp_n_p_o = np.zeros(len(n_p_o) * self.skill_num)
        tmp_n_p_o[self.index * len(n_p_o):(self.index + 1) * len(n_p_o)] = n_p_o
        n_p_o = tmp_n_p_o
        t = 0
        local = 0
        total_performance = 0
        fail_time = 0
        while t < 1000:
            t_p_o = torch.from_numpy(n_p_o).type(torch.float32).to(device)
            n_a = self.policy.action(t_p_o, per_one=1)
            n_o, n_r, n_d, info = self.env.step(n_a)
            tmp_n_o = np.zeros(len(n_o) * self.skill_num)
            tmp_n_o[self.index * len(n_o):(self.index + 1) * len(n_o)] = n_o
            n_o = tmp_n_o
            total_performance = total_performance + n_r
            n_p_o = n_o
            self.env.render()
            t = t + 1
            local = local + 1
            if n_d:
                print("Episode finished after {} timesteps".format(local+1))
                fail_time = fail_time + 1
                local = 0
                self.env.reset()
        print("performance = ", total_performance/fail_time)
