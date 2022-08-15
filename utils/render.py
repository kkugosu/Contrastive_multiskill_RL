import torch
from policy import BASE
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Render:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env

    def rend(self):
        n_p_o = self.env.reset()
        t = 0
        local = 0
        total_performance = 0
        fail_time = 0
        while t < 1000:
            n_a = self.policy.action(n_p_o)
            n_o, n_r, n_d, info = self.env.step(n_a)
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
