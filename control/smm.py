import torch
from NeuralNetwork import basic_nn
import numpy as np
from control import BASE


class SMM(BASE.BaseControl):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.cont_name = "smm"
        self.discriminator = basic_nn.ProbNN(self.s_l, self.s_l * self.sk_n, self.sk_n).to(self.device)
        # state -> skill
        self.optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=self.l_r)

    def reward(self, *trajectory):
        n_p_s, n_a, n_s, n_r, n_d, skill_idx = np.squeeze(trajectory)

        distance = np.square(np.repeat(np.expand_dims(n_p_s, 0), len(n_p_s), axis=0) -
                             np.repeat(np.expand_dims(n_p_s, 1), len(n_p_s), axis=1))
        density = np.exp(-np.sum(distance, -1))
        density = torch.from_numpy(density).to(self.device)

        reward = torch.sum(density, -1).squeeze()

        n_p_s, n_a, n_s, n_r, n_d, skill_idx = np.squeeze(trajectory)
        skill_idx = torch.from_numpy(skill_idx).to(self.device).type(torch.int64)
        t_p_s = torch.from_numpy(n_p_s).to(self.device).type(torch.float32)
        skill_idx = skill_idx.unsqueeze(-1)
        out = torch.gather(self.discriminator(t_p_s), 1, skill_idx).squeeze()

        return - reward + torch.log(out)

    def update(self, memory_iter, *trajectory):
        i = 0
        loss1 = None
        loss2_ary = None
        while i < memory_iter:
            i = i + 1
            loss2_ary = self.policy.update(1, trajectory)
            out = self.reward(trajectory)
            loss1 = - torch.sum(out)
            self.optimizer.zero_grad()
            loss1.backward()
            for param in self.discriminator.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        loss_ary = torch.cat((loss2_ary, loss1.unsqueeze(0)), -1)
        return loss_ary

    def load_model(self, path):
        self.discriminator.load_state_dict(torch.load(path + self.cont_name))
        self.policy.load_model(path)

    def save_model(self, path):
        torch.save(self.discriminator.state_dict(), path + self.cont_name)
        models = self.policy.save_model(path)
        return (self.discriminator,) + models
