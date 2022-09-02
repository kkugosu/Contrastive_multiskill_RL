import torch
from NeuralNetwork import basic_nn
import numpy as np
from control import BASE


class VIC(BASE.BaseControl):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.cont_name = "vic"
        self.discriminator = basic_nn.ProbNN(self.s_l + self.s_l, self.s_l * self.sk_n, self.sk_n).to(
            self.device)
        # first_state + current_state -> skill
        self.optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=self.l_r)

    def reward(self, *trajectory):
        n_p_s, n_a, n_s, n_r, n_d, skill_idx = np.squeeze(trajectory)
        t_p_s = torch.from_numpy(n_p_s).to(self.device).type(torch.float32)
        i = 0
        skill_maybe = torch.zeros(size=(1000, self.sk_n))
        initial_skill_maybe = torch.zeros(size=(1000, self.sk_n)) + 1/self.sk_n
        tmp_state = t_p_s[0].unsqueeze(0)
        while i < len(n_d):
            if n_d[i] == 1:
                tmp_state = t_p_s[i].unsqueeze(0)
                initial_pair = torch.cat((tmp_state, tmp_state), 0)
                initial_skill_maybe[i] = self.discriminator(initial_pair)
            current_pair = torch.cat((tmp_state, t_p_s[i].unsqueeze(0)), 0)
            skill_maybe[i] = self.discriminator(current_pair)
            i = i + 1
        entropy = torch.sum(-initial_skill_maybe*torch.log(initial_skill_maybe))
        out = torch.gather(self.discriminator(skill_maybe), 1, skill_idx)
        return out + entropy

    def update(self, memory_iter, *trajectory):
        i = 0
        loss1 = None
        loss2_ary = None
        while i < memory_iter:
            i = i + 1
            loss2_ary = self.policy.update(1, trajectory)
            loss1 = self.reward(trajectory)
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
