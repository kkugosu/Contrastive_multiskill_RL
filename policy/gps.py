from policy import BASE, ilqr
import torch
import numpy as np
from torch import nn
from NeuralNetwork import basic_nn, bayesian_nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import kl
from utils import converter
GAMMA = 0.98


class GPS(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.i_lqr_step = 3
        self.Dynamics = bayesian_nn.BayesianModel(self.s_l + self.a_l, self.s_l*self.sk_n, self.s_l).to(self.device)
        self.Reward = basic_nn.ValueNN(self.s_l*self.sk_n, self.s_l*self.sk_n, self.a_l**2 + self.a_l + 1).to(self.device)
        self.R_NAF = converter.NAFReward(self.s_l*self.sk_n, self.a_l, self.Reward)
        self.Policy_net = basic_nn.ValueNN(self.s_l*self.sk_n, self.s_l*self.sk_n, self.a_l**2 + self.a_l).to(self.device)
        self.P_NAF = converter.NAFPolicy(self.s_l*self.sk_n, self.a_l, self.Policy_net)
        self.iLQG = ilqr.IterativeLQG(self.Dynamics, self.R_NAF, self.P_NAF, self.s_l, self.a_l,
                                      self.b_s, self.i_lqr_step, self.device)
        self.optimizer_D = torch.optim.SGD(self.Dynamics.parameters(), lr=self.l_r)
        self.optimizer_R = torch.optim.SGD(self.Reward.parameters(), lr=self.l_r)
        self.optimizer_P = torch.optim.SGD(self.Policy_net.parameters(), lr=self.l_r)
        self.criterion = nn.MSELoss(reduction='mean')
        self.lamb = 1
        self.policy_name = "gps"

    def action(self, n_s, index, per_one=1):
        n_s = self.skill_state_converter(n_s, index, per_one=per_one)
        t_s = torch.from_numpy(n_s).type(torch.float32).to(self.device)
        if random.random() < 1.1:
            with torch.no_grad():
                t_a = self.iLQG.get_global_action(t_s)
            n_a = t_a.cpu().numpy()
            return n_a
        else:
            with torch.no_grad():
                t_a = self.iLQG.get_local_action(t_s)
            n_a = t_a.cpu().numpy()
            return n_a

    def update(self, memory_iter=0, *trajectory):
        i = 0
        dyn_loss = None
        rew_loss = None
        self.Dynamics.set_freeze(0)
        if memory_iter != 0:
            self.m_i = memory_iter
        else:
            self.m_i = 1
        while i < self.m_i:
            # print(i)
            n_p_s, n_a, n_s, n_r, n_d, sk_idx = np.squeeze(trajectory)
            t_p_s = torch.tensor(n_p_s, dtype=torch.float32).to(self.device)
            t_a = torch.tensor(n_a, dtype=torch.float32).to(self.device)
            t_s = torch.tensor(n_s, dtype=torch.float32).to(self.device)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)
            sa_in = torch.cat((t_p_s, t_a), dim=-1)
            predict_o = self.Dynamics(sa_in)
            dyn_loss = self.criterion(t_s, predict_o) + self.Dynamics.kld_loss()

            predict_r = self.R_NAF.sa_reward(sa_in)
            print(predict_r)
            rew_loss = self.criterion(t_r, predict_r)

            print(predict_r.size())
            print(t_r.size())

            self.optimizer_D.zero_grad()
            dyn_loss.backward(retain_graph=True)
            for param in self.Dynamics.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_D.step()

            self.optimizer_R.zero_grad()
            rew_loss.backward()
            for param in self.Reward.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_R.step()

            i = i + 1
        print("loss1 = ", dyn_loss)
        print("loss2 = ", rew_loss)

        i = 0
        kld = 0
        self.Dynamics.set_freeze(1)
        while i < self.m_i:
            # print(i)
            n_p_o, n_a, n_o, n_r, n_d, sk_idx = np.squeeze(trajectory)
            n_p_s = self.skill_state_converter(n_p_s, sk_idx)
            t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(self.device)
            t_a = torch.tensor(n_a, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                t_mean, t_var = self.iLQG.fit(t_p_o, t_a, self.b_s)
            # update local policy all we need is action(mean) and var
            mean, var = self.P_NAF.prob(t_p_o)
            mean_d = (mean - t_mean).unsqueeze(-2)
            mean_d_t = torch.transpose(mean_d, -2, -1)
            kld = kld + torch.log(torch.linalg.det(t_var)) - torch.log(torch.linalg.det(var)).squeeze()
            pre_trace = torch.matmul(torch.linalg.inv(t_var), var)
            kld = kld + pre_trace.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).squeeze()
            kld = kld + torch.matmul(torch.matmul(mean_d, torch.linalg.inv(t_var)), mean_d_t).squeeze()
            kld = kld.sum()
            kld = kld * self.lamb
            # kl - divergence - between - two - multivariate - gaussians
            self.optimizer_P.zero_grad()
            kld.backward(retain_graph=True)
            for param in self.Policy_net.parameters():
                param.grad.data.clamp_(-0.1, 0.1)
            self.optimizer_P.step()
            # update global policy
            # update lambda

            i = i + 1
        print("policy loss = ", kld)

        return torch.stack((dyn_loss.squeeze(), rew_loss.squeeze(), kld.squeeze()))

    def load_model(self, path):
        self.Dynamics.load_state_dict(torch.load(path))
        self.Reward.load_state_dict(torch.load(path))
        self.Policy_net.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.Dynamics, path)
        torch.save(self.Reward, path)
        torch.save(self.Policy_net, path)
        return self.Policy_net, self.Reward, self.Dynamics
