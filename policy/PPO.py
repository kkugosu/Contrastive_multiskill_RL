from policy import BASE
import torch
import numpy as np
from torch import nn
from NeuralNetwork import basic_nn
GAMMA = 0.98


class PPOPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.upd_policy = basic_nn.ProbNN(self.s_l*self.sk_n, self.s_l*self.sk_n, self.a_index_l).to(self.device)
        self.base_policy = basic_nn.ProbNN(self.s_l*self.sk_n, self.s_l*self.sk_n, self.a_index_l).to(self.device)
        self.upd_queue = basic_nn.ValueNN(self.s_l*self.sk_n, self.s_l*self.sk_n, self.a_index_l).to(self.device)
        self.base_queue = basic_nn.ValueNN(self.s_l*self.sk_n, self.s_l*self.sk_n, self.a_index_l).to(self.device)
        self.optimizer_p = torch.optim.SGD(self.upd_policy.parameters(), lr=self.l_r)
        self.optimizer_q = torch.optim.SGD(self.upd_queue.parameters(), lr=self.l_r)
        self.criterion = nn.MSELoss(reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def action(self, t_s, per_one=1):
        with torch.no_grad():
            probability = self.upd_policy(t_s)

        t_a_index = torch.multinomial(probability, 1)
        if per_one == 0:
            n_a = self.converter.index2act(t_a_index.squeeze(-1), per_one)
        else:
            n_a = self.converter.index2act(t_a_index.squeeze(-1))
        return n_a

    def update(self, *trajectory):
        i = 0
        queue_loss = None
        policy_loss = None
        self.base_queue.load_state_dict(self.upd_queue.state_dict())
        self.base_queue.eval()
        self.base_policy.load_state_dict(self.upd_queue.state_dict())
        self.base_policy.eval()
        while i < self.m_i:
            # print(i)
            n_p_s, n_a, n_s, n_r, n_d, sk_idx = np.squeeze(trajectory)
            t_p_s = torch.tensor(n_p_s, dtype=torch.float32).to(self.device)
            t_a_index = self.converter.act2index(n_a).unsqueeze(axis=-1)
            t_s = torch.tensor(n_s, dtype=torch.float32).to(self.device)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)
            t_p_weight = torch.gather(self.upd_policy(t_p_s), 1, t_a_index)
            t_p_base_weight = torch.gather(self.base_policy(t_p_s), 1, t_a_index)
            ratio = t_p_weight/t_p_base_weight
            # at first step, ratio will be 1
            t_p_qvalue = torch.gather(self.upd_queue(t_p_s), 1, t_a_index)
            weight = torch.transpose(ratio, 0, 1)

            # we can restrict result of network by clamp but we can't restrict network parameter which result in fluctuate
            # so we add action kld term like trpo
            state_entropy_bonus = -torch.sum(torch.log(self.upd_policy(t_p_s)) * self.upd_policy(t_p_s))
            policy_loss = -torch.matmul(weight, t_p_qvalue) - state_entropy_bonus*0.1
            t_trace = torch.tensor(n_d, dtype=torch.float32).to(self.device).unsqueeze(-1)

            with torch.no_grad():
                n_a_expect = self.action(t_s, 0)
                t_a_index = self.converter.act2index(n_a_expect).unsqueeze(-1)
                t_qvalue = torch.gather(self.base_queue(t_s), 1, t_a_index)
                t_qvalue = t_qvalue*(GAMMA**t_trace) + t_r.unsqueeze(-1)

            queue_loss = self.criterion(t_p_qvalue, t_qvalue)

            self.optimizer_p.zero_grad()
            policy_loss.backward(retain_graph=True)
            for param in self.upd_policy.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_p.step()

            self.optimizer_q.zero_grad()
            queue_loss.backward()
            for param in self.upd_queue.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_q.step()

            with torch.no_grad():
                tmp_a_distribution = self.base_policy(t_p_s).clone().detach()
            kl_pg_loss = self.kl_loss(torch.log(self.upd_policy(t_p_s)), tmp_a_distribution)
            self.optimizer_p.zero_grad()
            kl_pg_loss.backward()
            for param in self.upd_policy.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_p.step()
            i = i + 1
        print("loss1 = ", policy_loss)
        print("loss2 = ", queue_loss)

        return [policy_loss, queue_loss]

    def load_model(self, path):
        self.upd_policy.load_state_dict(torch.load(path))
        self.upd_queue.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.upd_policy, path)
        torch.save(self.upd_queue, path)
        return self.upd_policy, self.upd_queue
