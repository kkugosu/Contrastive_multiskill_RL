from policy import BASE
import torch
import numpy as np
from torch import nn
from NeuralNetwork import basic_nn
GAMMA = 0.98


class DDPGPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.upd_policy = basic_nn.ValueNN(self.s_l*self.sk_n, self.s_l*self.sk_n, self.a_l).to(self.device)
        self.upd_queue = basic_nn.ValueNN(self.s_l*self.sk_n + self.a_l, self.s_l*self.sk_n, 1).to(self.device)
        self.base_queue = basic_nn.ValueNN(self.s_l*self.sk_n + self.a_l, self.s_l*self.sk_n, 1).to(self.device)
        self.optimizer_p = torch.optim.SGD(self.upd_policy.parameters(), lr=self.l_r)
        self.optimizer_q = torch.optim.SGD(self.upd_queue.parameters(), lr=self.l_r)
        self.criterion = nn.MSELoss(reduction='mean')

    def action(self, t_s):
        with torch.no_grad():
            t_a = self.upd_policy(t_s)
        n_a = t_a.cpu().numpy()
        return n_a

    def update(self, *trajectory):
        i = 0
        queue_loss = None
        policy_loss = None
        self.base_queue.load_state_dict(self.upd_queue.state_dict())
        self.base_queue.eval()
        while i < self.m_i:
            n_p_s, n_a, n_s, n_r, n_d = trajectory[0]
            t_p_s = torch.tensor(n_p_s, dtype=torch.float32).to(self.device)
            t_a = torch.tensor(n_a, dtype=torch.float32).to(self.device)
            t_s = torch.tensor(n_s, dtype=torch.float32).to(self.device)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)
            dqn_input = torch.cat((t_p_s, t_a), dim=-1)
            t_p_qvalue = self.upd_queue(dqn_input)
            dqn_input_req_grad = torch.cat((t_p_s, self.upd_policy(t_p_s)), dim=-1)
            policy_loss = - torch.mean(self.upd_queue(dqn_input_req_grad))
            t_trace = torch.tensor(n_d, dtype=torch.float32).to(self.device).unsqueeze(-1)

            with torch.no_grad():
                n_a_expect = self.action(t_s)
                t_a_expect = torch.tensor(n_a_expect).to(self.device)
                dqn_input = torch.cat((t_s, t_a_expect), dim=-1)
                t_qvalue = self.base_queue(dqn_input)*(GAMMA**t_trace) + t_r.unsqueeze(-1)

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
            i = i + 1

        print("loss1 = ", policy_loss)
        print("loss2 = ", queue_loss)

        return policy_loss, queue_loss

    def load_model(self, path):
        self.upd_policy.load_state_dict(torch.load(path))
        self.upd_queue.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.upd_policy, path)
        torch.save(self.upd_queue, path)
        return self.upd_policy, self.upd_queue
