from policy import BASE, act
import gym
import torch
import numpy as np
import sys
from torch import nn
from NeuralNetwork import basic_nn
from utils import buffer
import random
import torch.onnx as onnx
GAMMA = 0.98


class SACPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.upd_policy = basic_nn.ProbNN(self.o_s, self.h_s, self.a_index_s).to(self.device)
        self.upd_queue = basic_nn.ValueNN(self.o_s, self.h_s, self.a_index_s).to(self.device)
        self.base_queue = basic_nn.ValueNN(self.o_s, self.h_s, self.a_index_s).to(self.device)
        self.base_queue.eval()
        self.policy = policy.Policy(self.cont, self.upd_policy, self.converter)
        self.buffer = buffer.Simulate(self.env, self.policy, step_size=self.e_trace, done_penalty=self.d_p)
        self.optimizer_p = torch.optim.SGD(self.upd_policy.parameters(), lr=self.lr)
        self.optimizer_q = torch.optim.SGD(self.upd_queue.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss(reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def get_policy(self):
        return self.policy

    def training(self, load=int(0)):

        if int(load) == 1:
            print("loading")
            self.upd_policy.load_state_dict(torch.load(self.PARAM_PATH + "/1.pth"))
            self.upd_queue.load_state_dict(torch.load(self.PARAM_PATH + "/2.pth"))
            self.base_queue.load_state_dict(self.upd_queue.state_dict())
            self.base_queue.eval()
            print("loading complete")
        else:
            pass
        i = 0
        while i < self.t_i:
            print(i)

            i = i + 1
            self.buffer.renewal_memory(self.ca, self.data, self.dataloader)
            pg_loss, dqn_loss = self.train_per_buff()
            self.writer.add_scalar("pg/loss", pg_loss, i)
            self.writer.add_scalar("dqn/loss", dqn_loss, i)
            self.writer.add_scalar("performance", self.buffer.get_performance(), i)
            torch.save(self.upd_policy.state_dict(), self.PARAM_PATH + "/1.pth")
            torch.save(self.upd_queue.state_dict(), self.PARAM_PATH + '/2.pth')
            self.base_queue.load_state_dict(self.upd_queue.state_dict())
            self.base_queue.eval()

        for param in self.upd_queue.parameters():
            print("----------dqn-------------")
            print(param)
        for param in self.upd_policy.parameters():
            print("----------pg--------------")
            print(param)

        self.env.close()
        self.writer.flush()
        self.writer.close()

    def train_per_buff(self):
        i = 0
        queue_loss = None
        policy_loss = None
        while i < self.m_i:
            # print(i)
            n_p_o, n_a, n_o, n_r, n_d = next(iter(self.dataloader))
            t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(self.device)
            t_a_index = self.converter.act2index(n_a, self.b_s).unsqueeze(axis=-1)
            t_o = torch.tensor(n_o, dtype=torch.float32).to(self.device)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)
            # t_p_weight = torch.gather(self.updatedPG(t_p_o), 1, t_a_index)
            t_p_qvalue = torch.gather(self.upd_queue(t_p_o), 1, t_a_index)
            # policy_loss = torch.mean(torch.log(t_p_weight) - t_p_qvalue)
            # we already sampled according to policy

            policy_loss = self.kl_loss(self.log_softmax(self.base_queue(t_p_o)), self.upd_policy(t_p_o))

            t_trace = torch.tensor(n_d, dtype=torch.float32).to(self.device).unsqueeze(-1)

            with torch.no_grad():
                n_a_expect = self.policy.select_action(n_o)
                t_a_index = self.converter.act2index(n_a_expect, self.b_s).unsqueeze(-1)
                t_qvalue = torch.gather(self.base_queue(t_o), 1, t_a_index)
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

            i = i + 1
        print("loss1 = ", policy_loss)
        print("loss2 = ", queue_loss)

        return policy_loss, queue_loss
