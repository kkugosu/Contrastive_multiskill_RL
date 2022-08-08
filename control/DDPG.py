from control import BASE, policy
import gym
import torch
import numpy as np
import sys
from torch import nn
from NeuralNetwork import NN
from utils import buffer
import random
import torch.onnx as onnx
GAMMA = 0.98


class DDPGPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.updatedPG = NN.HopeNN(self.o_s, self.h_s, self.a_s).to(self.device)
        self.updatedDQN = NN.ValueNN(self.o_s + self.a_s, self.h_s, 1).to(self.device)
        self.baseDQN = NN.ValueNN(self.o_s + self.a_s, self.h_s, 1).to(self.device)
        self.baseDQN.eval()
        self.policy = policy.Policy(self.cont, self.updatedPG, self.converter)
        self.buffer = buffer.Simulate(self.env, self.policy, step_size=self.e_trace, done_penalty=self.d_p)
        self.optimizer_p = torch.optim.SGD(self.updatedPG.parameters(), lr=self.lr/100)
        self.optimizer_q = torch.optim.SGD(self.updatedDQN.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss(reduction='mean')

    def get_policy(self):
        return self.policy

    def training(self, load=int(0)):

        if int(load) == 1:
            print("loading")
            self.updatedPG.load_state_dict(torch.load(self.PARAM_PATH + "/1.pth"))
            self.updatedDQN.load_state_dict(torch.load(self.PARAM_PATH + "/2.pth"))
            self.baseDQN.load_state_dict(self.updatedDQN.state_dict())
            self.baseDQN.eval()
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
            torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/1.pth")
            torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/2.pth')
            self.baseDQN.load_state_dict(self.updatedDQN.state_dict())
            self.baseDQN.eval()

        for param in self.updatedDQN.parameters():
            print("----------dqn-------------")
            print(param)
        for param in self.updatedPG.parameters():
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

            n_p_o, n_a, n_o, n_r, n_d = next(iter(self.dataloader))
            t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(self.device)
            t_a = torch.tensor(n_a, dtype=torch.float32).to(self.device)
            t_o = torch.tensor(n_o, dtype=torch.float32).to(self.device)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)
            dqn_input = torch.cat((t_p_o, t_a), dim=-1)
            t_p_qvalue = self.updatedDQN(dqn_input)
            dqn_input_req_grad = torch.cat((t_p_o, self.updatedPG(t_p_o)), dim=-1)
            policy_loss = - torch.mean(self.updatedDQN(dqn_input_req_grad))
            t_trace = torch.tensor(n_d, dtype=torch.float32).to(self.device).unsqueeze(-1)

            with torch.no_grad():
                n_a_expect = self.policy.select_action(n_o)
                t_a_expect = torch.tensor(n_a_expect).to(self.device)
                dqn_input = torch.cat((t_o, t_a_expect), dim=-1)
                t_qvalue = self.baseDQN(dqn_input)*(GAMMA**t_trace) + t_r.unsqueeze(-1)

            queue_loss = self.criterion(t_p_qvalue, t_qvalue)

            self.optimizer_p.zero_grad()
            policy_loss.backward(retain_graph=True)
            for param in self.updatedPG.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_p.step()

            self.optimizer_q.zero_grad()
            queue_loss.backward()
            for param in self.updatedDQN.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_q.step()

            i = i + 1

        print("loss1 = ", policy_loss)
        print("loss2 = ", queue_loss)

        return policy_loss, queue_loss
