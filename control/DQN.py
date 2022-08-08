from control import BASE, policy
import torch
from NeuralNetwork import NN
from utils import buffer
from torch import nn
import numpy as np
GAMMA = 0.98


class DQNPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.MainNetwork = NN.ValueNN(self.o_s, self.h_s, self.a_index_s).to(self.device)
        self.baseDQN = NN.ValueNN(self.o_s, self.h_s, self.a_index_s).to(self.device)
        self.baseDQN.eval()
        self.policy = policy.Policy(self.cont, self.MainNetwork, self.converter)
        self.buffer = buffer.Simulate(self.env, self.policy, step_size=self.e_trace, done_penalty=self.d_p)
        self.optimizer = torch.optim.SGD(self.MainNetwork.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss(reduction='mean')

    def get_policy(self):
        return self.policy

    def training(self, load=int(0)):

        if int(load) == 1:
            print("loading")
            self.MainNetwork.load_state_dict(torch.load(self.PARAM_PATH + '/1.pth'))
            self.baseDQN.load_state_dict(self.MainNetwork.state_dict())
            self.baseDQN.eval()
            print("loading complete")
        else:
            pass
        i = 0
        self.buffer.renewal_memory(self.ca, self.data, self.dataloader)
        while i < self.t_i:
            i = i + 1
            self.buffer.renewal_memory(self.ca, self.data, self.dataloader)
            loss = self.train_per_buf()
            if loss < 1:
                break

            print(i)
            print(loss)
            self.writer.add_scalar("dqn/loss", loss, i)
            self.writer.add_scalar("performance", self.buffer.get_performance(), i)
            torch.save(self.MainNetwork.state_dict(), self.PARAM_PATH + '/1.pth')
            self.baseDQN.load_state_dict(self.MainNetwork.state_dict())
            self.baseDQN.eval()

        for p in self.baseDQN.parameters():
            print(p)
        for p in self.MainNetwork.parameters():
            print(p)
        self.env.close()
        self.writer.flush()
        self.writer.close()

    def train_per_buf(self):

        i = 0
        while i < self.m_i:
            n_p_o, n_a, n_o, n_r, n_d = next(iter(self.dataloader))
            t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(self.device)
            t_a_index = self.converter.act2index(n_a, self.b_s).unsqueeze(axis=-1)
            t_o = torch.tensor(n_o, dtype=torch.float32).to(self.device)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)
            t_p_qvalue = torch.gather(self.MainNetwork(t_p_o), 1, t_a_index)
            t_trace = torch.tensor(n_d, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                t_qvalue = self.baseDQN(t_o)
                t_qvalue = torch.max(t_qvalue, dim=1)[0] * (GAMMA**t_trace)
                t_qvalue = t_qvalue + t_r

            loss = self.criterion(t_p_qvalue, t_qvalue.unsqueeze(axis=-1))
            if loss < 1:
                break
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.MainNetwork.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            i = i + 1

        return loss
