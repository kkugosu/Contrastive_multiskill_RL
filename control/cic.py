import torch
from NeuralNetwork import basic_nn
import numpy as np
from control import BASE


class CIC(BASE.BaseControl):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.cont_name = "cic"
        self.key = basic_nn.ValueNN(2 * self.s_l, self.s_l, self.sk_n).to(self.device)
        self.query = basic_nn.ValueNN(self.sk_n, self.sk_n, self.sk_n).to(self.device)
        self.key_optimizer = torch.optim.SGD(self.key.parameters(), lr=self.l_r)
        self.query_optimizer = torch.optim.SGD(self.query.parameters(), lr=self.l_r)

    def encoder_decoder_training(self, *trajectory):
        n_p_s, n_a, n_s, n_r, n_d, sk_idx = np.squeeze(trajectory)
        t_p_s = torch.from_numpy(n_p_s).to(self.device).type(torch.float32)
        t_s = torch.from_numpy(n_s).to(self.device).type(torch.float32)
        sk_idx = torch.from_numpy(sk_idx).to(self.device).type(torch.int64)
        state_pair = torch.cat((t_p_s, t_s), -1)
        all_skill = torch.eye(10).expand(1000, 10, 10)
        base_batch_batch_matrix = torch.matmul(self.key(state_pair).unsqueeze(-2), self.query(all_skill)).exp()
        output = torch.gather(base_batch_batch_matrix, 1, sk_idx)
        bellow = base_batch_batch_matrix.sum(-1) - output
        # size = batchsize*1
        output = output/bellow
        loss = -torch.sum(output)
        self.key_optimizer.zero_grad()
        self.query_optimizer.zero_grad()
        loss.backward()
        for param in self.key.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.query.parameters():
            param.grad.data.clamp_(-1, 1)
        self.key_optimizer.step()
        self.query_optimizer.step()

    def reward(self, *trajectory):
        # as far as gain more advantage
        n_p_s, n_a, n_s, n_r, n_d, sk_idx = np.squeeze(trajectory)
        t_p_s = torch.from_numpy(n_p_s).to(self.device).type(torch.float32)
        t_s = torch.from_numpy(n_s).to(self.device).type(torch.float32)
        state_pair = torch.cat((t_p_s, t_s), -1)
        distance_mat = torch.square(self.key(state_pair) - self.key(state_pair).T)
        sorted_mat = torch.sort(distance_mat)
        return sorted_mat[-10:-1].sum(-1).squeeze()

    def update(self, memory_iter, *trajectory):
        i = 0
        loss1 = None
        loss2_ary = None
        self.encoder_decoder_training(trajectory)
        while i < memory_iter:
            i = i + 1
            loss2_ary = self.policy.update(1, trajectory)
            out = self.reward(trajectory)
            loss1 = - torch.sum(torch.log(out))
            self.key_optimizer.zero_grad()
            loss1.backward()
            for param in self.key.parameters():
                param.grad.data.clamp_(-1, 1)
            self.key_optimizer.step()
        loss_ary = torch.cat((loss2_ary, loss1.unsqueeze(0)), -1)
        return loss_ary

    def load_model(self, path):
        self.key.load_state_dict(torch.load(path + self.cont_name))
        self.query.load_state_dict(torch.load(path + self.cont_name))
        self.policy.load_model(path)

    def save_model(self, path):
        torch.save(self.key.state_dict(), path + self.cont_name)
        torch.save(self.query.state_dict(), path + self.cont_name)
        models = self.policy.save_model(path)
        return (self.key, self.query) + models
