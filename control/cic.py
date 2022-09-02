import torch
from NeuralNetwork import basic_nn
import numpy as np
from control import BASE


class CIC(BASE.BaseControl):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.cont_name = "cic"
        self.key = basic_nn.ValueNN(self.s_l, self.s_l, self.sk_n).to(self.device)
        self.query = basic_nn.ValueNN(self.s_l, self.s_l, self.sk_n).to(self.device)
        self.key_optimizer = torch.optim.SGD(self.key.parameters(), lr=self.l_r)
        self.query_optimizer = torch.optim.SGD(self.query.parameters(), lr=self.l_r)

    def encoder_decoder_training(self, *trajectory):
        n_p_s, n_a, n_s, n_r, n_d, skill_idx = np.squeeze(trajectory)
        base_batch_batch_matrix = torch.matmul(self.key(n_p_s + n_s).T, self.query(all_skills)).exp()
        output = torch.gather(base_batch_batch_matrix, 1, skill_idx)
        bellow = base_batch_batch_matrix.sum(-1) - output
        # size = batchsize*1
        output = output/bellow
        loss = -output/len(n_p_s)
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
        n_p_s, n_a, n_s, n_r, n_d, skill_idx = np.squeeze(trajectory)
        distance_mat = torch.square(self.key(n_p_s + n_s) - self.key(n_p_s + n_s).T)
        sorted_mat = torch.sort(distance_mat)
        return sorted_mat[-10:-1].sum(-1)

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
