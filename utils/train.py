import torch
from utils import buffer, dataset, dataloader
from torch.utils.tensorboard import SummaryWriter
from control import diayn
import numpy as np


class Train:
    def __init__(self, train_iter, memory_iter, batch_size, skill_n, policy,
                 capacity, env, cont, env_n, e_trace, d_p, load_):

        self.t_i = train_iter
        self.m_i = memory_iter
        self.capacity = capacity
        self.policy = policy
        self.b_s = batch_size
        self.cont = cont
        self.env = env
        self.skill_num = skill_n
        self.load = load_
        self.buffer = buffer.Memory(self.env, self.cont, step_size=e_trace, done_penalty=d_p, skill_num=self.skill_num)
        self.data = dataset.SimData(capacity=self.capacity)
        self.dataloader = dataloader.CustomDataLoader(self.data, batch_size=self.b_s)

        self.writer = SummaryWriter('Result/' + env_n + self.cont.name())

        self.PARAM_PATH = 'Parameter/' + env_n
        print("parameter path is " + self.PARAM_PATH)

        self.PARAM_PATH_TEST = 'Parameter/' + env_n + '_test'
        print("tmp parameter path is " + self.PARAM_PATH_TEST)

    def pre_train(self):
        if self.load == 1:
            self.cont.load_model(self.PARAM_PATH)
        i = 0
        while i < self.t_i:
            print(i)
            i = i + 1
            self.buffer.simulate(self.capacity, self.data, self.dataloader)
            loss = self.cont.update(self.m_i, next(iter(self.dataloader)))
            print("loss = ", loss)
            j = 0
            while j < len(loss):
                self.writer.add_scalar("loss " + str(j), loss[j], i)
                j = j + 1
            self.writer.add_scalar("performance", self.buffer.get_performance(), i)
            model = self.cont.save_model(self.PARAM_PATH)

        i = 0
        self.env.close()
        self.writer.flush()
        self.writer.close()

    def train(self):
        i = 0
        pre_performance = 0
        maxp_index = 0
        self.cont.load_model(self.PARAM_PATH)
        while i < self.skill_num:
            performance = self.buffer.simulate(self.capacity, self.data, self.dataloader, index=i, pretrain=0)
            print(performance)
            if performance > pre_performance:
                maxp_index = i
                pre_performance = performance
            i = i + 1
        print("max = ", maxp_index)
        print("select complete")

        model = None
        i = 0
        while i < self.t_i:
            print(i)
            i = i + 1
            self.buffer.simulate(self.capacity, self.data, self.dataloader, index=maxp_index, pretrain=0)
            loss = self.policy.update(self.m_i, next(iter(self.dataloader)))
            print("loss = ", loss)
            j = 0
            while j < len(loss):
                self.writer.add_scalar("loss " + str(j), loss[j], i)
                j = j + 1
            self.writer.add_scalar("performance", self.buffer.get_performance(), i)
            model = self.cont.save_model(self.PARAM_PATH)

        i = 0

        while i < len(model):

            for param in model[i].parameters():
                print("-----------" + str(i) + "-------------")
                print(param)
            i = i + 1

        self.env.close()
        self.writer.flush()
        self.writer.close()

        return self.policy, maxp_index

