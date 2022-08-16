import torch
from utils import buffer, dataset, dataloader
from torch.utils.tensorboard import SummaryWriter
from control import diayn


class Train:
    def __init__(self, learning_rate, policy, train_iter, memory_iter, batch_size,
                 control_n, cont, d_p, capacity,  env_n, env, e_trace):

        self.t_i = train_iter
        self.m_i = memory_iter
        self.ca = capacity
        self.b_s = batch_size
        self.l_r = learning_rate
        self.policy = policy
        self.env = env
        self.cont = cont
        self.buffer = buffer.Memory(self.env, self.cont, step_size=e_trace, done_penalty=d_p)
        self.data = dataset.SimData(capacity=self.ca)
        self.dataloader = dataloader.CustomDataLoader(self.data, batch_size=self.b_s)

        self.writer = SummaryWriter('Result/' + env_n + '/' + control_n)

        self.PARAM_PATH = 'Parameter/' + env_n + control_n
        print("parameter path is " + self.PARAM_PATH)

        self.PARAM_PATH_TEST = 'Parameter/' + env_n + control_n + '_test'
        print("tmp parameter path is " + self.PARAM_PATH_TEST)

    def pre_train(self):

        self.cont.load_model(self.PARAM_PATH)
        self.policy.load_model(self.PARAM_PATH)
        c_model = None
        p_model = None
        i = 0
        while i < self.t_i:
            print(i)
            i = i + 1
            self.buffer.simulate(self.ca, self.data, self.dataloader)
            loss = self.cont.update(next(iter(self.dataloader)))
            print("loss = ", loss)
            j = 0
            while j < len(loss):
                self.writer.add_scalar("loss " + str(j), loss, i)
                j = j + 1
            self.writer.add_scalar("performance", self.buffer.get_performance(), i)
            c_model = self.cont.save_model(self.PARAM_PATH)
            p_model = self.policy.save_model(self.PARAM_PATH)

        i = 0
        while i < len(c_model):
            for param in c_model[i].parameters():
                print("-----------" + str(i) + "-------------")
                print(param)
            i = i + 1
        i = 0
        while i < len(p_model):
            for param in p_model[i].parameters():
                print("-----------" + str(i) + "-------------")
                print(param)
            i = i + 1

        self.env.close()
        self.writer.flush()
        self.writer.close()

    def train(self):
        i = 0
        pre_performance = 0
        maxp_index = 0
        while i < skill_num:
            j = 0
            performance = 0
            while j < 10:
                j = j + 1
                performance = + self.buffer.simulate(self.ca, self.data, self.dataloader,i)
            i = i + 1
            if performance > pre_performance:
                maxp_index = i
                pre_performance = performance

        self.cont.load_model(self.PARAM_PATH)
        self.policy.load_model(self.PARAM_PATH)
        c_model = None
        p_model = None
        i = 0
        while i < self.t_i:
            print(i)
            i = i + 1
            self.buffer.simulate(self.ca, self.data, self.dataloader,maxp_index)
            loss = self.policy.update(next(iter(self.dataloader)))
            print("loss = ", loss)
            j = 0
            while j < len(loss):
                self.writer.add_scalar("loss " + str(j), loss, i)
                j = j + 1
            self.writer.add_scalar("performance", self.buffer.get_performance(), i)
            c_model = self.cont.save_model(self.PARAM_PATH)
            p_model = self.policy.save_model(self.PARAM_PATH)

        i = 0
        while i < len(c_model):
            for param in c_model[i].parameters():
                print("-----------" + str(i) + "-------------")
                print(param)
            i = i + 1
        i = 0
        while i < len(p_model):
            for param in p_model[i].parameters():
                print("-----------" + str(i) + "-------------")
                print(param)
            i = i + 1

        self.env.close()
        self.writer.flush()
        self.writer.close()
