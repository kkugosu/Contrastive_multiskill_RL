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

        self.buffer = buffer.Simulate(self.env, self.cont, step_size=e_trace, done_penalty=d_p)
        self.data = dataset.SimData(capacity=self.ca)
        self.dataloader = dataloader.CustomDataLoader(self.data, batch_size=self.b_s)

        self.writer = SummaryWriter('Result/' + env_n + '/' + control_n)

        self.PARAM_PATH = 'Parameter/' + env_n + control_n
        print("parameter path is " + self.PARAM_PATH)

        self.PARAM_PATH_TEST = 'Parameter/' + env_n + control_n + '_test'
        print("tmp parameter path is " + self.PARAM_PATH_TEST)

    def loading(self):
        print("loading")
        i = 0
        while i < len(self.model):
            self.model[i].load_state_dict(torch.load(self.PARAM_PATH + '/' + str(i) + '.pth'))
            i = i + 1
        print("loading complete")

    def saving(self):
        i = 0
        while i < len(self.model):
            torch.save(self.model[i].state_dict(), self.PARAM_PATH + '/' + str(i) + '.pth')
            i = i + 1

    def train(self):
        self.model = self.policy.getmodel
        self.model = self.cont.getmodel
        self.loading()
        i = 0
        while i < self.t_i:
            print(i)
            i = i + 1
            self.buffer.renewal_memory(self.ca, self.data, self.dataloader)
            loss = self.cont.update(next(iter(self.dataloader)))
            print("loss = ", loss)
            j = 0
            while j < len(loss):
                self.writer.add_scalar("loss " + str(j), loss, i)
                j = j + 1
            self.writer.add_scalar("performance", self.buffer.get_performance(), i)
            self.saving()

        i = 0
        while i < len(self.model):
            for param in self.model[i].parameters():
                print("-----------" + str(i) + "-------------")
                print(param)
            i = i + 1

        self.env.close()
        self.writer.flush()
        self.writer.close()
