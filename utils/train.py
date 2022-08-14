import torch
from utils import buffer, dataset, dataloader
from torch.utils.tensorboard import SummaryWriter
from utils import converter
import gym
from control import diayn


class Train:
    def __init__(self,  train_iter, memory_iter, batch_size,
                 control_n, d_p, capacity,  env_n, precision, e_trace):

        self.t_i = train_iter
        self.m_i = memory_iter
        self.ca = capacity
        self.b_s = batch_size
        self.cont = None
        if env_n == "cart":
            self.env = gym.make('CartPole-v1')
        elif env_n == "hope":
            self.env = gym.make('Hopper-v3')
        else:
            self.env = gym.make('Hopper-v3')

        self.s_l = len(self.env.observation_space.sample())
        if env_n == "cart":
            self.a_l = 2
            self.a_index_l = 2
        else:
            self.a_l = len(self.env.action_space.sample())
            self.a_index_l = precision ** self.a_l

        self.buffer = buffer.Simulate(self.env, control, step_size=e_trace, done_penalty=d_p)
        self.data = dataset.SimData(capacity=self.ca)
        self.dataloader = dataloader.CustomDataLoader(self.data, batch_size=self.b_s)
        self.converter = converter.IndexAct(env_n, self.a_l, precision)
        self.writer = SummaryWriter('Result/' + env_n + '/' + self.cont)
        if control_n == "diayn":
            self.cont = diayn.DIAYN()
        else:
            print("control name error")
        print("state_space = ", self.env.observation_space)
        print("STATE_SIZE(input) = ", self.s_l)
        print("action_space = ", self.env.action_space)
        print("ACTION_SIZE(output) = ", self.a_l)
        print("ACTION_INDEX_SIZE(output) = ", self.a_index_l)

        self.PARAM_PATH = 'Parameter/' + env_n + self.cont
        print("parameter path is " + self.PARAM_PATH)

        self.PARAM_PATH_TEST = 'Parameter/' + env_n + self.cont + '_test'
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
        self.model = policy.getmodel
        self.model = diayn.getmodel
        self.loading()
        i = 0
        while i < self.t_i:
            print(i)
            i = i + 1
            self.buffer.renewal_memory(self.ca, self.data, self.dataloader)
            loss = self.diayn.update()
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
