import numpy as np
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class IndexAct:
    """
    torch index
    -> numpy action
    numpy action
    -> torch index
    """
    def __init__(self, env_name, a_l, precision, b_s):
        self.env_name = env_name
        self.a_l = a_l
        self.precision = precision
        self.gauge = 2.0/(self.precision - 1.0)
        self.batch = b_s

    def index2act(self, _input, per_one=1):
        if self.env_name == "hope":
            batch = 1
            if per_one == 0:
                batch = self.batch
            # only used in predict
            i = 0
            out = torch.zeros((batch, self.a_l), device=DEVICE)
            while i < batch:
                precision = torch.tensor(self.precision).to(DEVICE)
                if batch == 1:
                    div_1 = torch.div(_input, precision, rounding_mode='floor')
                else:
                    div_1 = torch.div(_input[i], precision, rounding_mode='floor')
                div_2 = torch.div(div_1, precision, rounding_mode='floor')
                if batch == 1:
                    a_1 = _input % precision
                else:
                    a_1 = _input[i] % precision
                a_2 = div_1 % precision
                a_3 = div_2 % precision
                out[i] = torch.tensor([a_1, a_2, a_3], device=DEVICE)*self.gauge - 1
                i = i + 1
            out = out.squeeze()
            return out.cpu().numpy()
        elif self.env_name == "cart":
            return _input.cpu().numpy()
        else:
            print("converter error")

    def act2index(self, _input):
        if self.env_name == "hope":
            i = 0
            out = np.zeros(self.batch)
            while i < self.batch:
                new_input = (_input[i]+1)/self.gauge
                out[i] = new_input[2] * self.precision**2 + new_input[1] * self.precision + new_input[0]
                i = i + 1
            out = out.squeeze()
            return torch.from_numpy(out).to(DEVICE).type(torch.int64)
        elif self.env_name == "cart":
            return torch.from_numpy(_input).to(DEVICE).type(torch.int64)
        else:
            print("converter error")

    def rand_act(self):
        if self.env_name == "hope":
            return (np.random.randint(self.precision, size=(self.a_l,))*self.gauge) - 1
        elif self.env_name == "cart":
            _a = np.random.randint(self.a_l, size=(1,))
            return _a.squeeze()
        else:
            print("converter error")


class NAFReward:
    def __init__(self, sl, al, re):
        self.sl = sl
        self.al = al
        self.re = re

    def sa_reward(self, state_action):
        state, action = torch.split(state_action, [self.sl, self.al], dim=-1)
        pre_psd, bias, value = torch.split(self.re(state), [self.al ** 2, self.al, 1], dim=-1)

        pre_psd = torch.reshape(pre_psd, (-1, self.al, self.al)).squeeze()
        pre_psd_trans = torch.transpose(pre_psd, -2, -1)
        psd = torch.matmul(pre_psd, pre_psd_trans)
        a_b = (action - bias).unsqueeze(-2)
        a_b_t = torch.transpose(a_b, -2, -1)

        last_val = torch.matmul(torch.matmul(a_b, psd), a_b_t).squeeze() - value.squeeze()

        return last_val

    def prob(self, state_action):
        state, action = torch.split(state_action, [self.sl, self.al], dim=-1)
        pre_psd, bias, value = torch.split(self.re(state), [self.al ** 2, self.al, 1], dim=-1)

        pre_psd = torch.reshape(pre_psd, (-1, self.al, self.al)).squeeze()
        pre_psd_trans = torch.transpose(pre_psd, -2, -1)
        psd = torch.matmul(pre_psd, pre_psd_trans)
        # psd = psd no exception occered
        return bias, psd


class NAFPolicy:
    def __init__(self, sl, al, policy):
        self.sl = sl
        self.al = al
        self.policy = policy

    def prob(self, state):
        pre_psd, mean = torch.split(self.policy(state), [self.al**2, self.al], dim=-1)
        pre_psd = torch.reshape(pre_psd, (-1, self.al, self.al)).squeeze()
        pre_psd_trans = torch.transpose(pre_psd, -2, -1)
        psd = torch.matmul(pre_psd, pre_psd_trans)
        # psd = cov matrix
        # MultivariateNormal(mean, psd)
        # psd = psd no exception occered
        return mean, psd


class StateConvert:
    def __init__(self, sl, skill):
        self.sl = sl
        self.skill = skill

    def convert(self, state, index):
        tmp_n_p_o = np.zeros(len(state) * self.skill)
        tmp_n_p_o[index * len(state):(index + 1) * len(state)] = state
        n_p_o = tmp_n_p_o
        t_p_o = torch.from_numpy(n_p_o).type(torch.float32).to(DEVICE)
        return t_p_o
