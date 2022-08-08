import torch
import random
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Policy:
    def __init__(self, policy, model, converter):
        self.policy = policy
        self.model = model
        self.converter = converter
        self.softmax = nn.Softmax(dim=-1)

    def select_action(self, n_p_o):
        t_p_o = torch.tensor(n_p_o, device=device, dtype=torch.float32)

        if self.policy == "gps":
            if random.random() < 1.1:
                with torch.no_grad():
                    t_a = self.model.get_global_action(t_p_o)
                n_a = t_a.cpu().numpy()
                return n_a
            else:
                with torch.no_grad():
                    t_a = self.model.get_local_action(t_p_o)
                n_a = t_a.cpu().numpy()
                return n_a
        else:
            print("model name error")
            return None
