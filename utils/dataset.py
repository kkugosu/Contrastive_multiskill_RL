from collections import namedtuple, deque
from torch.utils.data import Dataset
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class SimData(Dataset):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def __getitem__(self, idx):
        return self.memory[idx]

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        self.memory.append(Transition(*args))
