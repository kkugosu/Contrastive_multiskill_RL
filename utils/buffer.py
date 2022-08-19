import torch
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
GAMMA = 0.98


class Memory:

    def __init__(self, env, control, step_size, done_penalty, skill_num):
        self.env = env
        self.step_size = step_size
        self.done_penalty = done_penalty
        self.performance = 0
        self.control = control
        self.index = None
        self.skill_num = skill_num

    def simulate(self, capacity, dataset, dataloader, index=None):
        total_num = 0
        pause = 0
        total_performance = 0
        failure = 0
        if index is not None:
            print("simulate after training")
            self.index = index
        else:
            print("simulate for training")
            self.index = np.random.randint(self.skill_num)
        while total_num < capacity - pause:
            n_p_o = self.env.reset()
            t = 0
            policy = self.control.policy
            print("episode start")
            while t < capacity - total_num: #if pg, gain accumulate
                tmp_n_p_o = np.zeros(len(n_p_o)*self.skill_num)
                print(tmp_n_p_o)
                print(np.shape(tmp_n_p_o))
                print(n_p_o)
                print(np.shape(n_p_o))
                tmp_n_p_o[index*len(n_p_o):(index+1)*len(n_p_o)] = n_p_o
                n_p_o = tmp_n_p_o
                t_p_o = torch.from_numpy(n_p_o).to(device)
                n_a = policy.action(t_p_o)
                n_o, n_r, n_d, n_i = self.env.step(n_a)
                t_r = self.control.reward(n_o)
                n_r = torch.from_numpy(t_r)
                dataset.push(n_p_o, n_a, n_o, n_r, np.float32(n_d))
                n_p_o = n_o
                t = t + 1
                total_performance = total_performance + n_r
                if n_d:
                    total_num += t
                    t = 0
                    failure = failure + 1
                    break
            pause = t
        self.performance = total_performance #/failure
        self._reward_converter(dataset, dataloader)
        return total_performance

    def get_performance(self):
        return self.performance

    def _reward_converter(self, dataset, dataloader):
        t = 0
        pre_observation, action, observation, reward, done = next(iter(dataloader))
        # cal per trajectary to_end length ex) 4 3 2 1 6 5 4 3 2 1
        # set step to upper bound ex) step = 5 ->  4 3 2 1 5 5 4 3 2 1
        global_index = len(done) - 1
        local_index = 0
        while 0 <= global_index:
            if done[global_index] == 1:
                local_index = 1
                done[global_index] = local_index
                reward[global_index] -= self.done_penalty
            else:
                local_index = local_index + 1
                if local_index > self.step_size:
                    local_index = self.step_size
                done[global_index] = local_index
            global_index = global_index - 1
        # cal newreward per state-action pair

        global_index = 0
        while global_index < len(done):
            observation[global_index] = observation[int(global_index + done[global_index] - 1)]
            # change observation to last step indexed observation state
            local_index = 1
            while local_index < done[global_index]:
                tmp = reward[global_index + local_index] * GAMMA ** local_index
                reward[global_index] += tmp
                local_index = local_index + 1
            global_index += 1
        global_index = 0
        while global_index < len(done):
            dataset.push(pre_observation[global_index], action[global_index], observation[global_index],
                         reward[global_index], np.float32(done[global_index]))
            global_index += 1
        return dataset

