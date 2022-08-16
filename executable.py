from policy import gps, AC, DDPG, PG, PPO, SAC, TRPO
from utils import render, train
from control import diayn
import gym
from utils import converter
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":

    BATCH_SIZE = 10000
    CAPACITY = 10000
    TRAIN_ITER = 100
    MEMORY_ITER = 100
    HIDDEN_SIZE = 32
    learning_rate = 0.01
    policy = None
    policy_name = None
    env = None
    env_name = None
    control = None
    control_name = None
    e_trace = 1
    precision = 3
    model_type = None

    def get_integer():
        _valid = 0
        while _valid == 0:
            integer = input("->")
            try:
                int(integer)
                if float(integer).is_integer():
                    _valid = 1
                    return int(integer)
                else:
                    print("enter integer")
            except ValueError:
                print("enter integer")

    def get_float():
        _valid = 0
        while _valid == 0:
            float_ = input("->")
            try:
                float(float_)
                _valid = 1
                return float(float_)
            except ValueError:
                print("enter float")

    valid = 0
    while valid == 0:
        print("enter envname, {cartpole as cart, hoppper as hope}")
        env_name = input("->")
        if env_name == "cart":
            env = gym.make('CartPole-v1')
            valid = 1
            print("we can't use DDPG")
        elif env_name == "hope":
            env = gym.make('Hopper-v3')
            valid = 1
            print("enter hopper precision 3 or 5")
            precision = get_integer()
        else:
            print("error")

    STATE_LENGTH = len(env.observation_space.sample())

    if env_name == "cart":
        ACTION_LENGTH = 2
        A_index_L = 2
    else:
        ACTION_LENGTH = len(env.action_space.sample())
        A_index_L = precision ** ACTION_LENGTH

    valid = 0
    while valid == 0:
        print("model_free : 0, model_based : 1, meta : 2")
        model_type = get_integer()
        if (model_type >= 0) | (model_type < 3):
            valid = 1

    print("enter HIDDEN_SIZE recommend 32")
    HIDDEN_SIZE = get_integer()

    print("enter batchsize recommend 1000")
    BATCH_SIZE = get_integer()

    print("enter memory capacity recommend 1000")
    CAPACITY = get_integer()

    print("memory reset time recommend 100")
    TRAIN_ITER = get_integer()

    print("train_iteration per memory recommend 10")
    MEMORY_ITER = get_integer()

    print("enter learning rate recommend 0.01")
    learning_rate = get_float()

    print("enter eligibility trace step, if pg: 100, if gps: 1")
    e_trace = get_integer()

    print("done penalty, if cartpole, recommend 10")
    done_penalty = get_integer()

    print("done penalty, if cartpole, recommend 10")
    skill_n = get_integer()

    print("load previous model 0 or 1")
    load_ = input("->")

    print("num_skills?")
    skill_num = get_integer()

    my_converter = converter.IndexAct(env_name, ACTION_LENGTH, precision, BATCH_SIZE)

    arg_list = [learning_rate, skill_num, TRAIN_ITER, MEMORY_ITER, STATE_LENGTH, ACTION_LENGTH,
                A_index_L, my_converter, DEVICE]

    if model_type == 0:
        valid = 0
        while valid == 0:
            print("enter RL policy, {PG, DQN, AC, TRPO, PPO, DDPG, SAC}")
            policy_n = input("->")
            if policy_n == "PG":
                policy = PG.PGPolicy(*arg_list)
                valid = 1
            elif policy_n == "AC":
                policy = AC.ACPolicy(*arg_list)
                valid = 1
            elif policy_n == "TRPO":
                policy = TRPO.TRPOPolicy(*arg_list)
                valid = 1
            elif policy_n == "PPO":
                policy = PPO.PPOPolicy(*arg_list)
                valid = 1
            elif policy_n == "DDPG":
                policy = DDPG.DDPGPolicy(*arg_list)
                valid = 1
            elif policy_n == "SAC":
                policy = SAC.SACPolicy(*arg_list)
                valid = 1
            else:
                print("error")
    else:
        valid = 0
        while valid == 0:
            print("enter RL policy, {gps}")
            control = input("->")
            if control == "gps":
                policy = gps.GPS(*arg_list)
                valid = 1
            else:
                print("error")

    valid = 0
    while valid == 0:
        print("enter RL control, {diayn}")
        control_name = input("->")
        if control_name == "diayn":
            cont = diayn.DIAYN(learning_rate, STATE_LENGTH, policy, skill_num, DEVICE)
        else:
            print("control name error")

    my_train = train.Train(TRAIN_ITER, MEMORY_ITER, BATCH_SIZE, skill_num, control_name,
                           CAPACITY, env, control, env_name, e_trace, done_penalty)

    policy = my_train.pre_train()
    optimal_policy = my_train.train()

    my_rend = render.Render(optimal_policy, env)
    my_rend.rend()
