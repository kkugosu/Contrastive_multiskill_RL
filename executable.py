from policy import gps, AC, DDPG, PG, PPO, SAC, TRPO
from utils import render, train
from control import diayn, dads, vic, valor, visr, edl, smm, aps, cic
import gym
from utils import converter
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":

    BATCH_SIZE = 10000
    CAPACITY = 10000
    TRAIN_ITER = 100
    MEMORY_ITER = 100
    learning_rate = 0.01
    policy = None
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
            print("enter hopper precision 3")
            precision = 3
            # precision = get_integer()
        else:
            print("error")

    STATE_LENGTH = len(env.observation_space.sample())

    if env_name == "cart":
        ACTION_LENGTH = 2
        A_index_L = 2
    else:
        ACTION_LENGTH = len(env.action_space.sample())
        A_index_L = precision ** ACTION_LENGTH
    """
    valid = 0
    while valid == 0:
        print("model_free : 0, model_based : 1, meta : 2")
        print(" meta ")
        model_type = get_integer()
        if (model_type >= 0) | (model_type < 3):
            valid = 1
            
    """

    # print("enter batchsize recommend 1000")
    # BATCH_SIZE = get_integer()

    print("enter memory capacity recommend 1000")
    print("batchsize = capacity")
    print("capacity = 1000")
    CAPACITY = 1000 # get_integer()
    BATCH_SIZE = CAPACITY

    print("memory reset time recommend 100")
    print("train iter = 100")
    TRAIN_ITER = 100

    print("train_iteration per memory recommend 10")
    print("memory iter = 10")
    MEMORY_ITER = 10

    print("enter learning rate recommend 0.01")
    print("learning rate = 0.0001")
    learning_rate = 0.001

    print("enter eligibility trace step, if pg: 100, if gps: 1")
    print("e_trace = 1")
    e_trace = 1

    print("done penalty, if cartpole, recommend 10")
    print("done penalty = 1")
    done_penalty = 1

    print("load previous model 0 or 1")
    load_ = input("->")

    print("num_skills?")
    print("skillnum = 10")
    skill_num = 10

    my_converter = converter.IndexAct(env_name, ACTION_LENGTH, precision, BATCH_SIZE)

    arg_list = [learning_rate, skill_num, TRAIN_ITER, MEMORY_ITER, STATE_LENGTH, ACTION_LENGTH,
                A_index_L, my_converter, DEVICE]
    model_type = 1

    if model_type == 1:
        valid = 0
        while valid == 0:
            print("enter RL policy, {PG, AC, TRPO, PPO, DDPG, SAC}")
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
    elif model_type == 2:
        valid = 0
        while valid == 0:
            print("enter RL policy, {gps}")
            control = input("->")
            if control == "gps":
                policy = gps.GPS(*arg_list)
                valid = 1
            else:
                print("error")
    else:
        print("error")
        
    valid = 0
    while valid == 0:
        print("enter RL control, {diayn}")
        control_name = input("->")
        if control_name == "diayn":
            control = diayn.DIAYN(learning_rate, STATE_LENGTH, ACTION_LENGTH, policy, skill_num, DEVICE)
            valid = 1
        elif control_name == "dads":
            control = dads.DADS(learning_rate, STATE_LENGTH, ACTION_LENGTH, policy, skill_num, DEVICE)
            valid = 1
        elif control_name == "vic":
            control = vic.VIC(learning_rate, STATE_LENGTH, ACTION_LENGTH, policy, skill_num, DEVICE)
            valid = 1
        elif control_name == "valor":
            control = smm.SMM(learning_rate, STATE_LENGTH, ACTION_LENGTH, policy, skill_num, DEVICE)
            valid = 1
        elif control_name == "visr":
            control = edl.EDL(learning_rate, STATE_LENGTH, ACTION_LENGTH, policy, skill_num, DEVICE)
            valid = 1
        elif control_name == "edl":
            control = valor.VALOR(learning_rate, STATE_LENGTH, ACTION_LENGTH, policy, skill_num, DEVICE)
            valid = 1
        elif control_name == "smm":
            control = visr.VISR(learning_rate, STATE_LENGTH, ACTION_LENGTH, policy, skill_num, DEVICE)
            valid = 1
        elif control_name == "aps":
            control = aps.APS(learning_rate, STATE_LENGTH, ACTION_LENGTH, policy, skill_num, DEVICE)
            valid = 1
        elif control_name == "cic":
            control = cic.CIC(learning_rate, STATE_LENGTH, ACTION_LENGTH, policy, skill_num, DEVICE)
            valid = 1
        else:
            print("control name error")

    my_train = train.Train(TRAIN_ITER, MEMORY_ITER, BATCH_SIZE, skill_num, policy,
                           CAPACITY, env, control, env_name, e_trace, done_penalty, load_)
    print("pre train")
    policy = my_train.pre_train()
    print("train")
    optimal_policy, index = my_train.train()
    print("rendering")
    my_rend = render.Render(optimal_policy, index, skill_num, env)
    my_rend.rend()
