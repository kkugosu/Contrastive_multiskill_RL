from policy import gps, AC, DDPG, PG, PPO, SAC, TRPO
from utils import render, train
from control import diayn
import gym
from utils import converter

if __name__ == "__main__":

    BATCH_SIZE = 10000
    CAPACITY = 10000
    TRAIN_ITER = 100
    MEMORY_ITER = 100
    HIDDEN_SIZE = 32
    learning_rate = 0.01
    policy = None

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

    env_name = None
    control = None
    e_trace = 1
    precision = 5
    env = None
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

    if env_name == "cart":
        a_l = 2
        a_index_l = 2
    else:
        a_l = len(env.action_space.sample())
        a_index_l = precision ** a_l

    model_type = None

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
    num_skill = get_integer()

    arg_list = [BATCH_SIZE, CAPACITY, HIDDEN_SIZE, learning_rate, skill_n,
                TRAIN_ITER, MEMORY_ITER, control, env_name, e_trace, precision, done_penalty]
    print(arg_list)

    policy = None

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

    s_l = len(env.observation_space.sample())
    my_converter = converter.IndexAct(env_name, a_l, precision, BATCH_SIZE)
    control_n = None
    cont = None
    valid = 0
    while valid == 0:
        print("enter RL control, {diayn}")
        control_n = input("->")
        if control_n == "diayn":
            cont = diayn.DIAYN(l_r, s_l, policy)
        else:
            print("control name error")

    my_train = train.Train(learning_rate, policy, TRAIN_ITER, MEMORY_ITER, BATCH_SIZE,
                          control_n, cont, done_penalty, CAPACITY,  env_name, env, e_trace)

    policy = my_train.train()
    selected_policy = initialize(policy, done_penalty, CAPACITY, env_name, env)
    optimal_policy = selected_policy.train()

    my_rend = render.Render(optimal_policy, env)
    my_rend.rend()
