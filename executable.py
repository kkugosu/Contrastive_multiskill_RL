from policy import gps, AC, DDPG, PG, PPO, SAC, TRPO
from utils import render
from control import diayn

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

    valid = 0
    while valid == 0:
        print("enter envname, {cartpole as cart, hoppper as hope}")
        env_name = input("->")
        if env_name == "cart":
            valid = 1
            print("we can't use DDPG")
        elif env_name == "hope":
            valid = 1
            print("enter hopper precision 3 or 5")
            precision = get_integer()
        else:
            print("error")
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

    print("load previous model 0 or 1")
    load_ = input("->")

    arg_list = [BATCH_SIZE, CAPACITY, HIDDEN_SIZE, learning_rate,
                TRAIN_ITER, MEMORY_ITER, control, env_name, e_trace, precision, done_penalty]
    print(arg_list)

    policy = None

    if model_type == 0:
        valid = 0
        while valid == 0:
            print("enter RL policy, {PG, DQN, AC, TRPO, PPO, DDPG, SAC}")
            control = input("->")
            if control == "PG":
                policy = PG.PGPolicy(*arg_list)
                valid = 1
            elif control == "AC":
                policy = AC.ACPolicy(*arg_list)
                valid = 1
            elif control == "TRPO":
                policy = TRPO.TRPOPolicy(*arg_list)
                valid = 1
            elif control == "PPO":
                policy = PPO.PPOPolicy(*arg_list)
                valid = 1
            elif control == "DDPG":
                policy = DDPG.DDPGPolicy(*arg_list)
                valid = 1
            elif control == "SAC":
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
    print("num_skills?")
    num_skill = get_integer()

    policy.training(load=load_)
    _policy = policy.get_policy()

    my_rend = render.Render(_policy, *arg_list)
    my_rend.rend()
