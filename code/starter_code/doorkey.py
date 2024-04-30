from utils import *
# from example import example_use_of_gym_env
import os 

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


def doorkey_problem(env):
    
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Feel Free to modify this fuction
    """
    optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    return optim_act_seq


def partA():
    
    # env_path = "./envs/known_envs/example-8x8.env"
    # env, info = load_env(env_path)  # load an environment
    dirpath = "./envs/known_envs"
    file_list = os.listdir(dirpath)
    file_list = [file for file in file_list if file.endswith('env')]
    file_dict = {}
    for file in file_list:
        env_path = os.path.join(dirpath, file)
        env, info = load_env(env_path)  # load an environment
        seq = doorkey_problem(env)
        file_dict[file] = seq
        draw_gif_from_seq(seq, load_env(env_path)[0])  # draw a GIF & save
        break
    return file_dict

    # # seq = doorkey_problem(env)  # find the optimal action sequence
    # draw_gif_from_seq(seq, load_env(env_path)[0])  # draw a GIF & save
    # return seq

def partB():
    env_folder = "./envs/random_envs"
    env, info, env_path = load_random_env(env_folder)


if __name__ == "__main__":
    # example_use_of_gym_env()
    partA()
    # partB()

