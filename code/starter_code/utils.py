import os
import numpy as np
import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
import imageio
import random
from minigrid.core.world_object import Goal, Key, Door

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


def step_cost(action, state):

    # You should implement the stage cost by yourself
    # Feel free to use it or not
    # ************************************************
    print(f"state_ terminated = {state['terminated']}")
    return (1-state['terminated'])  # the cost of action

def step_gif(env, action):
    actions = {
        0: env.unwrapped.actions.forward,
        1: env.unwrapped.actions.left,
        2: env.unwrapped.actions.right,
        3: env.unwrapped.actions.pickup,
        4: env.unwrapped.actions.toggle,
    }
    env.step(actions[action])
    return

def step(env, action, state, goal):
    """
    Take Action
    ----------------------------------
    actions:
        0 # Move forward (MF)
        1 # Turn left (TL)
        2 # Turn right (TR)
        3 # Pickup the key (PK)
        4 # Unlock the door (UD)
    """
    actions = {
        0: env.unwrapped.actions.forward,
        1: env.unwrapped.actions.left,
        2: env.unwrapped.actions.right,
        3: env.unwrapped.actions.pickup,
        4: env.unwrapped.actions.toggle,
    }
    cost = 0

    if np.all(state[0] == goal):
        return 0
    elif (env.unwrapped.front_pos[0] not in list(range(8) )or env.unwrapped.front_pos[1] not in list(range(8))):
            return np.inf
    else:
        env.step(actions[action])
        cost = 1

    return cost


def generate_random_env(seed, task):
    """
    Generate a random environment for testing
    -----------------------------------------
    seed:
        A Positive Integer,
        the same seed always produces the same environment
    task:
        'MiniGrid-DoorKey-5x5-v0'
        'MiniGrid-DoorKey-6x6-v0'
        'MiniGrid-DoorKey-8x8-v0'
    """
    if seed < 0:
        seed = np.random.randint(50)
    env = gym.make(task, render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def load_env(path):
    """
    Load Environments
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    with open(path, "rb") as f:
        env = pickle.load(f)

    info = {
        "height": env.unwrapped.height, 
        "width": env.unwrapped.width, 
        "init_agent_pos": env.unwrapped.agent_pos, 
        "init_agent_dir": env.unwrapped.agent_dir,
    }

    for i in range(env.unwrapped.height):
        for j in range(env.unwrapped.width):
            if isinstance(env.unwrapped.grid.get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.unwrapped.grid.get(j, i), Door):
                info["door_pos"] = np.array([j, i])
            elif isinstance(env.unwrapped.grid.get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])

    return env, info


def load_random_env(env_path):
    """
    Load a random DoorKey environment
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    # env_list = [os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder)]
    # env_path = random.choice(env_list)
    with open(env_path, "rb") as f:
        env = pickle.load(f)

    info = {
        "height": env.unwrapped.height,
        "width": env.unwrapped.width,
        "init_agent_pos": env.unwrapped.agent_pos,
        "init_agent_dir": env.unwrapped.dir_vec,
        "door_pos": [],
        "door_open": [],
    }

    for i in range(env.unwrapped.height):
        for j in range(env.unwrapped.width):
            if isinstance(env.grid.get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.grid.get(j, i), Door):
                info["door_pos"].append(np.array([j, i]))
                if env.grid.get(j, i).is_open:
                    info["door_open"].append(True)
                else:
                    info["door_open"].append(False)
            elif isinstance(env.grid.get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])

    return env, info, env_path


def save_env(env, path):
    with open(path, "wb") as f:
        pickle.dump(env, f)


def plot_env(env):
    """
    Plot current environment
    ----------------------------------
    """
    img = env.render()
    plt.figure()
    plt.imshow(img)
    plt.show()


def draw_gif_from_seq(seq, env,  path="./gif/doorkey.gif"):
    """
    Save gif with a given action sequence
    ----------------------------------------
    seq:
        Action sequence, e.g [0,0,0,0] or [MF, MF, MF, MF]

    env:
        The doorkey environment
    """
    with imageio.get_writer(path, mode="I", duration=0.8) as writer:
        img = env.render()
        writer.append_data(img)
        # step_gif(env, act)
        for act in seq:
            img = env.render()
            writer.append_data(img)
            step_gif(env, act)
        img = env.render()
        writer.append_data(img)
    print(f"GIF is written to {path}")
    return
    
    
