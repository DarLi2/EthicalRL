from src.environments import bridge
from src.environments import bridge_world_drowningPeople
from src.algorithms import q_learning_agent_table

import gymnasium as gym
from gymnasium import envs
from tqdm import tqdm
import json
import os
from typing import Dict
import pickle

# Get the current working directory
current_directory = os.getcwd()

# Specify the relative path to the parent folder
relative_folder_path = '..'

# Create the full path to the pickle file using the relative path and the current working directory
file_path = os.path.join(current_directory, "src/models/train_setup/agent_env_selection.json")
agents_file_path = os.path.join(current_directory, relative_folder_path, 'agents.pkl')

with open(file_path, "r") as read_file:
        config = json.load(read_file)

selected_env = config.get("env")
env = gym.make(selected_env, render_mode="human")

def test_rescue_action(agent_name):
        with open(agents_file_path, "rb") as file:
                agents = pickle.load(file)
        agent = agents[agent_name]
        agent.epsilon = 0
        state, info = env.reset()
        done = False
        action = 4
        next_state, reward, terminated, truncated, info = env.step(action)
        print("Reward:", reward)

        # update the agent
        agent.update(state, action, reward, terminated, next_state)

        # update if the environment is done and the current state
        done = terminated
        state = next_state

test_rescue_action("q_learning_agent")