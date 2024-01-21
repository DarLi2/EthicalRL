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


current_directory = os.getcwd()
    # Create the full path to the pickle file using the relative path and the current working directory
file_path = os.path.join(current_directory, "src/models/train_setup/agent_env_selection.json")

with open(file_path, "r") as read_file:
        selection = json.load(read_file)

#set up the environment
env = selection.get("env")
print(gym.envs.registry.keys())
env = gym.make(env, render_mode=None)

# set up agent with chosen hyperparamters (can be changed in the json file)
selected_agent = selection.get("selected_agent")

if selected_agent == "q_learning_agent_table":
    file_path = os.path.join(current_directory, "src/models/train_setup/hyperparameters/q_learning.json")
    with open(file_path, "r") as read_file:
        hyperparameters = json.load(read_file)
    n_episodes = hyperparameters.get("n_episodes", 100000)
    learning_rate = hyperparameters.get("learning_rate", 0.01)
    start_epsilon = hyperparameters.get("start_epsilon", 1.0)
    epsilon_decay = hyperparameters.get("epsilon_decay", start_epsilon / (n_episodes / 2))
    final_epsilon = hyperparameters.get("final_epsilon", 0.1)
    agent = q_learning_agent_table.QLearningAgent(
    env = env,
    q_values=None,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon
    )

def train():
    if selected_agent == "q_learning_agent_table":
        for episode in tqdm(range(n_episodes)):
            state, info = env.reset()
            done = False
            # play one episode
            while not done:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)

                # update the agent
                agent.update(state, action, reward, terminated, next_state)

                # update if the environment is done and the current state
                done = terminated
                state = next_state

            agent.decay_epsilon()

def save(agent_name: str):
    # Get the current working directory
    current_directory = os.getcwd()

    # Specify the relative path to the parent folder
    relative_folder_path = '..'

    # Create the full path to the pickle file using the relative path and the current working directory
    agents_file_path = os.path.join(current_directory, relative_folder_path, 'agents.pkl')

    try:
        with open(agents_file_path, 'rb') as file:
            agents = pickle.load(file)
    except FileNotFoundError:
        agents = {}

    agents[agent_name] = agent
    with open(agents_file_path, 'wb') as file:
        pickle.dump(agents, file)

#train()
save("q_learning_agent")