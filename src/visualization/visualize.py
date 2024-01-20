from src.environments import bridge
from src.environments import bridge_world_drowningPeople
from src.algorithms import q_learning
import numpy as np
import gymnasium as gym
import json
import pickle
import os

#env = gym.make('bridge_world-v0', render_mode=None)

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

def visualize(agent_name):
        with open(agents_file_path, "rb") as file:
                agents = pickle.load(file)
        agent = agents[agent_name]
        agent.epsilon = 0
        while True:
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

visualize("q_learning_agent")
    
# env = data.get("env", "bridge_world-v0")

# def instantiate_agent(agent_type, config):
#     if agent_type == "DQNAgent":
#         from q_learning import q_learning_agent
#         return (config["other_config_option"])
#     elif agent_type == "AnotherAgent":
#         pass
#     # Add more elif blocks for other agent types as needed
#     else:
#         raise ValueError(f"Unknown agent type: {agent_type}")

#TODO: funktion schreiben, die beiliebigen agent übergeben bekommt (typ: gym.agent oder so nimmt) und dessen verhalten dann visualisiert
#TODO: main funktion(?) schreiben, die aus konsole argument nimmt, welcher agent übergeben werden soll