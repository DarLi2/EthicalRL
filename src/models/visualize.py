from src.environments import bridge
from src.environments import bridge_world_drowningPeople
from src.algorithms import q_learning_agent_table
import numpy as np
import gymnasium as gym
import json
import pickle
import os

#env = gym.make('bridge_world-v0', render_mode=None)

path = os.path.abspath(os.path.dirname(__file__))
agents_file_path = os.path.join(path, 'agents.pkl')
json_file_path = os.path.join(path, "train_setup/agent_env_selection.json")


with open(json_file_path, "r") as read_file:
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
                # run one episode
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


#TODO: funktion schreiben, die beiliebigen agent übergeben bekommt (typ: gym.agent oder so nimmt) und dessen verhalten dann visualisiert