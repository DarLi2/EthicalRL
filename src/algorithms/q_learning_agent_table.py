import numpy as np

import gymnasium as gym

from collections import defaultdict


class QLearningAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        q_values = None,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
            envrionment: The environment the agent is trained on. 
        """

        self.env = env
        
        self.q_values = defaultdict(self.default_q_values)

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def default_q_values(self):
        return np.zeros(self.env.action_space.n)

#TODO: put function in extra module such that it can be called by other agents architectures
    def convert_observations(self, observations):
        obs_list = []

        # Convert "agent" and "target" to tuples
        for key in ["agent", "target"]:
            for element in observations[key]:
                obs_list.append(element)

        person = observations["person"]
        for element in person["position"]:
            obs_list.append(element)
        obs_list.append(int(person["in_water"]))

        tuple_obs = tuple(obs_list)
        return tuple_obs


    def get_action(self, obs) -> int:
        obs = self.convert_observations(obs)
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs
    ):

        """Updates the Q-value of an action."""
        obs = self.convert_observations(obs)
        next_obs = self.convert_observations(next_obs)

        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)