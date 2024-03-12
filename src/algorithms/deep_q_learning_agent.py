import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F

# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # ouptut layer w

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = self.out(x)         # Calculate output
        return x

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size) #oe sample is a tuple: (reward, state, truncated, done)

    def __len__(self):
        return len(self.memory)
    
class DeepQLearningAgent():
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        network_sync_rate: int,
        replay_memory_size: int,
        mini_batch_size: int,
        loss_fn,
        optimizer
    ):
        self.env = env

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.network_sync_rate = network_sync_rate
        self.replay_memory_size = replay_memory_size

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        num_states = env.observation_space.n
        num_actions = env.action_space.n
        
        epsilon = initial_epsilon # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        #State_dict:In PyTorch, the learnable parameters (i.e. weights and biases) of a torch.nn.Module model are contained in the model’s parameters (accessed with model.parameters()). A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        self.print_dqn(policy_dqn)

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. ; the policy network is optimized in every step, while the target network is kept stable; the update for the target network is done by copying the policy network's parameters as the target every few steps
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        self.step_count=0

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

            obs_tensor = torch.tensor(obs_list)
            return obs_tensor

        def get_action(state):
            obs = self.convert_observations(obs)
            # with probability epsilon return a random action to explore the environment
            if np.random.random() < self.epsilon:
                return self.env.action_space.sample()

            # with probability (1 - epsilon) act greedily (exploit)
            else:
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()
                return action
            
        def optimize(self, mini_batch, policy_dqn, target_dqn):

            # Get number of input nodes
            num_states = policy_dqn.fc1.in_features

            current_q_list = []
            target_q_list = []

            for state, action, new_state, reward, terminated in mini_batch:

                if terminated: 
                    # Agent either reached goal (reward=1) or fell into hole (reward=0)
                    # When in a terminated state, target q value should be set to the reward.
                    target = torch.FloatTensor([reward])
                else:
                    # Calculate target q value 
                    with torch.no_grad():
                        target = torch.FloatTensor(
                            # note that this line is equivalent to action = policy_dqn.forward(torch.tensor([state], dtype=torch.float32)).argmax().item(); i.e. calling the instance of the model is the same as calling the forward-method in the instance of the model
                            reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                        )

                # Get the current set of Q values
                current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
                current_q_list.append(current_q)

                # Get the target set of Q values
                target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 
                # Adjust the specific action to the target that was just calculated
                target_q[action] = target
                target_q_list.append(target_q)
                    
            # Compute loss for the whole minibatch
            loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def update(state, action, reward, terminated, new_state):

            self.step_count += 1
            memory.append((state, action, new_state, reward, terminated)) 

            if len(memory)>self.mini_batch_size:
                    # do an update step for the policy network by sampling from the mini-batch (experience-replay buffer)
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)        

                    # only every few steps: copy policy network to target network and thereby update the target network (avoid a "moving target")
                    if self.step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        self.step_count=0

        def decay_epsilon(self):
            self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


        #TODO:create an instance of the DQN
        #create new model with passed on paramters or loading model directly via torch.ooad