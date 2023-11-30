import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


# Defining the Q-network:
class Q_Network(nn.Module):
    def __init__(self, state_space, action_space):
        super(Q_Network, self).__init()
        self.fc1 = nn.Linear(state_space, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Defining the DQN agent:
class DQN_agent:
    def __init__(self, state_space, action_space, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen(1000))
        self.batch_size = 64
        self.model = Q_Network(state_space, action_space)
        self.target_model = Q_Network(state_space, action_space)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state)
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()

            state = torch.FloatTensor(state)
            q_values = self.model(state).tolist()
            q_values[action] = target
            states.append(state)
            targets.append(q_values)

        states = torch.stack(states)
        targets = torch.tensor(targets)
        self.optimizer.zero_grad()
        loss = F.mse_loss(self.model(states), targets)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state.dict())


# Example
# what actually is the state/action space in a game of Othello?
agent = DQN_agent(state_space=64 ** 4, action_space=8)
