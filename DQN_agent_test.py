import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from othello_env import OthelloEnv
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, discount_factor=0.99, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.discount_factor = discount_factor
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.model.fc3.out_features)
        state = torch.FloatTensor(state).view(1, -1).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1))

        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = rewards + (~dones) * self.discount_factor * next_q_values

        loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Example usage
env = OthelloEnv()
state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
action_size = env.action_space.n

replay_buffer = ReplayBuffer(capacity=1000)

# Training loop
num_episodes = 1000
epsilon = 0.1
batch_size = 5
update_target_interval = 10
total_rewards = []
agent1 = DQNAgent(state_size, action_size)
agent2 = DQNAgent(state_size, action_size)

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        current_player = env.current_player
        if current_player == 1:
            action = agent1.select_action(state, epsilon)
        else:
            action = agent2.select_action(state, epsilon)

        next_state, reward, done, _ = env.step(current_player, action)

        # Store the experience in the replay buffer
        replay_buffer.add((state.flatten(), action, reward, next_state.flatten(), done))

        # Sample a batch from the replay buffer and train the DQN agent
        if len(replay_buffer.buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            if current_player == 1:
                agent1.train(batch)
            else:
                agent2.train(batch)

        # Update the target network periodically
        if episode % update_target_interval == 0:
            agent1.update_target_model()
            agent2.update_target_model()

        total_reward += reward
        state = next_state

        if done:
            break

    total_rewards.append(total_reward)
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
# Plot the results
plt.plot(total_rewards)
plt.title('Total Rewards Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()