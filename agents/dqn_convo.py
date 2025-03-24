import torch
import numpy as np
import torch.nn as nn
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','winner'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_CNN(nn.Module):
    def __init__(self, in_channels=1, n_actions=7):
        super(DQN_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16,
                               kernel_size=3, stride=1,
                               padding=1)  # (6,7) → (6,7)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=1,
                               padding=1)  # (6,7) → (6,7)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 6 * 7, n_actions)

    def forward(self, x):
        x = x.view(-1, 1, 6, 7)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        return x.squeeze(0)  #


class DQN_CNN_agent(nn.Module):

    def __init__(self, env,
                 epsilon=0.99, eps_decay=.999,
                 gamma=1.0,
                 alpha=0.02,
                 state_size=42,
                 action_size=7,
                 batch_size=128,
                 target_update_freq=500) -> None:
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_decay = eps_decay
        self.alpha = alpha
        self.replay_memory = ReplayMemory(capacity=10000)
        self.memory = self.replay_memory.memory

        self.model = DQN_CNN()

        self.target_model = DQN_CNN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # freeze the weights

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.alpha)

        self.loss_list = []
        self.target_update_counter = 0
        self.target_update_freq = target_update_freq

    def reset(self):
        # reset our agent
        self.epsilon = max(self.eps_decay * self.epsilon, 0.05)

    def select_action(self, state, legal_moves):

        state = torch.tensor(state, dtype=torch.float32)

        if np.random.random() < self.epsilon:
            return np.random.choice(legal_moves)
        else:
            with torch.no_grad():
                q_values = self.model(state).cpu().numpy()
                return max(legal_moves, key=lambda x: q_values[x])

    def step(self, state, action, reward, next_state, winner):
        self.replay_memory.push(state, action, next_state, reward, winner)
        if len(self.memory) >= self.batch_size:
            self.update()

    def update(self):

        batch = self.replay_memory.sample(self.batch_size)
        states, actions, next_states, rewards, winners = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        terminals = torch.tensor([winner != -1 for winner in winners],
                                 dtype=torch.bool)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]

        target_q_values = rewards + (self.gamma * next_q_values * (~terminals))

        loss = nn.SmoothL1Loss()(current_q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.loss_list.append(loss.item())

        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
