import copy
import numpy as np
import torch
from ch08 import ReplayBuffer


class DQNNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 64
        self.state_dim = 4
        self.action_dim = 2
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.dqn_net = DQNNet(self.state_dim, self.action_dim)
        self.dqn_target = DQNNet(self.state_dim, self.action_dim)
        self.optimizer = torch.optim.Adam(self.dqn_net.parameters(), lr=self.lr)

    def sync_dqn_net(self):
        self.dqn_target = copy.deepcopy(self.dqn_net)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            # 增加小批量的维度
            state = state[np.newaxis, :]
            qs = self.dqn_net(torch.FloatTensor(state))
            return torch.argmax(qs).item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.get_batch()
        qs = self.dqn_net(torch.FloatTensor(states))
        q = qs[np.arange(self.batch_size), actions]

        with torch.no_grad():
            next_qs = self.dqn_target(torch.FloatTensor(next_states))
            next_q = torch.max(next_qs, dim=1)[0]
        target = rewards + self.gamma * next_q.detach().numpy() * (1 - dones)

        loss = torch.nn.MSELoss()(q, torch.FloatTensor(target))

        self.dqn_net.zero_grad()
        loss.backward()
        self.optimizer.step()
