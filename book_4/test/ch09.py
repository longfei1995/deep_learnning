import numpy as np
import gymnasium as gym
import torch


class Policy(torch.nn.Module):
    def __init__(self, action_size):
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(4, 128)
        self.fc2 = torch.nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x
    
class Agent:
    def __init__(self):
        self.gamma = 0.98       # 折现率
        self.lr = 0.0002        # 学习率
        self.action_size = 2    # 动作空间大小

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = torch.optim.Adam(self.pi.parameters(), lr=self.lr)

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(torch.FloatTensor(state))
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data.numpy())
        return action, probs[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        self.pi.zero_grad()

        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G
            loss += -torch.log(prob) * G

        loss.backward()
        self.optimizer.step()
        self.memory = []


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state = env.reset()[0]
    agent = Agent()
    
    action, prob = agent.get_action(state)
    print(f"Action: {action}, Probability: {prob.data.numpy()}")
    
    # 虚拟权重
    G = 100.0
    J =  - G * torch.log(prob)
    print(f"Loss: {J.data.numpy()}")
    
    # 求梯度
    J.backward()
    print(f"Gradient of fc1 weights: {agent.pi.fc1.weight.grad}")