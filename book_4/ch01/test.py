import numpy as np

class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)  # 每台机器的胜率
    
    def play(self, arm):
        rate = self.rates[arm]
        if (rate > np.random.rand()):
            return 1  # 赢
        else:
            return 0  # 输

class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.qs = np.zeros(action_size)  # 各动作的价值初始为0
        self.ns = np.zeros(action_size)  # 各动作的选择次数初始为0
    
    def update(self, action, reward):
        self.ns[action] += 1
        self.qs[action] += (reward - self.qs[action]) / self.ns[action]
    
    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.qs))  # 探索
        else:
            return np.argmax(self.qs)  # 利用




if __name__ == '__main__':
    bandit = Bandit()
    for i in range(3):
        print(bandit.play(0))