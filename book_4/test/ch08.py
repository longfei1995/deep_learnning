import gymnasium as gym
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)
    
    def get_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)


if __name__ == "__main__":
    env = gym.make("CartPole-v0", render_mode="human")
    replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=64)
    
    for episode in range(10):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = env.action_space.sample()  # Random action
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            if len(replay_buffer) > replay_buffer.batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.get_batch()
                print(f"Batch of states: {states.shape}, actions: {actions.shape}, rewards: {rewards.shape}, next_states: {next_states.shape}, dones: {dones.shape}")