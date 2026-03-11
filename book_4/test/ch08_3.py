from ch08_2 import DQNAgent

import gymnasium as gym






if __name__ == "__main__":
    episodes = 300
    sync_interval = 20
    env = gym.make("CartPole-v0", render_mode="human")
    agent = DQNAgent()
    reward_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
        reward_history.append(episode_reward)
        
        if (episode + 1) % sync_interval == 0:
            agent.sync_dqn_net()
    