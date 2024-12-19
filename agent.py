import torch
import gym
from DQN import DQN

max_episodes = 1000

def main():
    env = gym.make("CartPole-v1")
    agent = DQN(5e-5, env.observation_space.shape[0], env.action_space.n, 256, 256, None, 0.005, 0.99, 1, 0.05, 5e-4, 10000, 256)

    for i in range(max_episodes):
        total_reward_per_epi = 0
        state = env.reset()[0]
        while (True): 
            action = agent.choose_action(state, isTrain=True)
            next_state, reward, terminal, truncated, _ = env.step(action)
            agent.replay_buffer.store_once(state, action, reward, next_state, terminal)
            if terminal or truncated:
                continue
            state = next_state
        


        