import torch
import gym
from DQN import DQN
import numpy as np
from utils import *

max_episodes = 1000
render_mode = "human"
# render_mode = ''
# playground = "CartPole-v1"
playground = "MountainCar-v0"

def main():
    if render_mode == "human":
        env = gym.make(playground, render_mode="human")
    else:
        env = gym.make(playground)
    agent = DQN(5e-5, env.observation_space.shape[0], env.action_space.n, 256, 256, None, 0.005, 0.99, 1, 0.05, 5e-4, 10000, 256)


    total_rewards = []
    avg_rewards = []
    eps_history = []

    for i in range(max_episodes):
        total_reward_per_epi = 0
        state = env.reset()[0]
        while (True): 
            action = agent.choose_action(state, isTrain=True)
            next_state, reward, terminal, truncated, info = env.step(action)
            agent.remember(state, action, reward, next_state, terminal | truncated)
            agent.learn()
            total_reward_per_epi += reward
            if terminal or truncated:
                break
            state = next_state

        total_rewards.append(total_reward_per_epi)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        eps_history.append(agent.epsilon)
        print("Episode:{}, total_reward:{}, avg_reward:{}, epsilon:{}".
              format(i+1, total_reward_per_epi, avg_reward, agent.epsilon))
        
    episodes = [i for i in range(max_episodes)]
    plot_learning_curve(episodes, avg_rewards, 'Reward_avg', 'reward_avg', 'D:\\study\\rl\CartPole\\reward_avg.png')
    plot_learning_curve(episodes, total_rewards, 'Reward', 'reward', 'D:\\study\\rl\\CartPole\\reward.png')
    plot_learning_curve(episodes, eps_history, 'Epsilon', 'epsilon', 'D:\\study\\rl\\CartPole\\epsilon.png')




if __name__ == '__main__':
    main()
        


        