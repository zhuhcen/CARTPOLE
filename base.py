import gym

env = gym.make('CartPole-v1')

action = env.action_space
print(action)
