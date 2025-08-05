import gymnasium as gym
import numpy as np
from tqdm import tqdm 
"""
class Agent:
    def __init__(
        self,
        env: gym.Env,
        lr: float,
        #more
    ):
        self.env = env
        self.lr = lr

    def get_action(self):
        action = self.env.action_space.sample()
        return action

    def update(self):
        pass

#Train
n_episodes = 100
lr = 0.001
env = gym.make('CartPole-v1', render_mode="human")
agent = Agent(env,lr)

for episode in tqdm(range(n_episodes)):
    obs,info = env.reset(seed=42)
    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs,reward,terminated,truncated,info = env.step(action)

        agent.update(obs,action,reward,terminated,next_obs)

        done = terminated or truncated
        obs = next_obs

env.close()


"""
#Test version
#env = gym.make('CartPole-v1',render_mode="human")
env = gym.make('CartPole-v1')
obs,info = env.reset(seed=42)
total_reward = 0
episode_rewards = []
episode_durations = []
for i in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        episode_rewards.append(total_reward)
        if len(episode_durations) == 0:
            episode_durations.append(i)
        else:
            episode_durations.append(i-episode_durations[len(episode_durations)-1])
        total_reward = 0
        observation,info = env.reset()

env.close()
print(f"Durations: {episode_durations}")
print(f"Max Duation: {max(episode_durations)}")
print(f"Rewards: {episode_rewards}")
print(f"Max Reward: {max(episode_rewards)}")
