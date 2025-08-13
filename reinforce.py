import gymnasium as gym
import numpy as np
from tqdm import tqdm 
import torch 
import time


class Agent:
    def __init__(
        self,
        env: gym.Env,
        lr: float,
        max_iters:int
    ):
        self.env = env
        self.lr = lr

    def get_action(self):
        action = self.env.action_space.sample()
        return action

    def update(self):
        pass

    def loss(self):
        pass

    def one_iter(self):
        pass

    def train(self):
        for i in range(self.max_iters):
            self.one_iter()

class Policy(torch.nn.Module):
    def __init__(self, in_dim:int, out_dim:int):
        super().__init__()
        
        self.ffnn = torch.nn.Sequential(
            torch.nn.Linear(in_dim,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,out_dim),
        )

    def surrogate_loss_fn(self):
        pass

    def get_action(self,state) -> int:
        logits = self(state)
        prob_dist = torch.distributions.Categorical(logits=logits)
        act = prob_dist.sample()
        return act.item()
    
    def forward(self,x):
        logits = self.ffnn(x)
        return logits

class ValueFunction(torch.nn.Module):
    def __init__(self):
        pass
    
    def forward():
        pass

torch.manual_seed(27)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#env = gym.make('CartPole-v1', render_mode="human")
env = gym.make('CartPole-v1')
env_seed = 42

in_dim = env.observation_space.shape[0] # (4,)
out_dim = env.action_space.n # Discrete action space, (2,) possible actions for CartPole env
policy = Policy(in_dim,out_dim)
#policy.to(device=device)

max_iters = 1
batch_size = 2
T = 15 # change to 500
lr = 3e-4
#agent = Agent(env,lr)

start_time = time.perf_counter()

# training loop
for trajectory in tqdm(range(max_iters)):

    batch_rews = []
    batch_acts = []
    batch_done = []

    for _ in tqdm(range(batch_size)):
        # start a trajectory worth T timesteps of data
        state,info = env.reset(seed=env_seed)
        done = False
        
        rews_buf = []
        acts_buf = []
        terminated_buf = []
        truncated_buf = []

        for t in range(T):
            action = policy.get_action(torch.tensor(state))
            next_state,reward,terminated,truncated,_ = env.step(action)

            rews_buf.append(reward)
            acts_buf.append(action)
            terminated_buf.append(terminated)
            truncated_buf.append(truncated)

            if terminated or truncated:
                state,_ = env.reset()
            else:
                state = next_state

        # adds bufs to batch
        batch_rews.append(rews_buf)
        batch_acts.append(acts_buf)
        terminated_or_truncated_buf = np.logical_or(np.array(terminated_buf),np.array(truncated_buf))
        batch_done.append(terminated_or_truncated_buf.tolist())
    
    # update policy using data from batch
            
env.close()

print(f"Actions ({len(batch_acts)},{len(batch_acts[0])}):\n{batch_acts}")
print(f"Rewards ({len(batch_rews)},{len(batch_rews[0])}):\n{batch_rews}")
print(f"Done ({len(batch_done)},{len(batch_done[0])}):\n{batch_done}")

end_time = time.perf_counter()
duration_minutes = (end_time - start_time) / 60
print(f"Training time: {duration_minutes}")


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
"""