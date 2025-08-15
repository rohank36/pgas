import gymnasium as gym
import numpy as np
from tqdm import tqdm 
import torch 
import time

class Policy(torch.nn.Module):
    def __init__(self, in_dim:int, hidden_dim:int, out_dim:int):
        super().__init__()
        
        self.ffnn = torch.nn.Sequential(
            torch.nn.Linear(in_dim,hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim,out_dim),
        )

    def get_policy(self,state):
        logits = self.ffnn(state)
        prob_dist = torch.distributions.Categorical(logits=logits)
        return prob_dist

    def get_action(self,state):
        policy = self.get_policy(state)
        act = policy.sample()
        return act.item()
    
    def rtg(self,rews_buf):
        rtg_buf = [0]*len(rews_buf)
        for i in reversed(range(len(rtg_buf))):
            rtg_buf[i] = rews_buf[i] + (rtg_buf[i+1] if i+1 < len(rtg_buf) else 0)
        return rtg_buf
    
    def surrogate_loss(self,batch_acts,batch_states,batch_weights):
        #print(f"Batch Acts: {batch_acts.shape}")
        #print(f"Batch States: {batch_states.shape}")
        #print(f"Batch Weights: {batch_weights.shape}")
        
        logp = self.get_policy(batch_states).log_prob(batch_acts)
        return -(logp * batch_weights).mean()


torch.manual_seed(27)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v1', render_mode="human")
#env = gym.make('CartPole-v1')
env_seed = 42

in_dim = env.observation_space.shape[0] # (4,)
out_dim = env.action_space.n # Discrete action space, (2,) possible actions for CartPole env
hidden_dim = 32
policy = Policy(in_dim,hidden_dim,out_dim)
#policy.to(device=device)

max_iters = 1
batch_size = 2
lr = 3e-4
#gamma = 0.96
#lambda_param = 0.92
optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

start_time = time.perf_counter()

# training loop
for trajectory in tqdm(range(max_iters)):

    batch_acts = []
    batch_states = []
    batch_weights = []

    render = True

    for batch in range(batch_size):
        state,info = env.reset(seed=env_seed)
        done = False
        rews_buf = []

        while True:
            if render:
                env.render()

            batch_states.append(state)
            action = policy.get_action(torch.as_tensor(state,dtype=torch.float32))
            batch_acts.append(action)
            next_state,reward,terminated,truncated,_ = env.step(action)
            rews_buf.append(reward)
            
            if terminated or truncated:
                rtgs = policy.rtg(rews_buf)
                batch_weights.extend(rtgs)  
                break
            else:
                state = next_state

        render = False

    print(f"Batch {batch} avg reward: {sum(batch_weights)/len(batch_weights)}")
    
    # update policy using data from batch
    surrogate_loss = policy.surrogate_loss(
        torch.as_tensor(batch_acts,dtype=torch.float32),
        torch.from_numpy(batch_states),
        torch.as_tensor(batch_weights,dtype=torch.float32)
        )
    
    optimizer.zero_grad()
    surrogate_loss.backward()
    optimizer.step()
            
env.close()

#print(f"Actions ({len(batch_acts)},{len(batch_acts[0])}):\n{batch_acts}")
#print(f"Rewards ({len(batch_rews)},{len(batch_rews[0])}):\n{batch_rews}")
#print(f"RTGs ({len(batch_rtgs)},{len(batch_rtgs[0])}):\n{batch_rtgs}")
#print(f"Done ({len(batch_done)},{len(batch_done[0])}):\n{batch_done}")
#print(f"Traj1 avg reward: {sum(batch_rtgs[0])/T}")
#print(f"Traj2 avg reward: {sum(batch_rtgs[1])/T}")

end_time = time.perf_counter()
duration_minutes = (end_time - start_time) / 60
print(f"Training time: {duration_minutes}")