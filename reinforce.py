import gymnasium as gym
import numpy as np
from tqdm import tqdm 
import torch 
import time

class Policy(torch.nn.Module):
    def __init__(self, in_dim:int, out_dim:int):
        super().__init__()
        
        # what to make the in_dim
        # BxTx4 4x64 --> BxTx64
        # BxTx64 64x2 --> BxTx2
        self.ffnn = torch.nn.Sequential(
            torch.nn.Linear(in_dim,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,out_dim),
        )

    def get_policy(self,state):
        logits = self(state)
        prob_dist = torch.distributions.Categorical(logits=logits)
        return prob_dist

    def get_action(self,state):
        policy = self.get_policy(state)
        act = policy.sample()
        return act.item()
    
    def forward(self,x):
        logits = self.ffnn(x)
        return logits

    def rtg(self,rews_buf,done_buf):
        rtg_buf = [0]*len(rews_buf)
        for i in reversed(range(len(rtg_buf))):
            """
            if i+1 < len(rtg_buf):
                rtg_buf[i] = rews_buf[i] + ((1-done_buf[i])*rtg_buf[i+1])
            else:
                rtg_buf[i] = rews_buf[i]
            """
            rtg_buf[i] = rews_buf[i] + ((1-done_buf[i])*rtg_buf[i+1] if i+1 < len(rtg_buf) else 0)
        return rtg_buf
    
    def surrogate_loss(self,batch_acts,batch_states,batch_weights):
        # for loop through all the batches
        batch_losses = []
        for d in range(len(batch_acts)):
            logp = self.get_policy(batch_states[d]).log_prob(batch_acts[d])
            loss = -(logp * batch_weights[d]).mean()
            batch_losses.append(loss)
        return torch.as_tensor(batch_losses,dtype=torch.float32).mean()


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
T = 15
lr = 3e-4
gamma = 0.96
lamba = 0.92
optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

start_time = time.perf_counter()

# training loop
for trajectory in tqdm(range(max_iters)):

    batch_rews = []
    batch_acts = []
    batch_states = []
    batch_done = []
    batch_rtgs = []

    for i in range(batch_size):
        # start a trajectory worth T timesteps of data
        state,info = env.reset(seed=env_seed)
        done = False
        
        rews_buf = []
        acts_buf = []
        states_buf = []  
        terminated_buf = []
        truncated_buf = []

        for t in range(T):
            states_buf.append(torch.as_tensor(state,dtype=torch.float32))
            action = policy.get_action(torch.as_tensor(state,dtype=torch.float32))
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
        batch_states.append(states_buf)
        terminated_or_truncated_buf = np.logical_or(np.array(terminated_buf),np.array(truncated_buf))
        done_buf = terminated_or_truncated_buf.tolist()
        batch_done.append(done_buf)
        batch_rtgs.append(policy.rtg(rews_buf,done_buf))

    #batch_states = torch.as_tensor(batch_states,dtype=torch.float32)
    print(batch_states)
    print(type(batch_states[0]))
    print(type(batch_states))
    print(len(batch_states))
    print(len(batch_states[0]))
    break

    batch_avg_rew = torch.sum(torch.as_tensor(batch_rtgs,dtype=torch.float32),1,True).mean().item()
    print(f"Batch {i} avg reward: {batch_avg_rew}")
    
    # update policy using data from batch
    surrogate_loss = policy.surrogate_loss(
        torch.as_tensor(batch_acts,dtype=torch.float32),
        torch.as_tensor(batch_states,dtype=torch.float32),
        torch.as_tensor(batch_rtgs,dtype=torch.float32)
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