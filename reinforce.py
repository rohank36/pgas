import gymnasium as gym
import torch
import time
import matplotlib.pyplot as plt

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
        logp = self.get_policy(batch_states).log_prob(batch_acts)
        return -(logp * batch_weights).mean()
    

torch.manual_seed(27)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_render = gym.make('CartPole-v1', render_mode="human")
env_headless = gym.make('CartPole-v1')
env = env_headless

in_dim = env.observation_space.shape[0] # (4,)
out_dim = env.action_space.n # Discrete action space, (2,) possible actions for CartPole env
hidden_dim = 32
policy = Policy(in_dim,hidden_dim,out_dim)
#policy.to(device=device)

max_iters = 31 #100
batch_size = 30 # 50
lr = 3e-3
#gamma = 0.96
#lambda_param = 0.92
optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

start_time = time.perf_counter()

print("\nstart training...")

# training loop
avg_reward = []
for batch in range(max_iters):

    batch_acts, batch_states, batch_weights = [], [], []

    env = env_render if batch % 10 == 0 else env_headless

    for episode in range(batch_size):
        state,info = env.reset()
        done = False
        rews_buf = []

        while True:
            batch_states.append(state.tolist())
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

    batch_avg_reward = sum(batch_weights)/len(batch_weights)
    avg_reward.append(batch_avg_reward) # use to plot later
    if batch % 10 == 0:
        print(f"Batch {batch} avg reward: {batch_avg_reward}")
    
    surrogate_loss = policy.surrogate_loss(
        torch.as_tensor(batch_acts,dtype=torch.float32),
        torch.as_tensor(batch_states,dtype=torch.float32),
        torch.as_tensor(batch_weights,dtype=torch.float32)
        )
    
    optimizer.zero_grad()
    surrogate_loss.backward()
    optimizer.step()
            
env_render.close()
env_headless.close()

end_time = time.perf_counter()
train_duration = end_time - start_time
print(f"Training time (mins): {train_duration/60}")
print(f"Training time (secs): {train_duration}")

plt.plot([i for i in range(1,max_iters+1)], avg_reward) 
plt.xlabel("Batch")  
plt.ylabel("Average Reward")  
plt.title("Agent Reward")
plt.show()        
#plt.savefig('batch_avg_reward.png')