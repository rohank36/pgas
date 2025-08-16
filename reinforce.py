import time
import gymnasium as gym
import torch 

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
#env = gym.make('CartPole-v1', render_mode="human")
env = gym.make('CartPole-v1')
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

print("\nstart training...")

# training loop
for trajectory in range(max_iters):

    batch_acts = []
    batch_states = []
    batch_weights = []

    render = True

    for batch in range(batch_size):
        state,info = env.reset(seed=env_seed)
        done = False
        rews_buf = []

        while True:
            #if render:
                #env.render()

            batch_states.append(state.tolist())
            action = policy.get_action(torch.as_tensor(state,dtype=torch.float32))
            batch_acts.append(action)
            next_state,reward,terminated,truncated,_ = env.step(action)
            rews_buf.append(reward)
            
            if terminated or truncated:
                rtgs = policy.rtg(rews_buf)
                print(rtgs)
                batch_weights.extend(rtgs)  
                break
            else:
                state = next_state

        render = False

    print(f"Batch {batch} avg reward: {sum(batch_weights)/len(batch_weights)}")
    
    # update policy using data from batch
    surrogate_loss = policy.surrogate_loss(
        torch.as_tensor(batch_acts,dtype=torch.float32),
        torch.as_tensor(batch_states,dtype=torch.float32),
        torch.as_tensor(batch_weights,dtype=torch.float32)
        )
    
    optimizer.zero_grad()
    surrogate_loss.backward()
    optimizer.step()
            
env.close()

end_time = time.perf_counter()
train_duration = end_time - start_time
print(f"Training time (mins): {train_duration/60}")
print(f"Training time (secs): {train_duration}")