import gymnasium as gym
import torch
import time
import matplotlib.pyplot as plt

class PPOPolicy(torch.nn.Module):
    def __init__(self, in_dim:int, hidden_dim:int, out_dim:int):
        super().__init__()
        self.epsilon = 0.2

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
    
    
    def rtg(self,rews_buf,gamma):
        rtg_buf = [0]*len(rews_buf)
        for i in reversed(range(len(rtg_buf))):
            rtg_buf[i] = rews_buf[i] + (gamma*rtg_buf[i+1] if i+1 < len(rtg_buf) else 0)
        return rtg_buf
    
    def surrogate_loss(self,batch_acts,batch_states,batch_weights):
        #print(f"Batch acts: {batch_acts.shape} ")
        #print(f"Batch states: {batch_states.shape} ")
        #print(f"Batch weights: {batch_weights.shape} ")



        logp = self.get_policy(batch_states).log_prob(batch_acts)
        return -(logp * batch_weights).mean()
    

class ValueFunction(torch.nn.Module):
    def __init__(self, in_dim:int, hidden_dim:int, out_dim:int):
        super().__init__()

        self.ffnn = torch.nn.Sequential(
            torch.nn.Linear(in_dim,hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim,out_dim),
        )

    def get_values(self,state):
        logits = self.ffnn(state)
        return logits.squeeze(-1)
    
    def loss_fn(self,batch_states,targets):
        preds = self.ffnn(batch_states)
        preds = preds.squeeze(-1)

        #print(f"Preds: {preds.shape}")
        #print(f"Targets: {targets.shape} ")
        #print("\n")

        return torch.nn.functional.mse_loss(preds,targets)
    
def GAE(value_fn,rewards,states,next_states,lam,gam,terminated):
    # TD_t = rew[t] + gam*V(s_t+1)*(1-done_t) - V(s_t)
    # A_t = TD_t + gam*lam*A_t+1*(1-done_t)

    with torch.no_grad():
        v_s = value_fn.get_values(states)
        v_last = torch.tensor(0.0) if terminated else value_fn.get_values(next_states[-1]) # if terminated last value should be 0 else bootstrap
        v_full = torch.cat([v_s, v_last.unsqueeze(0)], dim=0) # +1

    dones = torch.zeros(len(rewards), dtype=torch.float32)
    if terminated:
        dones[-1] = 1.0
    not_done = 1.0 - dones


    adv = torch.zeros_like(rewards)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        td = rewards[t] + gam * v_full[t+1] * not_done[t] - v_full[t]
        gae   = td + gam * lam * not_done[t] * gae
        adv[t] = gae

    targets = adv + v_full[:-1]

    """
    print(f"T: {len(rewards)}")
    print(f"V(s): {v_s.shape}")
    print(f"V full: {v_full.shape}")
    print(f"advantage type: {type(adv)}")
    print(f"advantage type[0]: {type(adv[0])}")
    print(adv)
    print(f"targets type: {type(targets)}")
    print(f"targets type[0]: {type(targets[0])}")
    print(targets)
    print("\n")
    """

    return adv.tolist(),targets.tolist()

    
if __name__ == "__main__":
    saved_policy_filename = "ppo_policy.pth"
    saved_valuefn_filename = "ppo_value.pth"
    serious_training_run = False

    torch.manual_seed(27)
    env_render = gym.make('CartPole-v1', render_mode="human")
    env_headless = gym.make('CartPole-v1')
    env = env_headless

    in_dim = env.observation_space.shape[0] # (4,)
    out_dim = env.action_space.n # (2,)
    hidden_dim = 32
    policy = PPOPolicy(in_dim,hidden_dim,out_dim)
    value_fn = ValueFunction(in_dim,hidden_dim,1)
    
    max_iters = 100
    batch_size = 32 #32
    policy_lr = 3e-4
    value_lr  = 1e-3
    gam = 0.99
    lam = 0.95
    optimizer_policy = torch.optim.AdamW(policy.parameters(), lr=policy_lr)
    optimizer_value_fn = torch.optim.AdamW(value_fn.parameters(), lr=value_lr)

    start_time = time.perf_counter()

    print("\nstart training...")

    # training loop
    avg_advantage = []
    avg_len = []
    value_fn_loss = []

    for batch in range(max_iters):

        batch_acts, batch_states, batch_weights, batch_lens, batch_targets = [], [], [], [], []

        for episode in range(batch_size):
            state,info = env.reset()
            done = False
            rews_buf = []
            states_buf = []
            next_states_buf = []
            t = 0

            while True:
                t += 1
                batch_states.append(state.tolist())
                states_buf.append(state.tolist())
                action = policy.get_action(torch.as_tensor(state,dtype=torch.float32))
                batch_acts.append(action)
                next_state,reward,terminated,truncated,_ = env.step(action)
                next_states_buf.append(next_state.tolist())
                rews_buf.append(reward)
                
                if terminated or truncated:
                    #rtgs = policy.rtg(rews_buf,gam)
                    advs,targets = GAE(
                        value_fn=value_fn,
                        rewards = torch.as_tensor(rews_buf, dtype=torch.float32),
                        states=torch.as_tensor(states_buf,dtype=torch.float32),
                        next_states=torch.as_tensor(next_states_buf,dtype=torch.float32),
                        lam=lam,
                        gam=gam,
                        terminated=terminated
                    )
                    batch_weights.extend(advs)
                    batch_targets.extend(targets) 
                    #batch_targets.extend(rtgs) 
                    batch_lens.append(len(rews_buf))  
                    break
                else:
                    state = next_state

        batch_avg_advantage = sum(batch_weights)/len(batch_weights)
        batch_avg_len = sum(batch_lens)/len(batch_lens)
        avg_advantage.append(batch_avg_advantage) 
        avg_len.append(batch_avg_len)

        if batch % 10 == 0:
            print(f"Batch {batch} avg adv: {batch_avg_advantage}")
            print(f"Batch {batch} avg len: {batch_avg_len}")
        
        # normalize batch weights (advantages)
        batch_weights_tensor = torch.as_tensor(batch_weights,dtype=torch.float32)
        batch_weights_normd = (batch_weights_tensor - batch_weights_tensor.mean()) / (batch_weights_tensor.std() + 1e-8)

        surrogate_loss = policy.surrogate_loss(
            torch.as_tensor(batch_acts,dtype=torch.long),
            torch.as_tensor(batch_states,dtype=torch.float32),
            batch_weights_normd
            )
        
        value_loss = value_fn.loss_fn(
            torch.as_tensor(batch_states,dtype=torch.float32),
            torch.as_tensor(batch_targets,dtype=torch.float32)
            )
        
        optimizer_policy.zero_grad()
        surrogate_loss.backward()
        optimizer_policy.step()

        optimizer_value_fn.zero_grad()
        value_loss.backward()
        optimizer_value_fn.step()

        value_fn_loss.append(value_loss.item())
                
    env_render.close()
    env_headless.close()

    end_time = time.perf_counter()
    train_duration = end_time - start_time
    print(f"Training time (mins): {train_duration/60}")
    print(f"Training time (secs): {train_duration}")

    if serious_training_run:
        torch.save(policy.state_dict(),saved_policy_filename)
        torch.save(value_fn.state_dict(),saved_valuefn_filename)

    plt.plot([i for i in range(1,max_iters+1)], avg_advantage) 
    plt.xlabel("Batch")  
    plt.ylabel("Average Advantage")  
    plt.title("Agent Advantage")
    plt.savefig('ppo_batch_avg_advantage.png')
    plt.show()        

    plt.plot([i for i in range(1,max_iters+1)], avg_len) 
    plt.xlabel("Batch")  
    plt.ylabel("Average Len")  
    plt.title("Episode Lengths")
    plt.savefig('ppo_batch_avg_len.png')     
    plt.show()  

    plt.plot([i for i in range(1,max_iters+1)], value_fn_loss) 
    plt.xlabel("Batch")  
    plt.ylabel("MSE Loss")  
    plt.title("Value Function Loss")
    plt.savefig('ppo_value_fn_loss.png')     
    plt.show() 