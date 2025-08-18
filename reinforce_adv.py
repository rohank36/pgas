import gymnasium as gym
import torch
import time
import matplotlib.pyplot as plt

class ReinforcePolicy(torch.nn.Module):
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
    
    def rtg(self,rews_buf,gam=0.99):
        rtg_buf = [0]*len(rews_buf)
        for i in reversed(range(len(rtg_buf))):
            rtg_buf[i] = rews_buf[i] + (gam*rtg_buf[i+1] if i+1 < len(rtg_buf) else 0)
        return rtg_buf
    
    def surrogate_loss(self,batch_acts,batch_states,batch_weights):
        logp = self.get_policy(batch_states).log_prob(batch_acts)
        return -(logp * batch_weights).mean()
    

class Value(torch.nn.Module):
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
        
        if torch.var(targets)<1e-8:
            ev = 0
        else:
            ev = 1 - (torch.var(targets-preds)/torch.var(targets))
 
        return torch.nn.functional.huber_loss(preds,targets), ev.item()

def GAE(V,rewards,states,last,lam,gam,terminated):
    with torch.no_grad():
        v_s = V.get_values(states) # V(s)
        v_last = torch.tensor(0.0) if terminated else V.get_values(last) # if terminated last value should be 0 else bootstrap
        v_next = torch.cat((v_s[1:], v_last.unsqueeze(0))) # V(s_t+1)

    dones = torch.zeros(len(rewards), dtype=torch.float32)
    if terminated:
        dones[-1] = 1.0
    not_done = 1.0 - dones

    td = rewards + gam * v_next * not_done - v_s
    adv = torch.zeros_like(rewards)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        gae = td[t] + gam * lam * not_done[t] * gae
        adv[t] = gae

    targets = adv + v_s

    return adv.tolist(),targets.tolist()
    

if __name__ == "__main__":
    saved_policy_filename = "reinforce_gae_policy.pth"
    saved_value_filename = "reinforce_gae_value.pth"
    serious_training_run = False

    torch.manual_seed(27)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_render = gym.make('CartPole-v1', render_mode="human")
    env_headless = gym.make('CartPole-v1')
    env = env_headless

    in_dim = env.observation_space.shape[0] # (4,)
    out_dim = env.action_space.n # Discrete action space, (2,) possible actions for CartPole env
    hidden_dim = 32
    policy = ReinforcePolicy(in_dim,hidden_dim,out_dim)
    value = Value(in_dim,hidden_dim,1)
    #policy.to(device=device)

    gam = 0.99
    lam = 0.95
    max_iters = 71
    batch_size = 32
    policy_lr = 3e-2
    value_lr = 3e-3
    optimizer_policy = torch.optim.AdamW(policy.parameters(), lr=policy_lr)
    optimizer_value = torch.optim.AdamW(value.parameters(), lr=value_lr, weight_decay=0.0)

    start_time = time.perf_counter()

    print("\nstart training...")

    # training loop
    avg_len = []
    batch_value_loss = []
    ev_vals = []
    epochs = 20
    for batch in range(max_iters):

        batch_acts, batch_states, batch_weights, batch_lens, batch_targets = [], [], [], [], []

        #env = env_render if batch % 10 == 0 else env_headless

        for episode in range(batch_size):
            state,info = env.reset()
            done = False
            rews_buf = []
            states_buf = []

            while True:
                batch_states.append(state.tolist())
                states_buf.append(state.tolist())
                action = policy.get_action(torch.as_tensor(state,dtype=torch.float32))
                batch_acts.append(action)
                next_state,reward,terminated,truncated,_ = env.step(action)
                rews_buf.append(reward)
                
                if terminated or truncated:
                    """
                    # for adv = rtg - V(s)
                    rtgs = policy.rtg(rews_buf,gam)
                    baseline = value.get_values(torch.as_tensor(states_buf,dtype=torch.float32))
                    advantage = (torch.as_tensor(rtgs,dtype=torch.float32) - baseline)
                    batch_weights.extend(advantage.tolist())
                    batch_targets.extend(rtgs)
                    """
                    # for GAE
                    advs,targets = GAE(
                        value,
                        torch.as_tensor(rews_buf,dtype=torch.float32),
                        torch.as_tensor(states_buf,dtype=torch.float32),
                        torch.as_tensor(next_state,dtype=torch.float32),
                        lam,
                        gam,
                        terminated
                    )
                    batch_weights.extend(advs)
                    batch_targets.extend(targets)
                    ###

                    batch_lens.append(len(rews_buf))  
                    break
                else:
                    state = next_state

        batch_avg_len = sum(batch_lens)/len(batch_lens)
        avg_len.append(batch_avg_len)

        batch_weights_tensor =  torch.as_tensor(batch_weights,dtype=torch.float32)
        batch_weights_normd = (batch_weights_tensor - batch_weights_tensor.mean())/(batch_weights_tensor.std() + 1e-8)
    
        surrogate_loss = policy.surrogate_loss(
            torch.as_tensor(batch_acts,dtype=torch.long),
            torch.as_tensor(batch_states,dtype=torch.float32),
            batch_weights_normd
            )
        
        optimizer_policy.zero_grad()
        surrogate_loss.backward()
        optimizer_policy.step()
        
        for epoch in range(epochs):
            evs_in_epochs = []
            value_loss_in_epochs = []
            value_loss,ev = value.loss_fn(
                torch.as_tensor(batch_states,dtype=torch.float32),
                torch.as_tensor(batch_targets,dtype=torch.float32)
                )
            evs_in_epochs.append(ev)
            value_loss_in_epochs.append(value_loss.item())

            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()
        
        avg_value_loss_in_epochs = sum(value_loss_in_epochs)/epochs
        avg_ev_in_epochs = sum(evs_in_epochs)/epochs
        batch_value_loss.append(avg_value_loss_in_epochs)
        ev_vals.append(avg_ev_in_epochs)

        if batch % 10 == 0:
            print(f"Batch {batch} avg len: {batch_avg_len}")
            print(f"Batch {batch} value loss: {avg_value_loss_in_epochs}")
            print(f"Batch {batch} ev: {avg_ev_in_epochs}")
            print("\n")


    env_render.close()
    env_headless.close()

    end_time = time.perf_counter()
    train_duration = end_time - start_time
    print(f"Training time (mins): {train_duration/60}")
    print(f"Training time (secs): {train_duration}")

    if serious_training_run:
        torch.save(policy.state_dict(),saved_policy_filename)
        torch.save(value.state_dict(),saved_value_filename)

    plt.plot([i for i in range(1,max_iters+1)], avg_len) 
    plt.xlabel("Batch")  
    plt.ylabel("Average Len")  
    plt.title("Episode Lengths")
    plt.savefig('batch_avg_len_gae.png')     
    plt.show()  

    plt.plot([i for i in range(1,max_iters+1)], batch_value_loss) 
    plt.xlabel("Batch")  
    plt.ylabel("Loss")  
    plt.title("Value Function Loss")
    plt.savefig('value_fn_loss_gae.png')     
    plt.show() 

    plt.plot([i for i in range(1,(max_iters+1))], ev_vals) 
    plt.xlabel("Batch")  
    plt.ylabel("EV")  
    plt.title("Explained Variance")
    plt.savefig('ev_gae.png')     
    plt.show() 