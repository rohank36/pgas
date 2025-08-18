#from reinforce import ReinforcePolicy
from reinforce_adv import ReinforcePolicy
import gymnasium as gym
import torch

#policy_filepath = "reinforce/reinforce_policy_rtg.pth"
policy_filepath = "reinforce_gae/reinforce_gae_policy.pth"

env = gym.make('CartPole-v1', render_mode="human")

in_dim = env.observation_space.shape[0] # (4,)
out_dim = env.action_space.n # Discrete action space, (2,) possible actions for CartPole env
hidden_dim = 32
policy = ReinforcePolicy(in_dim,hidden_dim,out_dim)

policy.load_state_dict(torch.load(policy_filepath,weights_only=True))

state,info = env.reset()
done = False
step_counter = 0

while True:
    action = policy.get_action(torch.as_tensor(state,dtype=torch.float32))
    next_state,reward,terminated,truncated,_ = env.step(action)
    step_counter += 1

    if terminated: print("Terminated")
    elif truncated: print("Truncated")
    
    if terminated or truncated:
        print(f"Steps: {step_counter}")
        break
    else:
        state = next_state