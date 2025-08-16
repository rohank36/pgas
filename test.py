from reinforce import ReinforcePolicy
import gymnasium as gym
import torch

env = gym.make('CartPole-v1', render_mode="human")
in_dim = env.observation_space.shape[0] # (4,)
out_dim = env.action_space.n # Discrete action space, (2,) possible actions for CartPole env
hidden_dim = 32
policy = ReinforcePolicy(in_dim,hidden_dim,out_dim)
policy.load_state_dict(torch.load("policy.pth",weights_only=True))

state,info = env.reset()
done = False

while True:
    action = policy.get_action(torch.as_tensor(state,dtype=torch.float32))
    next_state,reward,terminated,truncated,_ = env.step(action)

    if terminated: print("Terminated")
    elif truncated: print("Truncated")
    
    if terminated or truncated:
        break
    else:
        state = next_state