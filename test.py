from reinforce import ReinforcePolicy
import gymnasium as gym
import torch

env = gym.make('CartPole-v1', render_mode="human")
policy = ReinforcePolicy()
policy.load_state_dict(torch.load("policy.pth",weights_only=True))

state,info = env.reset()
done = False

while True:
    action = policy.get_action(torch.as_tensor(state,dtype=torch.float32))
    next_state,reward,terminated,truncated,_ = env.step(action)
    
    if terminated or truncated:
        break
    else:
        state = next_state