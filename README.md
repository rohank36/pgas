# Policy Gradient Algorithms

Implemented
- reinforce 
- ppo
- rtg and gae for weights

<table>
  <tr>
    <td><img src="ppo_gae/batch_avg_len_ppo_gae.png" alt="Batch Avg Length (PPO + GAE)" width="420"></td>
    <td><img src="ppo_gae/value_fn_loss_ppo_gae.png" alt="Value Function Loss (Huber, PPO + GAE)" width="420"></td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <em>CartPole (PPO + GAE): episodes approach the 500 step cap (max reward)</em>
    </td>
  </tr>
</table>


To test:
```
>>> git clone https://github.com/rohank36/pgas.git
# have env with Gymnasium and PyTorch activated
# choose which policy to test in test.py
>>> python test.py
```