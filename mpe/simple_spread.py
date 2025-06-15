from mpe2 import simple_spread_v3
env = simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=100, render_mode='human')

env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print(f"Agent: {agent}, Observation: {observation}, Reward: {reward}")

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy
        print(f"Action: {action}")

    env.step(action)

env.close()