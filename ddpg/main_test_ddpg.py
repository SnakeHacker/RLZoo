import torch
import gym
import torch.nn as nn
import time
import pygame
import numpy as np
import os
import glob
import os


env = gym.make('Pendulum-v1', render_mode="rgb_array")
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]


# Load para
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
actor_path = model+"actor_ddpg_actor_20250612-224847.pth"


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tan(self.fc3(x))*2
        return x

def process_frame(frame):
    frame = np.transpose(frame, (1,0,2))
    frame = pygame.surfarray.make_surface(frame)
    return pygame.transform.scale(frame, (width,height))

actor = Actor(STATE_DIM, ACTION_DIM)
actor.load_state_dict(torch.load(actor_path))


# Initialize pygame
pygame.init()
width,height = 600, 600
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Test phase

NUM_EPISODE=30
NUM_STEP=200

for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    episode_reward = 0


    for step_i in range(NUM_STEP):
        action = actor(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()[0]
        next_state, reward, done, truncation, info = env.step(action)
        state = next_state
        episode_reward += reward
        print(f"Step {step_i}: ", action)

        frame = env.render()
        frame = process_frame(frame)
        screen.blit(frame, (0,0))
        pygame.display.flip()
        clock.tick(60) # FPS

    print(f"Episode {episode_i} Reward: {episode_reward}")

pygame.quit()
env.close()
