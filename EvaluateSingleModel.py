import sys
import time
import numpy as np
import torch
import os
import gymnasium
from Q_Network import Q_Network
import flappy_bird_gymnasium

models_directory = "./models"
env = gymnasium.make("FlappyBird-v0", audio_on=False, pipe_gap=100)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
action_size = env.action_space.n
state_size = env.observation_space.shape[0]
episodes_to_test = 100
max_time_steps = sys.maxsize

def preprocess_state(state):
    state = torch.tensor(state, dtype=torch.float32)
    state = torch.unsqueeze(state, dim=0)
    state = state.to(device)
    return state

def get_action(model, state):
    model.eval()
    with torch.no_grad():
        Q_values = int(torch.argmax(model(state)[0]))
        pass
    return Q_values

model_path = os.path.join(models_directory, "2200_backup_flappy_bird.pt")
model = Q_Network(device=device, action_space_size=action_size, observation_space_size=state_size).to(device)
model.load_state_dict(torch.load(model_path))
for episode in range(episodes_to_test):
    observation, _ = env.reset()
    observation = preprocess_state(observation)
    episode_reward = 0
    for time_step in range(max_time_steps):
        time.sleep(1/30)
        action = get_action(model, observation)
        next_observation, reward, done, truncated, meta_data = env.step(action)
        next_observation = preprocess_state(next_observation)
        episode_reward += reward
        observation = next_observation
        env.render()
        if done:
            print("Episode reward = {0}".format(episode_reward))
            break
        pass
    pass
print("Average reward for model, {0} = {1}".format(model_path, np.mean(model_episode_rewards)))
