import sys
import time
import flappy_bird_gymnasium
import gymnasium
import torch
from DQN import DQN
from collections import deque
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

device = torch.device('cpu')

num_episodes = sys.maxsize
num_timesteps = 20000
batch_size =8
num_completed_steps = 0
all_episodes_return = deque(maxlen=100)
is_training = False

env = gymnasium.make("FlappyBird-v0", audio_on=False, pipe_gap=100)

action_size = env.action_space.n
state_size = env.observation_space.shape[0]
dqn = DQN(action_size=action_size, state_size=state_size, device=device, is_flappyBirdEnv=True, load_model=False)
def expand_state_dims(state):
    state = torch.tensor(state, dtype=torch.float32)
    state = torch.unsqueeze(state, dim=0)
    state = state.to(device)
    return state

for episode_number in range(num_episodes):

    if episode_number % dqn.model_save_rate == 0 and episode_number > 0 and is_training:
        dqn.save_model("C:/Users/Allan/Desktop/Models/FlappyBirdModels2/{0}.pt".format(episode_number))

    total_return = 0
    init_state, _ = env.reset()
    state = init_state
    state = expand_state_dims(state)
    for time_step in range(num_timesteps):
        num_completed_steps += 1
        if num_completed_steps % dqn.update_rate == 0:
            dqn.update_target_network()

        action = dqn.epsilon_greedy(state)

        next_state, reward, done, _, meta_data = env.step(action)
        env.render()
        next_state = expand_state_dims(next_state)
        dqn.store_transition(state, action, reward, next_state, done)

        state = next_state
        total_return += reward
        if done:
            print("Total reward for episode {1}: {0} || Epsilon = {2}".format(total_return, episode_number, dqn.get_epsilon()))
            all_episodes_return.append(total_return)

            break

        if len(dqn.replay_buffer) > batch_size:
            is_training = True
            dqn.train_double_DQN(batch_size=batch_size)
            pass

    pass
    dqn.decay_epsilon()

env.close()

