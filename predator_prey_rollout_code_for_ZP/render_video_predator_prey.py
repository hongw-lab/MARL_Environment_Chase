import argparse
import os
import cv2
import dm_env
import numpy as np
import pickle
import sys
sys.path.insert(0, os.path.dirname('/examples/rllib/'))
import utils ## examples.rllib.utils
from meltingpot.utils.policies.saved_model_policy import TF2SavedModelPolicy
import time
import concurrent.futures
from meltingpot import substrate


# Function to handle the bot step
def bot_step(bot, timestep_bot, state):
  return bot.step(timestep_bot, state)

# Function to write video frames to a file in the background
def write_video(frames, video_path, fps, size):
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


def is_file_larger_than_4_5mb(filepath):
  file_size = os.path.getsize(filepath)  # Get the file size in bytes
  return file_size > 4.5 * 1024 * 1024  # 4.5 MB in bytes



def run_rollout(bot_reference, output_path, video_name, num_episodes=20,
                substrate_name='predator_prey__simplified10x10_1v1',
                model_path='/home/lime/Documents/GitHub/meltingpot-2.2.0/meltingpot/assets/saved_models/predator_prey_general/',
                num_predators=None, num_preys=None, num_predator_models=None, num_prey_models=None,
                **kwargs):
  # Ensure the output directory exists
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  # player_roles = ['predator', 'prey']
  if (num_predators is None) and (num_preys is None):
    player_roles = substrate.get_config(substrate_name).default_player_roles
  else:
    player_roles = ['predator'] * num_predators + ['prey'] * num_preys
  env_config = {"substrate": substrate_name, "roles": player_roles}

  if num_predator_models is None:
    # List all possible combinations of predator and prey by checking the model_path
    model_directories = sorted(os.listdir(model_path))
    num_predator_models = [int(d.split('_')[-1]) for d in model_directories if 'predator' in d]
    num_prey_models = [int(d.split('_')[-1]) for d in model_directories if 'prey' in d]

  model_directories = sorted(os.listdir(model_path))
  predator_model_names = [d for d in model_directories if 'predator' in d]
  prey_model_names = [d for d in model_directories if 'prey' in d]

  bots = [TF2SavedModelPolicy(model_path + name, device_name='cpu') for name in predator_model_names] + \
         [TF2SavedModelPolicy(model_path + name, device_name='cpu') for name in prey_model_names]

  env = utils.env_creator(env_config).get_dmlab2d_env()

  fps = 10
  shape = env.observation_spec()[0]["WORLD.RGB"].shape
  size = (shape[1], shape[0])

  for episode in range(num_episodes):
    print(f'Running episode {episode + 1}/{num_episodes}...')
    if os.path.exists(os.path.join(output_path, f"{video_name}_{episode + 1}.pkl")):
      print(f"Skipping episode {episode + 1} as it already exists.")
      continue
    video_writer = cv2.VideoWriter(os.path.join(output_path, f"{video_name}_{episode + 1}.avi"),
                                   cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

    start_time = time.time()
    infos = []

    # reset the environment and bots
    timestep = env.reset()
    states = [bots[i].initial_state() for i in bot_reference]
    # actions_cat = []
    for time_t in range(1000):
      # if time_t % 10 == 0:
      #     print(f'Frame {time_t}, {time.time() - start_time:.2f} seconds passed.')

      obs = timestep.observation[0]["WORLD.RGB"]
      video_writer.write(obs)

      timestep_bots = [dm_env.TimeStep(
        step_type=timestep.step_type,
        reward=timestep.reward[i],
        discount=timestep.discount,
        observation={k: timestep.observation[i][k] for k in ["RGB", "STAMINA", "POSITION"]}
      ) for i in range(len(bot_reference))]

      # Process bot steps using parallel execution
      with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(bot_step, bots[bot_reference[i]], timestep_bots[i], states[i])
                   for i in range(len(bot_reference))]
        results = [f.result() for f in futures]
        actions, states = zip(*results)
        # actions_cat.append(actions)

      # observations = [timestep_bot.observation for timestep_bot in timestep_bots]
      # RGB = [timestep_bot.observation["RGB"] for timestep_bot in timestep_bots]
      STAMINA = [timestep_bot.observation["STAMINA"] for timestep_bot in timestep_bots]
      POSITION = [timestep.observation[i]["POSITION"] for i in range(len(bot_reference))]
      ORIENTATION = [timestep.observation[i]["ORIENTATION"] for i in range(len(bot_reference))]

      rewards = [timestep_bot.reward for timestep_bot in timestep_bots]
      discounts = [timestep_bot.discount for timestep_bot in timestep_bots]
      # actions_cat = [action.numpy() for action in actions_cat]
      actions = [action.numpy() for action in actions]
      partitionCalls = [state[0].numpy() for state in states]
      lstmMemory = []
      lstmCell = []
      for state in states:
        if isinstance(state[1], dict):
          lstmMemory.append(state[1]['lstm_state'].hidden)
          lstmCell.append(state[1]['lstm_state'].cell)
          #
        elif hasattr(state[1], 'memory'):
          # lstmMemory = state[1][0]
          # lstmCell = state[1][1]
          lstmMemory.append(state[1].memory.hidden)
          lstmCell.append(state[1].memory.cell)
        else:
          lstmMemory.append(state[0])
          lstmCell.append(state[1])
      infos.append({
        # 'RGB': np.array(RGB).astype(np.uint8),
        'STAMINA': np.array(STAMINA).astype(np.float32),
        'POSITION': np.array(POSITION).astype(int),
        'ORIENTATION': np.array(ORIENTATION).astype(int),
        'rewards': np.array(rewards),
        'discounts': np.array(discounts),
        # 'actions_cat': np.array(actions_cat).astype(int),
        'actions': np.array(actions).astype(int),
        'partitionCalls': np.array(partitionCalls),
        'lstmMemory': np.array(lstmMemory),
        'lstmCell': np.array(lstmCell)
      })

      timestep = env.step(actions)

    video_writer.release()

    # Save the information dict as a pickle file
    with open(os.path.join(output_path, f"{video_name}_{episode + 1}.pkl"), 'wb') as f:
      pickle.dump(infos, f)

    # At the end of the episode, print the time taken and indicate the completion
    print(f'Episode {episode + 1}/{num_episodes} completed in {time.time() - start_time:.2f} seconds.')
    # Plot cumulative reward curve
    # rewards = np.array([info['rewards'] for info in infos])
    # cumsum_rewards = np.cumsum(rewards, axis=0)
    # plt.figure()
    # plt.plot(cumsum_rewards)
    # plt.title(f'Cumulative Rewards for {video_name} - Episode {episode + 1}')
    # plt.xlabel('Time Step')
    # plt.ylabel('Cumulative Reward')
    # plt.savefig(os.path.join(output_path, f"{video_name}_{episode + 1}_reward_curve.png"))
    # plt.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run rollouts for different agent combinations in MeltingPot.")
  parser.add_argument('--bot_reference', type=int, nargs='+', default=[0,5],
                      help='Indices of bots to use.')
  parser.add_argument('--output_path', type=str, default='/home/lime/Documents/GitHub/meltingpot-2.2.0/examples/rllib/videos/',
                      help='Output directory for videos and information files.')
  parser.add_argument('--video_name', type=str, default='alley_hunt_combination',
                      help='Base name for output videos and files.')
  parser.add_argument('--num_episodes', type=int, default=20, help='Number of episodes to run.')
  parser.add_argument('--substrate-name', type=str, default='predator_prey__alley_hunt_simplified10x10_1v1',)
  args = parser.parse_args()
  run_rollout(args.bot_reference, args.output_path, args.video_name, args.num_episodes)
