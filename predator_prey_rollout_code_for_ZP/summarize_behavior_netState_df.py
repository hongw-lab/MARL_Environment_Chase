import pickle
import os
import numpy as np
import pandas as pd

def ori_position(A_pos, A_orient, B_pos):
    # Map from orientation ID to coordinate transformations
    orientation_transform = {
        0: lambda x, y: (x, y),  # UP: No change
        1: lambda x, y: (y, -x),  # RIGHT: Rotate right (clockwise)
        2: lambda x, y: (-x, -y),  # DOWN: Rotate 180 degrees
        3: lambda x, y: (-y, x)  # LEFT: Rotate left (counterclockwise)
    }
    # Get the transformation function for A's orientation
    transform = orientation_transform[A_orient]
    # Calculate the difference in position from B to A
    delta_x = B_pos[0] - A_pos[0]
    delta_y = B_pos[1] - A_pos[1]
    # Apply the transformation to the delta
    relative_x, relative_y = transform(delta_x, delta_y)
    return (relative_x, relative_y)


# Now, we are going to linearly summarize the results into a pandas dataframe
if __name__ == '__main__':
    # video_path = f'{os.environ["HOME"]}/Documents/GitHub/meltingpot-2.2.0/examples/videos/open_field_1_1/'
    video_path = f'{os.environ["HOME"]}/Documents/GitHub/meltingpot-2.2.0/examples/videos/open_field_1_1/'
    # Now, we are going to summarize the reward, distances of move, action info, time till catch

    predator_ids = [0, 1, 2, 3, 4]
    prey_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for predator_id in predator_ids:
        for prey_id in prey_ids:
            info_dict = {}
            network_states_dict = {}
            serial_data_dict = {}

            print(f'Processing {predator_id}_{prey_id}')
            title = f'{predator_id}_{prey_id}'
            for eId in range(1, 101):
                with open(f'{video_path}{title}_{eId}.pkl', 'rb') as f:
                    results = pickle.load(f)
                if title not in serial_data_dict:
                    serial_data_dict[title] = {title: [] for title in ['STAMINA', 'POSITION', 'ORIENTATION', 'rewards', 'actions',
                                                                       'distances']}
                serial_data_dict[title]['STAMINA'].append([info['STAMINA'] for info in results])
                serial_data_dict[title]['POSITION'].append([info['POSITION'] for info in results])
                serial_data_dict[title]['ORIENTATION'].append([info['ORIENTATION'] for info in results])
                serial_data_dict[title]['rewards'].append([info['rewards'] for info in results])
                serial_data_dict[title]['actions'].append([info['actions'] for info in results])
                duration = len(results)
                if 'predator_id' not in info_dict:
                    info_dict['predator_id'] = []
                info_dict['predator_id'].extend([predator_id] * duration)
                if 'prey_id' not in info_dict:
                    info_dict['prey_id'] = []
                info_dict['prey_id'].extend([prey_id] * duration)
                if 'eId' not in info_dict:
                    info_dict['eId'] = []
                info_dict['eId'].extend([eId] * duration)

                for key in ['STAMINA', 'POSITION', 'ORIENTATION', 'rewards', 'actions']:
                    for agent_id in range(2):
                        key_id = key + f'_{agent_id}'
                        if key_id not in info_dict:
                            info_dict[key_id] = []
                        info_dict[key_id].extend([info[key][agent_id] for info in results])
                ACTION_MAP = {0: 'NOOP', 1: 'FORWARD', 2: 'BACKWARD', 3: 'STEP_LEFT', 4: 'STEP_RIGHT', 5: 'TURN_LEFT',
                              6: 'TURN_RIGHT', 7: 'INTERACT'}
                ORIENTATION_MAP = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT'}
                # Now encode the reletive position of the predator to the prey and the prey to the predator depending on their orientation
                positions = np.array([info['POSITION'] for info in results])
                position_0_1_x = positions[:, 0, 0] - positions[:, 1, 0]
                position_0_1_y = positions[:, 0, 1] - positions[:, 1, 1]
                position_1_0_x = -position_0_1_x
                position_1_0_y = -position_0_1_y
                if f'rel_position_0_1_x' not in info_dict:
                    info_dict[f'rel_position_0_1_x'] = []
                if f'rel_position_0_1_y' not in info_dict:
                    info_dict[f'rel_position_0_1_y'] = []
                if f'rel_position_1_0_x' not in info_dict:
                    info_dict[f'rel_position_1_0_x'] = []
                if f'rel_position_1_0_y' not in info_dict:
                    info_dict[f'rel_position_1_0_y'] = []
                info_dict[f'rel_position_0_1_x'].extend(position_0_1_x)
                info_dict[f'rel_position_0_1_y'].extend(position_0_1_y)
                info_dict[f'rel_position_1_0_x'].extend(position_1_0_x)
                info_dict[f'rel_position_1_0_y'].extend(position_1_0_y)
                # Based on the orientation, we determine the relative position
                orientations = np.array([info['ORIENTATION'] for info in results]).astype(int)
                for agent_id in range(2):
                    ori_positions = []
                    for i in range(len(orientations)):
                        opponent_id = 1 - agent_id
                        ori_positions.append(ori_position(positions[i, agent_id], orientations[i, agent_id], positions[i, opponent_id]))
                    ori_positions = np.array(ori_positions).astype(int)
                    for ii, xyname in enumerate(['x', 'y']):
                        if f'rel_position_{agent_id}_{xyname}' not in info_dict:
                            info_dict[f'rel_position_{agent_id}_{xyname}'] = []
                        info_dict[f'rel_position_{agent_id}_{xyname}'].extend(ori_positions[:, ii])
                        if f'ori_position_{agent_id}_{opponent_id}_{xyname}' not in info_dict:
                            info_dict[f'ori_position_{agent_id}_{opponent_id}_{xyname}'] = []
                        info_dict[f'ori_position_{agent_id}_{opponent_id}_{xyname}'].extend(ori_positions[:, ii])

                # Now encode orientation to one hot
                orientations_0 = np.zeros((orientations.shape[0], 4))
                orientations_1 = np.zeros((orientations.shape[0], 4))
                orientations_0[np.arange(orientations.shape[0]), orientations[:, 0]] = 1
                orientations_1[np.arange(orientations.shape[0]), orientations[:, 1]] = 1
                for agent_id, orientations in enumerate([orientations_0, orientations_1]):
                    for i in range(orientations.shape[1]):
                        if f'orientations_{agent_id}_{i}' not in info_dict:
                            info_dict[f'orientations_{agent_id}_{i}'] = []
                        info_dict[f'orientations_{agent_id}_{i}'].extend(orientations[:, i])

                # Now encode if the distances between the predator and the prey and if the prey is approaching or moving away
                # And if the predator is approaching or moving away
                distances = []
                distance_to_predator = np.linalg.norm(positions[1:, 0] - positions[:-1, 1], axis=1)
                distance_to_prey = np.linalg.norm(positions[1:, 1] - positions[:-1, 0], axis=1)
                distance_t0 = np.linalg.norm(positions[0, 0] - positions[0, 1])
                distance_to_predator = np.concatenate([[distance_t0], distance_to_predator])
                distance_to_prey = np.concatenate([[distance_t0], distance_to_prey])
                approaching_predator = [distance_to_predator[i] < distance_to_predator[i-1] for i in range(1, len(distance_to_predator))]
                approaching_prey = [distance_to_prey[i] < distance_to_prey[i-1] for i in range(1, len(distance_to_prey))]
                away_predator = [distance_to_predator[i] > distance_to_predator[i-1] for i in range(1, len(distance_to_predator))]
                away_prey = [distance_to_prey[i] > distance_to_prey[i-1] for i in range(1, len(distance_to_prey))]
                approaching_predator = np.concatenate([[False], approaching_predator])
                approaching_prey = np.concatenate([[False], approaching_prey])
                away_predator = np.concatenate([[False], away_predator])
                away_prey = np.concatenate([[False], away_prey])
                if f'distance_to_predator' not in info_dict:
                    info_dict[f'distance_to_predator'] = []
                    info_dict[f'approaching_predator'] = []
                    info_dict[f'away_predator'] = []
                if f'distance_to_prey' not in info_dict:
                    info_dict[f'distance_to_prey'] = []
                    info_dict[f'approaching_prey'] = []
                    info_dict[f'away_prey'] = []
                info_dict[f'distance_to_predator'].extend(distance_to_predator)
                info_dict[f'distance_to_prey'].extend(distance_to_prey)
                info_dict[f'approaching_predator'].extend(approaching_predator)
                info_dict[f'approaching_prey'].extend(approaching_prey)
                info_dict[f'away_predator'].extend(away_predator)
                info_dict[f'away_prey'].extend(away_prey)



                # for state_names in ['lstmMemory', 'lstmCell']:
                #   for agent_id in range(2):
                #     network_states = [info[state_names][agent_id] for info in results]
                #     network_states = np.array(network_states)
                #     for i in range(network_states.shape[1]):
                #       if f'{state_names}_{agent_id}_{i}' not in network_states_dict:
                #         network_states_dict[f'{state_names}_{agent_id}_{i}'] = []
                #       network_states_dict[f'{state_names}_{agent_id}_{i}'].extend(network_states[:, i])


                # Now, encode the actions to one hot
                actions = np.array([info['actions'] for info in results])
                actions_0 = np.zeros((actions.shape[0], 8))
                actions_1 = np.zeros((actions.shape[0], 8))
                actions_0[np.arange(actions.shape[0]), actions[:, 0]] = 1
                actions_1[np.arange(actions.shape[0]), actions[:, 1]] = 1
                for agent_id, tmp in enumerate([actions_0, actions_1]):
                    for i in range(8):
                        if f'actions_{agent_id}_{i}' not in info_dict:
                            info_dict[f'actions_{agent_id}_{i}'] = []
                        info_dict[f'actions_{agent_id}_{i}'].extend(tmp[:, i])



                # Now, calculate the distance moved by each agent in each episode
                distances = []
                positions = np.array([info['POSITION'] for info in results])
                num_agents = 2
                for i in range(num_agents):
                    position = positions[:, i]
                    distance = np.linalg.norm(position[1:] - position[:-1], axis=1)
                    distance = np.concatenate([[0], distance])
                    distances.append(distance)
                serial_data_dict[title]['distances'].append(distances)


                # Now, calculate the time till catch
                rewards = np.array([info['rewards'] for info in results])
                # Processing logic for safe area, catching, and apple collection
                safe_grass = [[i, j] for i in [8, 9, 10] for j in [4, 5, 6]]
                t_catch = np.where(rewards[:, 0] == 1)[0]
                t_apple = np.where(rewards[:, 1] == 1)[0]
                t_respawn = t_catch + 21
                t_respawn = np.insert(t_respawn, 0, 0)

                # Now fill the temporal info to the info_dict
                keys = ['alive', 'on_grass', 'off_grass', 'apple', 'catch']
                alive_vec = np.zeros(len(rewards))
                on_grass_vec = np.zeros(len(rewards))
                off_grass_vec = np.zeros(len(rewards))
                apple_vec = np.zeros(len(rewards))
                catch_vec = np.zeros(len(rewards))
                for t_r, t_c in zip(t_respawn[:-1], t_catch[1:]):
                    alive_vec[t_r:t_c] = 1
                    for ti in range(t_r, t_c):
                        if list(positions[ti, 1, :]) in safe_grass:
                            on_grass_vec[ti] = 1
                        else:
                            off_grass_vec[ti] = 1
                for ti in t_apple:
                    apple_vec[ti] = 1
                for ti in t_catch:
                    catch_vec[ti] = 1
                for key, vec in zip(keys, [alive_vec, on_grass_vec, off_grass_vec, apple_vec, catch_vec]):
                    if f'{key}' not in info_dict:
                        info_dict[f'{key}'] = []
                    info_dict[f'{key}'].extend(vec.tolist())

            # with open(f'{video_path}{title}_info.pkl', 'wb') as f:
            #   pickle.dump(info_dict, f)
            # with open(f'{video_path}{title}_network_states.pkl', 'wb') as f:
            #   pickle.dump(network_states_dict, f)
            info_df = pd.DataFrame(info_dict)
            # network_states_df = pd.DataFrame(network_states_dict)
            # serial_data_df = pd.DataFrame(serial_data_dict[title])
            info_df.to_csv(f'{video_path}{title}_info.csv', index=False)
            # network_states_df.to_csv(f'{video_path}{title}_network_states.csv', index=False)

