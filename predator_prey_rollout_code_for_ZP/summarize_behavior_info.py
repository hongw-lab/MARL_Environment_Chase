import pickle
import os
import numpy as np

if __name__ == '__main__':
    video_path = f'{os.environ["HOME"]}/Documents/GitHub/meltingpot-2.2.0/examples/videos/open_field_1_1/'
    # Now, we are going to summarize the reward, distances of move, action info, time till catch
    predator_ids = [0, 1, 2, 3, 4]
    prey_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    serial_data_dict = {}
    cumulative_dict = {}
    for predator_id in predator_ids:
        for prey_id in prey_ids:
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

                if title not in cumulative_dict:
                    summary_titles = ['round', 'time_per_round', 'prey_move_distances_per_round',
                                      'predator_move_distances_per_round', 'num_apple_collected_per_round',
                                      'prey_rotate_per_round', 'predator_rotate_per_round',
                                      'time_on_grass_per_round', 'time_off_grass_per_round',
                                      'frac_off_grass_per_round',
                                      'frac_moving_away_per_round', 'percent_time_in_3_steps',
                                      'percent_time_in_5_steps']
                    cumulative_dict[title] = {title: [] for title in summary_titles}

                for round, t_start in enumerate(t_respawn[:-1]):
                    t_end = np.min([t_respawn[round + 1], len(rewards)])
                    t_leave_safe = np.nan
                    for ti in range(t_start, t_end):
                        if list(positions[ti, 1, :]) not in safe_grass:
                            t_leave_safe = ti
                            break

                    if np.isnan(t_leave_safe):
                        continue

                    t_catch_i = t_catch[t_catch >= t_leave_safe]
                    t_catch_i = t_catch_i[0] if len(t_catch_i) > 0 else np.nan
                    time_per_round = t_catch_i - t_leave_safe
                    t_apple_i = t_apple[(t_leave_safe <= t_apple) & (t_apple < t_end)]
                    num_apple = len(t_apple_i)

                    cumulative_dict[title]['round'].append(f'{eId}_round{round}')
                    cumulative_dict[title]['time_per_round'].append(time_per_round)
                    cumulative_dict[title]['num_apple_collected_per_round'].append(num_apple)

                    # First, get total distances of move
                    total_distances = np.sum(np.abs(distances)==1, axis=1)
                    cumulative_dict[title]['total_distances'] = total_distances
                    prey_move_distances_per_round = np.sum(distances[1][t_start:t_end])
                    predator_move_distances_per_round = np.sum(distances[0][t_start:t_end])
                    cumulative_dict[title]['prey_move_distances_per_round'].append(prey_move_distances_per_round)
                    cumulative_dict[title]['predator_move_distances_per_round'].append(predator_move_distances_per_round)

                    # second, get the rotation, which is change in orientation
                    orientations = np.array([info['ORIENTATION'] for info in results])
                    total_rotations = np.sum(np.abs(orientations[1:] - orientations[:-1]), axis=0)
                    cumulative_dict[title]['total_rotations'] = total_rotations
                    prey_rotate_per_round = np.sum(np.abs(orientations[1][t_start:t_end] - orientations[1][t_start-1:t_end-1]))
                    predator_rotate_per_round = np.sum(np.abs(orientations[0][t_start:t_end] - orientations[0][t_start-1:t_end-1]))
                    cumulative_dict[title]['prey_rotate_per_round'].append(prey_rotate_per_round)
                    cumulative_dict[title]['predator_rotate_per_round'].append(predator_rotate_per_round)

                    ## Now, check the time on grass per round and the time off the grass per round
                    time_off_grass_per_round = 0
                    time_on_grass_per_round = 0
                    if not np.isfinite(t_catch_i):
                        t_catch_i = len(rewards)
                    for ti in range(t_start, t_catch_i):
                        if list(positions[ti, 1, :]) not in safe_grass:
                            time_off_grass_per_round += 1
                        else:
                            time_on_grass_per_round += 1
                    if time_on_grass_per_round < 0:
                        raise ValueError('Time on grass is negative')
                    cumulative_dict[title]['time_on_grass_per_round'].append(time_on_grass_per_round)
                    cumulative_dict[title]['time_off_grass_per_round'].append(time_off_grass_per_round)
                    cumulative_dict[title]['frac_off_grass_per_round'].append(time_off_grass_per_round / (time_on_grass_per_round + time_off_grass_per_round))

                    # Now check for each time step in this round if the prey's position change cause a longer distance with the predator
                    if t_leave_safe > 0 and t_catch_i < len(rewards):
                        distance_to_predator = np.linalg.norm(positions[t_leave_safe-1:t_catch_i-1, 0] - positions[t_leave_safe:t_catch_i, 1], axis=1)
                        t_moved = [i - t_leave_safe for i in range(t_leave_safe, t_catch_i) if (positions[i, 1] != positions[i-1, 1]).any()]
                        distance_to_predator = distance_to_predator[t_moved]
                        t_moving_away = np.sum([distance_to_predator[i] > distance_to_predator[i-1] for i in range(1, len(distance_to_predator))])
                        frac_moving_away = t_moving_away / len(t_moved)
                        cumulative_dict[title]['frac_moving_away_per_round'] = frac_moving_away
                        manhattan_distance = np.sum(np.abs(positions[t_leave_safe:t_catch_i, 0] - positions[t_leave_safe:t_catch_i, 1]), axis=1)
                        percent_time_in_3_steps = np.sum(manhattan_distance < 3) / len(manhattan_distance)
                        cumulative_dict[title]['percent_time_in_3_steps'] = percent_time_in_3_steps
                        percent_time_in_5_steps = np.sum(manhattan_distance < 5) / len(manhattan_distance)
                        cumulative_dict[title]['percent_time_in_5_steps'] = percent_time_in_5_steps

                cumulative_dict[title]['mean_time_per_round'] = np.nanmean(cumulative_dict[title]['time_per_round'])
                cumulative_dict[title]['mean_apple_per_round'] = np.nanmean(cumulative_dict[title]['num_apple_collected_per_round'])
                cumulative_dict[title]['mean_prey_move_distances_per_round'] = np.nanmean(cumulative_dict[title]['prey_move_distances_per_round'])
                cumulative_dict[title]['mean_predator_move_distances_per_round'] = np.nanmean(cumulative_dict[title]['predator_move_distances_per_round'])
                cumulative_dict[title]['mean_prey_rotate_per_round'] = np.nanmean(cumulative_dict[title]['prey_rotate_per_round'])
                cumulative_dict[title]['mean_predator_rotate_per_round'] = np.nanmean(cumulative_dict[title]['predator_rotate_per_round'])

    with open(f'{video_path}serial_results_dict.pkl', 'wb') as f:
        pickle.dump(serial_data_dict, f)
    with open(f'{video_path}cumulative_results_dict.pkl', 'wb') as f:
        pickle.dump(cumulative_dict, f)
