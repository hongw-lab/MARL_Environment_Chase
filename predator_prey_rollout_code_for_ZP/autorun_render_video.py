import argparse
import os
import itertools
from render_video_predator_prey import run_rollout

# Function to create the bot reference for a combination
def create_bot_reference(predator_type, prey_type, num_predator_models=5, num_prey_models=8, num_predators=5, num_preys=8,  **kwargs):
    return [predator_type] * num_predators + [prey_type + num_predator_models] * num_preys

# Function to generate a name for a combination
def generate_video_name(predator_type, prey_type, num_predators=5, num_preys=17,  **kwargs):
    return ''.join([str(predator_type)] * num_predators) + '_' + ''.join([str(prey_type)] * num_preys)

def run_all_combinations(output_path, num_episodes=20, model_path=None, **kwargs):
    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # List all possible combinations of predator and prey by checking the model_path
    model_directories = os.listdir(model_path)
    model_directories.sort()
    predator_types = [int(d.split('_')[-1]) for d in model_directories if 'predator' in d]
    prey_types = [int(d.split('_')[-1]) for d in model_directories if 'prey' in d]



    for predator_type, prey_type in itertools.product(predator_types, prey_types):
        if (kwargs['predator_id'] is not None) and (predator_type != kwargs['predator_id']):
            continue
        if (kwargs['prey_id'] is not None) and (prey_type != kwargs['prey_id']):
            continue
        bot_reference = create_bot_reference(predator_type, prey_type, len(predator_types), len(prey_types), **kwargs)
        video_name = generate_video_name(predator_type, prey_type, **kwargs)

        print(f'Running combination: Predator {predator_type}, Prey {prey_type}...')
        run_rollout(bot_reference, output_path, video_name, num_episodes, num_predator_models=len(predator_types), num_prey_models=len(prey_types), model_path=model_path, **kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run all predator-prey combinations in MeltingPot.")
    parser.add_argument('--output-path', type=str, default='/home/lime/Documents/GitHub/meltingpot-2.2.0/examples/videos/open_field_1_1_acorn/',
                        help='Output directory for videos and information files.')
    parser.add_argument('--model-path', type=str, default='/home/lime/Documents/GitHub/meltingpot-2.2.0/meltingpot/assets/saved_models/predator_prey_general/')
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of episodes to run per combination.')
    parser.add_argument('--num-predators', type=int, default=2, help='Number of predators in the environment.')
    parser.add_argument('--num-preys', type=int, default=3, help='Number of preys in the environment.')
    parser.add_argument('--predator-id', type=int, default=2, help='ID of the predator model to run.')
    parser.add_argument('--prey-id', type=int, default=1, help='ID of the prey model to run.')
    parser.add_argument('--substrate-name', type=str, default='predator_prey__simplified10x10_1v1', help='Environment to run.')

    args = parser.parse_args()
    run_all_combinations(**vars(args))
