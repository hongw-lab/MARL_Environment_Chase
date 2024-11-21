import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap
import os
HOME = os.environ['HOME']
if __name__ == "__main__":
    naive_predator_ids = list(range(5))
    naive_prey_ids = list(range(5))
    trained_predator_ids = list(range(5))
    trained_prey_ids = [3, 4, 14, 15, 16]
    naive_path = f'{HOME}/Documents/GitHub/meltingpot-2.2.0/examples/videos/open_field_1_1_single_acorn_loc_naive/'
    trained_path = f'{HOME}/Documents/GitHub/meltingpot-2.2.0/examples/videos/open_field_1_1_single_acorn_loc/'
    tpnp_path = f'{HOME}/Documents/GitHub/meltingpot-2.2.0/examples/videos/open_field_1_1_single_acorn_loc_trained_vs_naive/'
    naive_results = {}
    trained_results = {}
    naive_plsc_results = {}

    with open(f'{naive_path}serial_results_dict.pkl', 'rb') as f:
        naive_results_tmp = pickle.load(f)
    with open(f'{trained_path}serial_results_dict.pkl', 'rb') as f:
        trained_results_tmp = pickle.load(f)
    with open(f'{tpnp_path}serial_results_dict.pkl', 'rb') as f:
        tpnp_results_tmp = pickle.load(f)
    with open(f'{naive_path}PLSC_results_dict.pkl', 'rb') as f:
        naive_plsc_results = pickle.load(f)
    with open(f'{trained_path}PLSC_results_dict.pkl', 'rb') as f:
        trained_plsc_results = pickle.load(f)
    with open(f'{tpnp_path}PLSC_results_dict.pkl', 'rb') as f:
        tpnp_plsc_results = pickle.load(f)
    with open(f'{naive_path}PLSC_results_cross_rollout_dict_NP.pkl', 'rb') as f:
        naive_plsc_cross_results = pickle.load(f)
    with open(f'{trained_path}PLSC_results_cross_rollout_dict_NP.pkl', 'rb') as f:
        trained_plsc_cross_results = pickle.load(f)
    with open(f'{tpnp_path}PLSC_results_cross_rollout_dict_NP.pkl', 'rb') as f:
        tpnp_plsc_cross_results = pickle.load(f)


    naive_apple, naive_acorn, naive_catch = [], [], []
    trained_apple, trained_acorn, trained_catch = [], [], []
    tpnp_apple, tpnp_acorn, tpnp_catch = [], [], []
    for predator_id in naive_predator_ids:
        for prey_id in naive_prey_ids:
            title = f'{predator_id}_{prey_id}'
            apple_mean = (np.array(naive_results_tmp[title]['rewards'])[:,:,1] == 1).mean()
            acorn_mean = (np.array(naive_results_tmp[title]['rewards'])[:,:,1] > 1).mean()
            catch_mean = (np.array(naive_results_tmp[title]['rewards'])[:,:,0] == 1).mean()
            naive_apple.append(apple_mean)
            naive_acorn.append(acorn_mean)
            naive_catch.append(catch_mean)
    naive_apple = np.array(naive_apple)
    naive_acorn = np.array(naive_acorn)
    naive_catch = np.array(naive_catch)

    for predator_id in trained_predator_ids:
        for prey_id in trained_prey_ids:
            title = f'{predator_id}_{prey_id}'
            apple_mean = (np.array(trained_results_tmp[title]['rewards'])[:,:,1] == 1).mean()
            acorn_mean = (np.array(trained_results_tmp[title]['rewards'])[:,:,1] > 1).mean()
            catch_mean = (np.array(trained_results_tmp[title]['rewards'])[:,:,0] == 1).mean()
            trained_apple.append(apple_mean)
            trained_acorn.append(acorn_mean)
            trained_catch.append(catch_mean)
    trained_apple = np.array(trained_apple)
    trained_acorn = np.array(trained_acorn)
    trained_catch = np.array(trained_catch)

    for predator_id in naive_predator_ids:
        for prey_id in naive_prey_ids:
            title = f'{predator_id}_{prey_id}'
            apple_mean = (np.array(tpnp_results_tmp[title]['rewards'])[:,:,1] == 1).mean()
            acorn_mean = (np.array(tpnp_results_tmp[title]['rewards'])[:,:,1] > 1).mean()
            catch_mean = (np.array(tpnp_results_tmp[title]['rewards'])[:,:,0] == 1).mean()
            tpnp_apple.append(apple_mean)
            tpnp_acorn.append(acorn_mean)
            tpnp_catch.append(catch_mean)

    naive_plsc = []
    trained_plsc = []
    tpnp_plsc = []
    naive_plsc1 = []
    trained_plsc1 = []
    tpnp_plsc1 = []
    naive_plsc1_perm = []
    trained_plsc1_perm = []
    tpnp_plsc1_perm = []
    for predator_id in naive_predator_ids:
        for prey_id in naive_prey_ids:
            title = f'{predator_id}_{prey_id}'
            naive_plsc.append(naive_plsc_results[title]['rank'])
            naive_plsc1.append(naive_plsc_results[title]['cor'])
            naive_plsc1_perm.append(naive_plsc_results[title]['cor_perm_array'])
    for predator_id in trained_predator_ids:
        for prey_id in trained_prey_ids:
            title = f'{predator_id}_{prey_id}'
            trained_plsc.append(trained_plsc_results[title]['rank'])
            trained_plsc1.append(trained_plsc_results[title]['cor'])
            trained_plsc1_perm.append(trained_plsc_results[title]['cor_perm_array'])
    for predator_id in naive_predator_ids:
        for prey_id in naive_prey_ids:
            title = f'{predator_id}_{prey_id}'
            tpnp_plsc.append(tpnp_plsc_results[title]['rank'])
            tpnp_plsc1.append(tpnp_plsc_results[title]['cor'])
            tpnp_plsc1_perm.append(tpnp_plsc_results[title]['cor_perm_array'])
    naive_plsc = np.nanmean(naive_plsc, axis=1)
    trained_plsc = np.nanmean(trained_plsc, axis=1)
    tpnp_plsc = np.nanmean(tpnp_plsc, axis=1)
    naive_plsc1 = np.nanmean(np.array(naive_plsc1)[:,:,0], axis=1)
    trained_plsc1 = np.nanmean(np.array(trained_plsc1)[:,:,0], axis=1)
    tpnp_plsc1 = np.nanmean(np.array(tpnp_plsc1)[:,:,0], axis=1)
    naive_plsc1_perm = np.nanmean(np.array(naive_plsc1_perm)[:,:,:,0], axis=(1,2))
    trained_plsc1_perm = np.nanmean(np.array(trained_plsc1_perm)[:,:,:,0], axis=(1,2))
    tpnp_plsc1_perm = np.nanmean(np.array(tpnp_plsc1_perm)[:,:,:,0], axis=(1,2))
    delta_trained_plsc1 = trained_plsc1 - trained_plsc1_perm
    delta_naive_plsc1 = naive_plsc1 - naive_plsc1_perm
    delta_tpnp_plsc1 = tpnp_plsc1 - tpnp_plsc1_perm


    naive_plsc_cross = []
    trained_plsc_cross = []
    naive_plsc1_cross = []
    trained_plsc1_cross = []
    naive_plsc1_perm_cross = []
    trained_plsc1_perm_cross = []
    naive_plsc_cross_title_mean = []
    trained_plsc_cross_title_mean = []
    naive_plsc1_cross_title_mean = []
    trained_plsc1_cross_title_mean = []
    naive_plsc1_perm_cross_title_mean = []
    trained_plsc1_perm_cross_title_mean = []

    tpnp_plsc_cross = []
    tpnp_plsc1_cross = []
    tpnp_plsc1_perm_cross = []
    tpnp_plsc_cross_title_mean = []
    tpnp_plsc1_cross_title_mean = []
    tpnp_plsc1_perm_cross_title_mean = []

    title_used = []
    for predator_id in naive_predator_ids:
        for prey_id in naive_prey_ids:
            title = f'{predator_id}_{prey_id}'
            all_titles = [key for key in naive_plsc_cross_results.keys() if title in key]
            tmp1, tmp2, tmp3 = [], [], []
            for second_title in all_titles:
                if second_title not in title_used:
                    title_used = title_used + [second_title]
                    naive_plsc_cross.append(naive_plsc_cross_results[second_title]['rank'])
                    naive_plsc1_cross.append(naive_plsc_cross_results[second_title]['cor'])
                    naive_plsc1_perm_cross.append(naive_plsc_cross_results[second_title]['cor_perm_median'])

                tmp1.append(naive_plsc_cross_results[second_title]['rank'])
                tmp2.append(naive_plsc_cross_results[second_title]['cor'])
                tmp3.append(naive_plsc_cross_results[second_title]['cor_perm_median'])
            naive_plsc_cross_title_mean.append(np.nanmean(tmp1))
            naive_plsc1_cross_title_mean.append(np.nanmean(np.array(tmp2)[:,:,0]))
            naive_plsc1_perm_cross_title_mean.append(np.nanmean(np.array(tmp3)[:,:,0]))

    title_used = []
    for predator_id in trained_predator_ids:
        for prey_id in trained_prey_ids:
            title = f'{predator_id}_{prey_id}'
            all_titles = [key for key in trained_plsc_cross_results.keys() if title in key]
            tmp1, tmp2, tmp3 = [], [], []
            for second_title in all_titles:
                if second_title not in title_used:
                    title_used = title_used + [second_title]
                    trained_plsc_cross.append(trained_plsc_cross_results[second_title]['rank'])
                    trained_plsc1_cross.append(trained_plsc_cross_results[second_title]['cor'])
                    trained_plsc1_perm_cross.append(trained_plsc_cross_results[second_title]['cor_perm_median'])
                tmp1.append(trained_plsc_cross_results[second_title]['rank'])
                tmp2.append(trained_plsc_cross_results[second_title]['cor'])
                tmp3.append(trained_plsc_cross_results[second_title]['cor_perm_median'])
            trained_plsc_cross_title_mean.append(np.nanmean(tmp1))
            trained_plsc1_cross_title_mean.append(np.nanmean(np.array(tmp2)[:,:,0]))
            trained_plsc1_perm_cross_title_mean.append(np.nanmean(np.array(tmp3)[:,:,0]))

    title_used = []
    for predator_id in naive_predator_ids:
        for prey_id in naive_prey_ids:
            title = f'{predator_id}_{prey_id}'
            all_titles = [key for key in tpnp_plsc_cross_results.keys() if title in key]
            tmp1, tmp2, tmp3 = [], [], []
            for second_title in all_titles:
                if second_title not in title_used:
                    title_used = title_used + [second_title]
                    tpnp_plsc_cross.append(tpnp_plsc_cross_results[second_title]['rank'])
                    tpnp_plsc1_cross.append(tpnp_plsc_cross_results[second_title]['cor'])
                    tpnp_plsc1_perm_cross.append(tpnp_plsc_cross_results[second_title]['cor_perm_median'])
                tmp1.append(tpnp_plsc_cross_results[second_title]['rank'])
                tmp2.append(tpnp_plsc_cross_results[second_title]['cor'])
                tmp3.append(tpnp_plsc_cross_results[second_title]['cor_perm_median'])
            tpnp_plsc_cross_title_mean.append(np.nanmean(tmp1))
            tpnp_plsc1_cross_title_mean.append(np.nanmean(np.array(tmp2)[:,:,0]))
            tpnp_plsc1_perm_cross_title_mean.append(np.nanmean(np.array(tmp3)[:,:,0]))

    naive_plsc_cross = np.mean(naive_plsc_cross, axis=1)
    trained_plsc_cross = np.mean(trained_plsc_cross, axis=1)
    naive_plsc1_cross = np.mean(np.array(naive_plsc1_cross)[:,:,0], axis=1)
    trained_plsc1_cross = np.mean(np.array(trained_plsc1_cross)[:,:,0], axis=1)
    naive_plsc1_perm_cross = np.mean(np.array(naive_plsc1_perm_cross)[:,:,0], axis=1)
    trained_plsc1_perm_cross = np.mean(np.array(trained_plsc1_perm_cross)[:,:,0], axis=1)
    delta_trained_plsc1_cross = trained_plsc1_cross - trained_plsc1_perm_cross
    delta_naive_plsc1_cross = naive_plsc1_cross - naive_plsc1_perm_cross

    tpnp_plsc_cross = np.mean(tpnp_plsc_cross, axis=1)
    tpnp_plsc1_cross = np.mean(np.array(tpnp_plsc1_cross)[:,:,0], axis=1)
    tpnp_plsc1_perm_cross = np.mean(np.array(tpnp_plsc1_perm_cross)[:,:,0], axis=1)
    delta_tpnp_plsc1_cross = tpnp_plsc1_cross - tpnp_plsc1_perm_cross

    naive_plsc_cross_title_mean = np.array(naive_plsc_cross_title_mean)
    trained_plsc_cross_title_mean = np.array(trained_plsc_cross_title_mean)
    naive_plsc1_cross_title_mean = np.array(naive_plsc1_cross_title_mean)
    trained_plsc1_cross_title_mean = np.array(trained_plsc1_cross_title_mean)
    naive_plsc1_perm_cross_title_mean = np.array(naive_plsc1_perm_cross_title_mean)
    trained_plsc1_perm_cross_title_mean = np.array(trained_plsc1_perm_cross_title_mean)
    delta_naive_plsc1_cross_title_mean = naive_plsc1_cross_title_mean - naive_plsc1_perm_cross_title_mean
    delta_trained_plsc1_cross_title_mean = trained_plsc1_cross_title_mean - trained_plsc1_perm_cross_title_mean

    tpnp_plsc_cross_title_mean = np.array(tpnp_plsc_cross_title_mean)
    tpnp_plsc1_cross_title_mean = np.array(tpnp_plsc1_cross_title_mean)
    tpnp_plsc1_perm_cross_title_mean = np.array(tpnp_plsc1_perm_cross_title_mean)
    delta_tpnp_plsc1_cross_title_mean = tpnp_plsc1_cross_title_mean - tpnp_plsc1_perm_cross_title_mean

    # now store all the data in a dictionary and output to a csv
    data_dict = {
        'apple naive prey': naive_apple,
        'acorn naive prey': naive_acorn,
        'catch naive predator': naive_catch,
        'apple trained prey': trained_apple,
        'acorn trained prey': trained_acorn,
        'catch trained predator': trained_catch,
        'apple tpnp prey': tpnp_apple,
        'acorn tpnp prey': tpnp_acorn,
        'catch tpnp predator': tpnp_catch,
        '# PLSC shared dim naive': naive_plsc,
        '# PLSC shared dim trained': trained_plsc,
        '# PLSC shared dim tpnp': tpnp_plsc,
        'delta PLSC1 naive': delta_naive_plsc1,
        'delta PLSC1 trained': delta_trained_plsc1,
        'delta PLSC1 tpnp': delta_tpnp_plsc1,
        '# PLSC shared dim cross naive': naive_plsc_cross,
        '# PLSC shared dim cross trained': trained_plsc_cross,
        '# PLSC shared dim cross tpnp': tpnp_plsc_cross,
        'delta PLSC1 cross naive': delta_naive_plsc1_cross,
        'delta PLSC1 cross trained': delta_trained_plsc1_cross,
        'delta PLSC1 cross tpnp': delta_tpnp_plsc1_cross,
    }
    df = pd.DataFrame().from_dict(data_dict, orient='index').T
    df.to_csv('/home/mikan/Documents/GitHub/meltingpot-2.2.0/examples/analyzing_rollout_results/naive_vs_trained/behavior_metrics.csv')

    for part_key in ['apple', 'acorn', 'catch', '# PLSC', 'delta PLSC1']:
        plt.figure(figsize=(6, 4))
        keys = [key for key in df.columns if part_key in key]
        sns.boxplot(data=df[keys])
        sns.swarmplot(data=df[keys], color='black')
        medians_of_selection = df[keys].median()

        # Label each box with its median value
        for i in range(medians_of_selection.shape[0]):
            plt.text(i, medians_of_selection[i], medians_of_selection[i], ha='center', va='bottom', color='red')

        # Rotate and wrap x-tick labels
        labels = [textwrap.fill(label, 12) for label in keys]
        plt.xticks(range(len(labels)), labels, rotation=90)

        plt.title(f'{part_key} comparison between naive and trained')
        plt.tight_layout()
        plt.savefig(
            f'/home/mikan/Documents/GitHub/meltingpot-2.2.0/examples/analyzing_rollout_results/naive_vs_trained/{part_key}_comparison.png')
        plt.show()

    