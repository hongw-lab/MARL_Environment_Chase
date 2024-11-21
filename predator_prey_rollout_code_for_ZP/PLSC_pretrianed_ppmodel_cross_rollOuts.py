import os
import numpy as np
import pickle
from numba import jit
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import random

@jit(nopython=True)  # Use Numba decorator to compile this function to machine code
def PLSC(h1, h2):
    # Calculate covariance matrix
    h1_cont = np.ascontiguousarray(h1)
    h2_cont = np.ascontiguousarray(h2)
    n_samples = h1.shape[0]
    covMat = np.dot(h1_cont.T, h2_cont) / (n_samples - 1)
    U, s, Vh = np.linalg.svd(covMat, full_matrices=False)
    # Compute transformed matrices
    A = np.dot(h1_cont, U)
    B = np.dot(h2_cont, Vh.T)  # Transpose Vh to get V
    return A, B

@jit(nopython=True)  # Use Numba decorator to compile this function to machine code
def compute_diagonal_covariance_and_correlation(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same dimensions.")

    n, d = A.shape

    covariance = np.zeros(d)
    std_A = np.zeros(d)
    std_B = np.zeros(d)

    # Compute covariance and standard deviations manually
    for i in range(d):
        for j in range(n):
            covariance[i] += (A[j, i] * B[j, i])
            std_A[i] += A[j, i] ** 2
            std_B[i] += B[j, i] ** 2

        covariance[i] /= (n - 1)
        std_A[i] = np.sqrt(std_A[i] / n)
        std_B[i] = np.sqrt(std_B[i] / n)

    correlation = covariance / (std_A * std_B)

    return covariance, correlation

def isExceedingConfidence_linear_percentile(shuffle, pt, confidence=0.95):
    # data is in the shape of (nNeurons, n_shuffles)
    # pt is in the shape of (nNeurons)
    # return 1 for > right outreach, -1 for < left outreach, 0 for within bounds
    low = (1-confidence) / 2 * 100 # get the percentile required by the confidence interval
    low_bound, high_bound = np.nanpercentile(shuffle, [low, 100 - low], axis=1)
    pos_exceed_confidence = pt >= high_bound
    neg_exceed_confidence = pt <= low_bound
    significant_pts = 1 * pos_exceed_confidence - 1 * neg_exceed_confidence
    return significant_pts


scaler = StandardScaler()
def process_cross_episode(trial_directory, title, eId, timesteps=1000, num_perm=10, permutations=None, scaler=scaler,
                          **kwargs):
    """Function to process a single episode."""

    title_one = title.split(' ')[0]
    title_two = title.split(' ')[1]
    title_one = '_'.join(title_one.split('_')[:-1])
    title_two = '_'.join(title_two.split('_')[:-1])
    with open(os.path.join(trial_directory, f'{title_one}_{eId}.pkl'), 'rb') as f:
        data = pickle.load(f)
    h1 = [tmp['lstmMemory'][0] for tmp in data]
    with open(os.path.join(trial_directory, f'{title_two}_{eId}.pkl'), 'rb') as f:
        data = pickle.load(f)
    h2 = [tmp['lstmMemory'][1] for tmp in data]
    h1 = np.array(h1)[:timesteps]
    h2 = np.array(h2)[:timesteps]


    # Normalize h1 and h2
    h1 = scaler.fit_transform(h1)
    h2 = scaler.fit_transform(h2)
    try:
        A,B = PLSC(h1, h2)
        cov_diag, cor_diag = compute_diagonal_covariance_and_correlation(A, B)
    except:
        nan_array = np.zeros(h1.shape[-1]) * np.nan
        perm_nan_array = np.zeros((num_perm, h1.shape[-1])) * np.nan
        return np.nan, nan_array, nan_array, perm_nan_array, perm_nan_array, nan_array, nan_array
    if permutations is None:
        permutations = np.random.randint(low=0, high=A.shape[0], size=num_perm)
    cov_perm_array = np.zeros((num_perm, A.shape[1])) * np.nan
    cor_perm_array = np.zeros((num_perm, A.shape[1])) * np.nan

    for pi, perm in enumerate(permutations):
        h2_perm = np.roll(h2, perm, axis=0)
        try:
            A_perm, B_perm = PLSC(h1, h2_perm)
            cov_diag_perm, cor_diag_perm = compute_diagonal_covariance_and_correlation(A_perm, B_perm)
            cov_perm_array[pi] = cov_diag_perm
            cor_perm_array[pi] = cor_diag_perm
        except:
            continue
    cov_sig = isExceedingConfidence_linear_percentile(cov_perm_array.T, cov_diag) > 0
    cor_sig = isExceedingConfidence_linear_percentile(cor_perm_array.T, cor_diag) > 0
    rank_cov = np.where(cov_sig == 0)[0][0] if (cov_sig == 0).any() else len(cov_sig)
    rank_cor = np.where(cor_sig == 0)[0][0] if (cor_sig == 0).any() else len(cor_sig)
    rank = np.min([rank_cov, rank_cor])

    return rank, cov_diag, cor_diag, cov_perm_array, cor_perm_array, cov_sig, cor_sig


def processing_cross_episode_wrapper(**kwargs):
    try:
        # Extract the data needed from process_episode
        rank, cov_diag, cor_diag, cov_perm_array, cor_perm_array, cov_sig, cor_sig = process_cross_episode(**kwargs)
        # Return the unpacked results along with the title and episode ID for easy indexing
        return rank, cov_diag, cor_diag, cov_perm_array, cor_perm_array, cov_sig, cor_sig, kwargs['title'], kwargs['eId']
    except Exception as e:
        # Handle potential errors gracefully
        print(f"Error processing episode {kwargs['eId']} for {kwargs['title']}: {str(e)}")
        # Return None or a default value set to maintain structure
        return None, None, None, None, None, None, None, kwargs['title'], kwargs['eId']


if __name__ == '__main__':
    video_path = f'{os.environ["HOME"]}/Documents/GitHub/meltingpot-2.2.0/examples/videos/open_field_1_1/'
    predator_ids = [0, 1, 2, 3, 4]
    # prey_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    prey_ids = [3, 5, 6, 7, 8]
    titles = [f'{predator_id}_{prey_id}' for predator_id in predator_ids for prey_id in prey_ids]
    cross_titles = []
    for title in titles:
        m = 0
        while m < 10:
            second_title = random.choice(titles)
            while second_title == title:
                second_title = random.choice(titles)
            if f'{title}_predator {second_title}_prey' in cross_titles:
                continue
            m += 1
            cross_titles.append(f'{title}_predator {second_title}_prey')

    results = Parallel(n_jobs=24)(delayed(processing_cross_episode_wrapper)(
        trial_directory=video_path, video_path=video_path, title=cross_title,
        eId=eId, timesteps=1000, num_perm=200, permutations=None, scaler=scaler
    ) for eId in range(1,101) for cross_title in cross_titles)

    result_dict = {}
    for result in results:
        rank, cov_diag, cor_diag, cov_perm_array, cor_perm_array, cov_sig, cor_sig, title, eId = result
        if title not in result_dict:
            result_dict[title] = {title: [] for title in ['rank', 'cov', 'cor', 'cov_sig', 'cor_sig', 'cov_perm_mean', 'cov_perm_median', 'cor_perm_mean', 'cor_perm_median']}
        result_dict[title]['rank'].append(rank)
        result_dict[title]['cov'].append(cov_diag)
        result_dict[title]['cov_perm_mean'].append(np.nanmean(cov_perm_array, axis=0))
        result_dict[title]['cov_perm_median'].append(np.nanmedian(cov_perm_array, axis=0))
        result_dict[title]['cor_perm_mean'].append(np.nanmean(cor_perm_array, axis=0))
        result_dict[title]['cor_perm_median'].append(np.nanmedian(cor_perm_array, axis=0))
        result_dict[title]['cor'].append(cor_diag)
        result_dict[title]['cov_sig'].append(cov_sig)
        result_dict[title]['cor_sig'].append(cor_sig)
    with open(f'{video_path}PLSC_results_cross_rollout_dict_NP.pkl', 'wb') as f:
        pickle.dump(result_dict, f)
