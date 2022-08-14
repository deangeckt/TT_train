import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

SEQ_LEN = 50
NUM_FEATURES = 4 * 25  # 25 landmarks X 4 features each - x,y,z,vis


def read_raw_data(score_path, data_path):
    metadata = {}

    score_df = pd.read_csv(score_path)
    score_df = score_df[score_df['frames'] < SEQ_LEN]

    data_df = pd.read_csv(data_path)
    amount_of_shots = len(score_df)
    print('all examples:', amount_of_shots)

    x = np.zeros((amount_of_shots, SEQ_LEN, NUM_FEATURES))
    y = []
    last_frame = 0

    for shot_index, row in tqdm(score_df.iterrows()):

        frames = row['frames']
        score = row['score']
        shot_name = row['name']

        curr_shot = np.array(data_df.iloc[last_frame: frames + last_frame, 1:NUM_FEATURES + 1])
        padding = np.zeros((SEQ_LEN - frames, NUM_FEATURES))
        curr_shot = np.vstack((curr_shot, padding))
        last_frame = frames

        if shot_index == amount_of_shots:
            break

        y.append(score)
        x[shot_index] = np.nan_to_num(curr_shot)
        metadata[shot_index] = {'name': shot_name, 'frames': frames}
    return torch.from_numpy(x).float(), np.array(y), metadata
