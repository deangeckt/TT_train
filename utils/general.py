import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

SEQ_LEN = 50
NUM_FEATURES = 4 * 25  # 25 landmarks X 4 features each - x,y,z,vis


def read_raw_data(score_path, data_path):
    metadata = []
    score_df = pd.read_csv(score_path)
    data_df = pd.read_csv(data_path)

    x = []
    y = []
    last_frame = 0

    for shot_index, row in tqdm(score_df.iterrows()):
        frames = row['frames']
        if frames >= SEQ_LEN:
            last_frame += frames
            continue
        score = row['score']
        shot_name = row['name']

        curr_shot = np.array(data_df.iloc[last_frame: frames + last_frame, 1:NUM_FEATURES + 1])
        padding = np.zeros((SEQ_LEN - frames, NUM_FEATURES))
        curr_shot = np.vstack((curr_shot, padding))
        last_frame += frames

        y.append(score)
        x.append(np.nan_to_num(curr_shot))
        metadata.append({'name': shot_name, 'frames': frames})

    x = np.array(x)
    x = torch.from_numpy(x).float()
    return x, np.array(y), np.array(metadata)

