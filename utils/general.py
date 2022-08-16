import random
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

SEQ_LEN = 50
NUM_FEATURES = 4 * 25  # 25 landmarks X 4 features each - x,y,z,vis
right_hand = [12, 14, 16, 18, 20, 22]


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


def rand_shot(shot, amount, frame_prob=0.5):
    """
    as part of data augmentation - this randomizes a shot based on a given shot - on the right hand area
    :param shot: 2d matrix of a shot with full feature matrix, feature select can be done afterwards
    :param amount: amount of none zero / none padded frames
    :param frame_prob: the probability to choose a frame and randomizes it to a new frame
    :return: random shot in the same shape of the given shot
    """
    landmark_prob = 0.5
    x_prob, y_prob, z_prob = 0.5, 0.5, 0.5
    cord_change = (-0.01, 0.01)
    new_shot = shot.clone()

    for fr_idx in range(amount):
        if random.random() < frame_prob:
            continue
        frame = new_shot[fr_idx]
        if torch.equal(frame, torch.zeros(NUM_FEATURES)):
            continue
        for l_idx in right_hand:
            if random.random() < landmark_prob:
                continue

            landmark = frame[l_idx * 4: (l_idx * 4) + 4]
            x, y, z, _ = landmark
            if random.random() > x_prob:
                x += random.uniform(*cord_change)
            if random.random() > y_prob:
                y += random.uniform(*cord_change)
            if random.random() > z_prob:
                z += random.uniform(*cord_change)

            landmark[landmark > 1.0] = 1.0
            landmark[landmark < -1.0] = -1.0

    return new_shot
