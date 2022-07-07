import os

import cv2
import time
import pandas as pd
from tqdm import tqdm

from TT_trainer.post_est import PoseMovement

params = {'fts': {'mv_th': 0.6, 'mv_wd': 3, 'len': 6, 'shots_delta': 3},
          'fch': {'mv_th': 0.0, 'mv_wd': 5, 'len': 6, 'shots_delta': 3},
          'fco': {'mv_th': 0.6, 'mv_wd': 3, 'len': 6, 'shots_delta': 3},
          'bts': {'mv_th': 2, 'mv_wd': 3},
          'bch': {'mv_th': 2, 'mv_wd': 3},
          'bco': {'mv_th': 2, 'mv_wd': 3},
          'home_test': {'mv_th': 0.6, 'mv_wd': 3, 'len': 6, 'shots_delta': 3},
          'newfco': {'mv_th': 0.6, 'mv_wd': 3, 'len': 6, 'shots_delta': 3},
          }

single_pose_data = ['x', 'y', 'z', 'vis']
pose_columns = []
for i in range(0, PoseMovement.pose_max_idx + 1):
    pose_columns.extend(['{}_{}'.format(i, key) for key in single_pose_data])

score_df = pd.DataFrame(columns=['score', 'shot', 'frames'])
data_df = pd.DataFrame(columns=pose_columns)
# score_df = pd.read_excel('labels/fco_score.xlsx', index_col='name')
# data_df = pd.read_csv('labels/fco_data.csv', index_col='name')


def extract_from_file(full_path):
    mv_key = full_path.split('/')[1]
    file_name = full_path.split('/')[2].split('.mp4')[0]
    is_left = file_name.split('_')[-1] == 'l' or file_name.split('_')[-1] == 'L'

    mv_th = params[mv_key]['mv_th']
    mv_wd = params[mv_key]['mv_wd']
    length = params[mv_key]['len']
    shots_delta = params[mv_key]['shots_delta']

    pos = PoseMovement(mv_wd=mv_wd, mv_th=mv_th, shots_delta=shots_delta,
                       length=length, file_name=file_name)

    vid = cv2.VideoCapture(full_path)
    _, img = vid.read()
    while True:
        success, img = vid.read()
        if not success:
            break

        if is_left:
            img = cv2.flip(img, 1)
        pos.process(img)
        imS = cv2.resize(img, (540, 1000))  # Resize image

        cv2.imshow("Vid", imS)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # DEBUG
    cv2.destroyAllWindows()
    pos.debug_plot_diffs()
    # pos.debug_save_shots()
    return pos

    # REAL
    # pos.save_shots_labeled_csv(score_df, data_df)


# dir_path = 'data/fts'
# for fname in tqdm(os.listdir(dir_path)):
#     extract_from_file('{}/{}'.format(dir_path, fname))

# score_df.to_csv('labels/fts_score.csv')
# data_df.to_csv('labels/fts_data.csv')


pos = extract_from_file(r'data/fco/fco_12_5.mp4')

