import cv2

import pandas as pd
from tqdm import tqdm
import os
from pose_extractor import shotsExtractor

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
for i in range(0, shotsExtractor.pose_max_idx + 1):
    pose_columns.extend(['{}_{}'.format(i, key) for key in single_pose_data])

score_df = pd.DataFrame(columns=['score', 'shot', 'frames'])
data_df = pd.DataFrame(columns=pose_columns)


def extract_from_file(full_path, debug=True):
    mv_key = full_path.split('/')[1]
    file_name = full_path.split('/')[2].split('.mp4')[0]
    is_left = file_name.split('_')[-1] == 'l' or file_name.split('_')[-1] == 'L'

    mv_th = params[mv_key]['mv_th']
    mv_wd = params[mv_key]['mv_wd']
    length = params[mv_key]['len']
    shots_delta = params[mv_key]['shots_delta']

    pos = shotsExtractor(mv_wd=mv_wd, mv_th=mv_th, shots_delta=shots_delta,
                         min_length=length, file_name=file_name)

    vid = cv2.VideoCapture(full_path)
    _, img = vid.read()
    while True:
        success, img = vid.read()
        if not success:
            break
        if is_left:
            img = cv2.flip(img, 1)

        pos.process(img)

        if debug:
            imS = cv2.resize(img, (540, 1000))  # Resize image

            cv2.imshow("Vid", imS)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if debug:
        cv2.destroyAllWindows()
        pos.debug_plot_diffs()
        pos.debug_save_shots()
        return pos

    if not debug:
        pos.save_shots_labeled_csv(score_df, data_df)


# shot_type = 'fco'
# dir_path = f'data/{shot_type}'
# for fname in tqdm(os.listdir(dir_path)):
#     extract_from_file('{}/{}'.format(dir_path, fname), debug=False)
#
# score_df.to_csv(f'labels/{shot_type}_score.csv')
# data_df.to_csv(f'labels/{shot_type}_data.csv')

pos = extract_from_file(r'data_examples/fco/fco_21_9.mp4', debug=True)

