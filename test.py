from utils.general import read_raw_data
from visualization.points_visualization import create_gif_for_given_shot
import numpy as np

# x,y,n = read_raw_data(r'shot_extractor/data_shots/test_score.csv', r'shot_extractor/data_shots/test_data.csv')
# x, y, metadata = read_raw_data(score_path='labels/fco_score.csv', data_path='labels/fco_data.csv')
x, y, metadata = read_raw_data(score_path='labels/fts_score.csv', data_path='labels/fts_data.csv')

# score_threshold = 6
# labels = np.array(y)
# labels[labels <= score_threshold] = 0
# labels[labels > score_threshold] = 1
# unique, counts = np.unique(labels, return_counts=True)
# for idx, c in enumerate(counts):
#     print(f'label: {idx}: {c} samples')
#
#
# pos = np.where(labels == 1)
# neg = np.where(labels == 0)
# data = np.array(x)
# pos_data = data[pos]


create_gif_for_given_shot(metadata[0]['name'], x[0], metadata[0]['frames'])
