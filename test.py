from utils.general import read_raw_data
from visualization.points_visualization import create_gif_for_given_shot
import numpy as np

x,y,n = read_raw_data(r'shot_extractor/data_shots/test_score.csv', r'shot_extractor/data_shots/test_data.csv')


# x, y, metadata = read_raw_data(score_path='labels/fco_score.csv', data_path='labels/fco_data.csv')


create_gif_for_given_shot('small', x[1], 22, [])
