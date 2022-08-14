from utils.general import read_raw_data
from visualization.points_visualization import create_gif_for_given_shot

x,y,n = read_raw_data(r'shot_extractor/data_shots/test_score.csv', r'shot_extractor/data_shots/test_data.csv')
create_gif_for_given_shot('fco_16_9_1', x[1], n[1]['frames'], [1,2,3,4,5])
