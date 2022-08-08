import os
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
from PIL import Image
from visualization.custom_drawing_utils import custom_plot_landmarks
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils


def create_gif_for_given_shot(shot_name, frames_data):
    shot = convert_frames_to_mp_landmarks_shots(shot_name, frames_data)
    create_video_by_set_of_given_frames(shot)
    

def create_video_by_set_of_given_frames(shot):
    if not os.path.exists('visualization/shots_3d_demo/'):
        os.mkdir('visualization/shots_3d_demo/')
    if not os.path.exists('visualization/shots_3d_demo/' + shot.name):
        os.mkdir('visualization/shots_3d_demo/' + shot.name)
    i = 0
    for frame in shot.frames:
        i += 1
        points = Points_world_landmarks(frame.fr_points)
        plt = custom_plot_landmarks(points, mpPose.POSE_CONNECTIONS)
        plt.title(frame.name)
        filename = 'visualization/shots_3d_demo/' + shot.name + '/' + shot.name + i.__str__() + '.png'
        plt.savefig(filename, dpi=75)

    files = []
    for i in range(1, len(shot.frames)):
        seq = str(i)
        file_names = shot.name + seq + '.png'
        files.append(file_names)

    frames = []
    for i in files:
        new_frame = Image.open("visualization/shots_3d_demo/" + shot.name + '/' + i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save("visualization/shots_3d_demo/" + shot.name + '.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=500, loop=0)


def convert_frames_to_mp_landmarks_shots(file_name, frames):
    frames_t = []
    for row in frames:
        landmark = []
        index = 1
        for col in row:
            if index == 1:
                land = Landmark(col, 0, 0, 0)
            elif index == 2:
                land.y = col
            elif index == 3:
                land.z = col
            elif index == 4:
                land.visibility = col
                landmark.append(land)
                index = 0
            index += 1

        for i in range(10):
            land = Landmark(0, 0, 0, 0.4)
            landmark.append(land)
        frames_t.append(Frame(landmark, file_name))
    full_shots = Shot(frames_t, file_name)
    return full_shots

class Frame:
    def __init__(self, fr_points , name):
        self.fr_points = fr_points
        self.name = name


class Shot:
    def __init__(self, frames, name):
        self.frames = frames
        self.name = name


class Landmark:
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = 1

    #fake - checked in mediapope code
    def HasField(self, item):
        return True


class Points_world_landmarks:
    def __init__(self, landmark):
        self.landmark = landmark


# Example - run this code to display the gif that was created
# from IPython.display import Image
#
# with open('./fts_10_7.5_3.gif','rb') as f:
#     display(Image(data=f.read(), format='png'))


