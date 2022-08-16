import os
import mediapipe as mp
from PIL import Image
from visualization.custom_drawing_utils import custom_plot_landmarks

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils

vis_folder = 'visualization/shots_3d_demo'


def create_gif_for_given_shot(shot_name, frames_data, amount, landmark_to_remove=[]):
    """
    visualize a shot landmarks
    :param shot_name: name of the shot - will be saved in a new folder under this name
    :param frames_data: the 3D matrix of a shot
    :param amount: amount of none zero frames of the shot
    :param landmark_to_remove: indices of landmarks to remove - make sure to filter the features
    of the matrix via feature selection first
    :return:
    """
    shot = convert_frames_to_mp_landmarks_shots(shot_name, frames_data, amount, landmark_to_remove)
    create_video_by_set_of_given_frames(shot)
    print(f'finished saving gif: {shot_name}')


def create_video_by_set_of_given_frames(shot):
    os.makedirs(vis_folder, exist_ok=True)
    shot_folder = os.path.join(vis_folder, shot.name)
    os.makedirs(shot_folder, exist_ok=True)

    frames = []
    for frame in shot.frames:
        points = Points_world_landmarks(frame.fr_points)
        plt = custom_plot_landmarks(points, mpPose.POSE_CONNECTIONS)

        plt.title(f'{shot.name}_{frame.name}')
        filename = f'{shot_folder}/{shot.name}_{frame.name}.png'
        plt.savefig(filename, dpi=75)
        plt.close()
        frames.append(Image.open(filename))

    # Save into a GIF file that loops forever
    frames[0].save(f'{shot_folder}.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=250, loop=0)


def convert_frames_to_mp_landmarks_shots(file_name, frames, amount, landmark_to_remove):
    frames_t = []

    for fr_idx, row in enumerate(frames):
        landmark = []
        landmark_index = 0
        for i in range(0, 24 + 1):
            if i in landmark_to_remove:
                land = Landmark(0, 0, 0, 0.0)
                landmark.append(land)
            else:
                land = Landmark(row[landmark_index].item(),
                                row[landmark_index+1].item(),
                                row[landmark_index+2].item(),
                                row[landmark_index+3].item())
                landmark.append(land)
                landmark_index += 4

        # landmark 25-32 not our in use
        for i in range(10):
            land = Landmark(0, 0, 0, 0.0)
            landmark.append(land)
        frames_t.append(Frame(landmark, fr_idx))

        if fr_idx >= amount - 1:
            break

    full_shots = Shot(frames_t, file_name)
    return full_shots


class Frame:
    def __init__(self, fr_points, name):
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

    # fake - checked in mediapope code
    def HasField(self, item):
        return True


class Points_world_landmarks:
    def __init__(self, landmark):
        self.landmark = landmark
