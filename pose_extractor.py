import os
import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


def pose_est(img):
    res = []
    results = pose.process(img)
    if results.pose_landmarks:
        # mpDraw.plot_landmarks(results.pose_world_landmarks, mpPose.POSE_CONNECTIONS)
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            if id > shotsExtractor.pose_max_idx:
                continue
            res.append(lm)
            # h, w, c = img.shape
            # cx, cy = int(lm.x * w), int(lm.y * h)
            # cv2.putText(img, str(id), (cx, cy),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            # cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return res


def get_landmarks_diff(landmarks: list, coord: str):
    if len(landmarks) <= 1:
        return 0
    arr = []
    for lm in landmarks:
        arr.append(getattr(lm, coord))

    return np.mean(np.abs(np.diff(arr)))


class shotsExtractor:
    right_hand = [12, 14, 16, 18, 20, 22]
    left_hand = [11, 13, 15, 17, 19, 21]
    pose_max_idx = 24

    def __init__(self, file_name, mv_wd, mv_th, shots_delta, length):
        # thresholds
        self.mv_wd = mv_wd
        self.mv_th = mv_th
        self.length = length
        self.shots_delta = shots_delta

        self.file_name = file_name
        self.movements = []
        self.frames = []
        self.pose = []
        self.mv_start_idx = -1
        self.mv_end_idx = -1
        self.diffs = defaultdict(list)

    def save_shots_labeled_csv(self, score_df, data_df):
        self.debug_save_shots()
        label_score = None
        if len(self.file_name.split('_')) > 2:
            label_score = self.file_name.split('_')[2]

        for mv_idx, mv in enumerate(self.movements):
            for fr_idx, frame in enumerate(range(mv['s'], mv['e'] + 1)):
                data_key = '{}_{}_{}'.format(self.file_name, mv_idx, fr_idx)
                data_df.loc[data_key] = {}
                for pos_idx, pos in enumerate(self.pose[frame]):
                    if pos_idx > shotsExtractor.pose_max_idx:
                        break
                    data_df.loc[data_key]['{}_x'.format(pos_idx)] = pos.x
                    data_df.loc[data_key]['{}_y'.format(pos_idx)] = pos.y
                    data_df.loc[data_key]['{}_z'.format(pos_idx)] = pos.z
                    data_df.loc[data_key]['{}_vis'.format(pos_idx)] = pos.visibility

            score_key = '{}_{}'.format(self.file_name, mv_idx)
            score_df.loc[score_key] = {'score': label_score, 'shot': mv_idx,
                                       'frames': mv['e'] + 1 - mv['s']}

    def debug_save_shots(self):
        type_ = self.file_name.split('_')[0]
        dir_ = f'data_shots/{type_}'
        os.makedirs(dir_, exist_ok=True)

        dir_path = '{}/{}'.format(dir_, self.file_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for mv_idx, mv in enumerate(self.movements):
            for fr_idx, fr in enumerate(range(mv['s'], mv['e'] + 1)):
                cv2.imwrite(
                    '{}/{}/{}_{}_{}.jpg'.format(dir_, self.file_name, self.file_name, mv_idx,
                                                fr_idx), self.frames[fr])

    def debug_plot_diffs(self):
        st_idx = [m['s'] for m in self.movements]
        en_idx = [m['e'] for m in self.movements]
        zeros = [0] * len(st_idx)
        plt.scatter(x=st_idx, y=zeros, color='green')
        plt.scatter(x=en_idx, y=zeros, color='red')

        dyn_mv_th = self.mv_th * np.mean(self.diffs['mxy'])
        plt.axhline(y=dyn_mv_th, color='purple', linestyle='-')
        for st in st_idx:
            plt.axvline(x=st, color='green', ls='--')
        for en in en_idx:
            plt.axvline(x=en, color='red', ls='--')

        for d in self.diffs:
            plt.plot(self.diffs[d], label=d)
        plt.legend()
        plt.title(self.file_name)
        plt.show()

    def add_move(self):
        self.movements.append({'s': self.mv_start_idx, 'e': self.mv_end_idx})
        self.mv_start_idx = -1
        self.mv_end_idx = -1

    def get_change(self, frame_count, pose_idxs):
        landmarks = defaultdict(list)
        last_pose = self.pose[-frame_count:]
        x_means = []
        y_means = []
        for p in last_pose:
            if len(p) == 0:
                continue
            for idx in pose_idxs:
                landmarks[idx].append(p[idx])

        for idx in landmarks:
            x_ = get_landmarks_diff(landmarks[idx], 'x')
            y_ = get_landmarks_diff(landmarks[idx], 'y')
            x_means.append(x_)
            y_means.append(y_)

        if len(x_means):
            x_mean = np.mean(x_means)
            y_mean = np.mean(y_means)
            xy_mean = np.mean([x_mean, y_mean])

            self.diffs['mxy'].append(xy_mean)
            return xy_mean
        else:
            self.diffs['mxy'].append(0)

        return None

    def process(self, img):
        self.frames.append(img)
        self.pose.append(pose_est(img))

        total_frames = len(self.frames)
        if total_frames < self.mv_wd:
            return

        curr_frame = total_frames - self.mv_wd
        mv_metric = self.get_change(frame_count=self.mv_wd, pose_idxs=shotsExtractor.right_hand)
        if mv_metric is None:
            return

        dyn_metric = self.mv_th * np.mean(self.diffs['mxy'])
        if mv_metric >= dyn_metric:
            if self.mv_start_idx == -1:
                if len(self.movements) > 0:
                    last_end = self.movements[-1]['e']
                    if curr_frame - last_end < self.shots_delta:
                        # print('extend due delta')
                        last_mv = self.movements.pop()
                        self.mv_start_idx = last_mv['s']
                        return

                self.mv_start_idx = curr_frame
        else:
            if self.mv_start_idx != -1:
                curr_len = curr_frame - self.mv_start_idx
                if curr_len > self.mv_wd:
                    if curr_len <= self.length:
                        # print('reset due len ', curr_len)
                        self.mv_start_idx = -1
                        return
                    self.mv_end_idx = curr_frame
                    self.add_move()
