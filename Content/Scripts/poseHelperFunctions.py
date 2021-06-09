import tensorflow as tf
import numpy as np

from config import config


def get_keypoint_positions(heatmaps, num_kps, h, w):
    kps_pos = []
    for kp in range(num_kps):
        max_val = heatmaps[0][0][0][kp]
        max_row = 0
        max_col = 0

        for r in range(h):
            for c in range(w):
                curr_val = heatmaps[0][r][c][kp]
                if curr_val > max_val:
                    max_val = curr_val
                    max_row = r
                    max_col = c

        kps_pos.append([max_row, max_col])

    return kps_pos


def calc_offsets(heatmaps, kps_pos, offsets, num_kps, h, w):
    x_coords = np.empty(num_kps, dtype=np.int)
    y_coords = np.empty(num_kps, dtype=np.int)
    conf_scores = np.empty(num_kps, dtype=np.float)

    for i in range(num_kps):
        pos_y = kps_pos[i][0]
        pos_x = kps_pos[i][1]

        y_coords[i] = int(pos_y / float(h - 1) * config['IMAGEH'] + offsets[0][pos_y][pos_x][i])
        x_coords[i] = int(pos_x / float(w - 1) * config['IMAGEW'] + offsets[0][pos_y][pos_x][i])
        conf_scores[i] = sigmoid(heatmaps[0][pos_y][pos_x][i])

        return x_coords, y_coords, conf_scores