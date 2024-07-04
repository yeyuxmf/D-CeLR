import numpy as np


CLASS_NUMS = 19
PointNms = 19
IMG_Height = 1024
IMG_Width = 1024


VIEW_NUMS =10

SAVE_MODEL = 100


DECAY_STEPS = {
    100: 0.0001,
    200: 0.00005,
    300: 0.00001,
    400: 0.000001,
    500: 0.0000001
}
MAX_EPOCHS =601
LEARNING_RATE = 0.00015


ERROR_RANGE = [2, 2.5, 3, 4]

indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
DIFFv = np.array([
    [1.1, 0.9],
    [0.9, 0,7],
    [0, 0],
    [-12.9, 16.4],
    [2, -2.2],
    [-6.4, -5.7],
    [9.0, -24.5],
    [1.2, 0],
    [0.9, 0],
    [0.5, 0],
    [6.1,  9.6],
    [1.2, 0],
    [1.1, 0.3],
    [10.4, 23.0],
    [14.4, -11.9],
    [3.2, 1.1],
    [-18.5, 40.3],
    [-2.6, 1.1],
    [2.2, -1.1]])


