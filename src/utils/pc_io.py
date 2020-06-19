import multiprocessing
import tqdm
import functools
import numpy as np
import logging
import pandas as pd
from . import color_space
from pyntcloud import PyntCloud

logger = logging.getLogger(__name__)


def arr_to_pc(arr, cols, types):
    d = {}
    for i in range(arr.shape[1]):
        col = cols[i]
        dtype = types[i]
        d[col] = arr[:, i].astype(dtype)
    df = pd.DataFrame(data=d)
    pc = PyntCloud(df)
    return pc


# Fit points into a [0, 1] cube
def normalize_points(points):
    min_p = np.min(points)
    max_p = np.max(points)
    value_range = max_p - min_p
    points = (points - min_p) / value_range

    return points, (value_range, min_p)


def denormalize_points(points, norm_params):
    value_range, min_p = norm_params
    points = (points * value_range) + min_p
    return points


def load_file(f):
    pc = PyntCloud.from_file(f)
    geo_cols = ['x', 'y', 'z']
    color_cols = ['red', 'green', 'blue']

    points, norm_params = normalize_points(pc.points[geo_cols].values)
    pc.points[geo_cols] = points

    colors = pc.points[color_cols] / 255.
    pc.points[color_cols] = colors

    return pc.points[geo_cols + color_cols].values, norm_params


def load_points(files):
    with multiprocessing.Pool() as p:
        ret = list(tqdm.tqdm(p.imap(load_file, files), total=len(files)))
    return ret


def write_pc(arr, path):
    DATA_COLUMNS = ['x', 'y', 'z', 'red', 'green', 'blue']
    DATA_TYPES = (['float32'] * 3) + (['uint8'] * 3)
    d = {}
    for i, (col, dtype) in enumerate(zip(DATA_COLUMNS, DATA_TYPES)):
        d[col] = arr[:, i].astype(dtype)
    df = pd.DataFrame(data=d)
    pc = PyntCloud(df)
    pc.to_file(path)
