# Created by shaji on 21-Mar-23
import numpy as np
import torch
import math
import itertools


def hough_transform(x, origin=None):
    # https://stats.stackexchange.com/questions/375787/how-to-cluster-parts-of-broken-line-made-of-points
    if origin is None:
        origin = [0, 0]

    x = np.transpose(x - origin)
    dx = np.vstack((np.apply_along_axis(np.diff, 0, x), [0.0, 0.0]))
    v = np.vstack((-dx[:, 1], dx[:, 0])).T
    n = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2).reshape(-1, 1)
    res = np.column_stack((np.sum(x * (v / (n + 1e-20)), axis=1), np.arctan2(v[:, 1], v[:, 0]))).T
    return res


def extract_patterns(args, pm_prediction):
    positions = pm_prediction[:, :, :3]

    sub_pattern_numbers = 0
    for i in range(3, args.e + 1):
        sub_pattern_numbers += math.comb(args.e, i)

    sub_patterns = torch.zeros(size=(sub_pattern_numbers, positions.shape[0], positions.shape[1], positions.shape[2]))
    for i in range(3, args.e + 1):
        for ss_i, subset in enumerate(itertools.combinations(list(range(positions.shape[1])), i)):
            sub_patterns[ss_i, :, :len(subset), :] = positions[:, subset, :]

    return sub_patterns


