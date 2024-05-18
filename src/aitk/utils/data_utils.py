# Created by jing at 31.05.23
import itertools
import math
import torch
from sklearn.linear_model import LinearRegression

import config


def prop2index(props, g_type="group"):
    indices = []
    if g_type == "group":
        for prop in props:
            indices.append(config.group_tensor_index[prop])

    elif g_type == "object":
        for prop in props:
            indices.append(config.obj_tensor_index[prop])
    else:
        raise ValueError
    return indices


def get_comb(data, comb_size):
    pattern_numbers = math.comb(data.shape[0], comb_size)
    indices = torch.zeros(size=(pattern_numbers, comb_size), dtype=torch.uint8)

    for ss_i, subset_indice in enumerate(itertools.combinations(data.tolist(), comb_size)):
        indices[ss_i] = torch.tensor(sorted(subset_indice), dtype=torch.uint8)
    return indices


def in_ranges(value, line_ranges):
    for min_v, max_v in line_ranges:
        if value < max_v and value > min_v:
            return True
    return False


def euclidean_distance(point_groups_screen, center):
    squared_distance = torch.sum(torch.square(point_groups_screen - torch.tensor(center)), dim=1)
    distance = torch.sqrt(squared_distance)
    return distance


def to_line_tensor(objs, line_sc, line_error):
    obj_tensor_index = config.obj_tensor_index
    group_tensor_index = config.group_tensor_index
    line_tensor = torch.zeros(len(group_tensor_index.keys()))

    colors = objs[:, [obj_tensor_index[i] for i in config.obj_color]]
    shapes = objs[:, [obj_tensor_index[i] for i in config.obj_shapes]]

    colors_normalized = colors.sum(dim=0) / colors.shape[0]
    shapes_normalized = shapes.sum(dim=0) / shapes.shape[0]
    # 0:2 center_x, center_z
    # 2 slope
    # 3 x_length
    # 4 z_length
    # 5 is_line
    # 6 is_circle
    # 7 probability

    line_tensor[group_tensor_index["x"]] = objs[:, 0].mean()
    line_tensor[group_tensor_index["y"]] = objs[:, 1].mean()
    line_tensor[group_tensor_index["z"]] = objs[:, 2].mean()

    line_tensor[group_tensor_index["color_counter"]] = op_count_nonzeros(colors.sum(dim=0), axis=0, epsilon=1e-10)
    line_tensor[group_tensor_index["shape_counter"]] = op_count_nonzeros(shapes.sum(dim=0), axis=0, epsilon=1e-10)

    colors_normalized[colors_normalized < 0.99] = 0
    line_tensor[group_tensor_index['red']] = colors_normalized[0]
    line_tensor[group_tensor_index['green']] = colors_normalized[1]
    line_tensor[group_tensor_index['blue']] = colors_normalized[2]

    shapes_normalized[shapes_normalized < 0.99] = 0
    line_tensor[group_tensor_index['sphere']] = shapes_normalized[0]
    line_tensor[group_tensor_index['cube']] = shapes_normalized[1]

    # single_error = torch.tensor(line_error).sum() / torch.tensor(line_error).shape[0]
    line_tensor[group_tensor_index["line"]] = 1 - torch.tensor(line_error).sum() / torch.tensor(line_error).shape[0]
    line_tensor[group_tensor_index['circle']] = 0
    line_tensor[group_tensor_index["x_length"]] = objs[:, 0].max() - objs[:, 0].min()
    line_tensor[group_tensor_index["y_length"]] = objs[:, 1].max() - objs[:, 1].min()
    line_tensor[group_tensor_index["z_length"]] = objs[:, 2].max() - objs[:, 2].min()

    line_tensor[group_tensor_index["x_center_screen"]] = line_sc["center"][0]
    line_tensor[group_tensor_index["y_center_screen"]] = line_sc["center"][1]

    line_tensor[group_tensor_index["screen_left_x"]] = line_sc["end_A"][0]
    line_tensor[group_tensor_index["screen_left_y"]] = line_sc["end_A"][1]
    line_tensor[group_tensor_index["screen_right_x"]] = line_sc["end_B"][0]
    line_tensor[group_tensor_index["screen_right_y"]] = line_sc["end_B"][1]

    line_tensor[group_tensor_index["axis_x"]] = 0
    line_tensor[group_tensor_index["axis_z"]] = 0
    line_tensor[group_tensor_index["screen_axis_x"]] = 0
    line_tensor[group_tensor_index["screen_axis_z"]] = 0

    line_tensor[group_tensor_index["size"]] = objs.shape[0]

    line_tensor = line_tensor.reshape(-1)

    return line_tensor


def to_circle_tensor(objs, cir, cir_sc, cir_error):
    group_tensor_index = config.group_tensor_index
    obj_tensor_index = config.obj_tensor_index
    cir_tensor = torch.zeros(len(group_tensor_index.keys()))

    colors = objs[:, [obj_tensor_index[i] for i in config.obj_color]]
    shapes = objs[:, [obj_tensor_index[i] for i in config.obj_shapes]]
    colors_normalized = colors.sum(dim=0) / colors.shape[0]
    shapes_normalized = shapes.sum(dim=0) / shapes.shape[0]

    cir_tensor[group_tensor_index["x"]] = cir["center"][0]
    cir_tensor[group_tensor_index["y"]] = objs[:, 1].mean()
    cir_tensor[group_tensor_index["z"]] = cir["center"][1]

    cir_tensor[group_tensor_index["color_counter"]] = op_count_nonzeros(colors.sum(dim=0), axis=0, epsilon=1e-10)
    cir_tensor[group_tensor_index["shape_counter"]] = op_count_nonzeros(shapes.sum(dim=0), axis=0, epsilon=1e-10)

    colors_normalized[colors_normalized < 0.99] = 0
    cir_tensor[group_tensor_index['red']] = colors_normalized[0]
    cir_tensor[group_tensor_index['green']] = colors_normalized[1]
    cir_tensor[group_tensor_index['blue']] = colors_normalized[2]

    shapes_normalized[shapes_normalized < 0.99] = 0
    cir_tensor[group_tensor_index['sphere']] = shapes_normalized[0]
    cir_tensor[group_tensor_index['cube']] = shapes_normalized[1]

    cir_tensor[group_tensor_index["line"]] = 0
    cir_tensor[group_tensor_index["circle"]] = 1 - cir_error.sum() / cir_error.shape[0]

    cir_tensor[group_tensor_index["x_length"]] = objs[:, 0].max() - objs[:, 0].min()
    cir_tensor[group_tensor_index["y_length"]] = objs[:, 1].max() - objs[:, 1].min()
    cir_tensor[group_tensor_index["z_length"]] = objs[:, 2].max() - objs[:, 2].min()
    cir_tensor[group_tensor_index["x_center_screen"]] = cir_sc["center"][0]
    cir_tensor[group_tensor_index["y_center_screen"]] = cir_sc["center"][1]
    cir_tensor[group_tensor_index["screen_left_x"]] = 0
    cir_tensor[group_tensor_index["screen_left_y"]] = 0
    cir_tensor[group_tensor_index["screen_right_x"]] = 0
    cir_tensor[group_tensor_index["screen_right_y"]] = 0

    cir_tensor[group_tensor_index["axis_x"]] = cir["radius"]
    cir_tensor[group_tensor_index["axis_z"]] = cir["radius"]
    cir_tensor[group_tensor_index["screen_axis_x"]] = cir_sc["radius"]
    cir_tensor[group_tensor_index["screen_axis_z"]] = cir_sc["radius"]
    cir_tensor[group_tensor_index["size"]] = objs.shape[0]

    cir_tensor = cir_tensor.reshape(-1)

    return cir_tensor


def op_count_nonzeros(data, axis, epsilon):
    counter = (data / (data + epsilon)).sum(dim=axis)
    return counter


def to_conic_tensor(objs, conics, conics_sc, conic_error):
    group_tensor_index = config.group_tensor_index
    obj_tensor_index = config.obj_tensor_index
    conic_tensor = torch.zeros(len(group_tensor_index.keys()))

    colors = objs[:, [obj_tensor_index[i] for i in config.obj_color]]
    shapes = objs[:, [obj_tensor_index[i] for i in config.obj_shapes]]

    conic_tensor[group_tensor_index["x"]] = conics["center"][0]
    conic_tensor[group_tensor_index["y"]] = objs[:, 1].mean()
    conic_tensor[group_tensor_index["z"]] = conics["center"][1]

    conic_tensor[group_tensor_index["color_counter"]] = op_count_nonzeros(colors.sum(dim=0), axis=0, epsilon=1e-10)
    conic_tensor[group_tensor_index["shape_counter"]] = op_count_nonzeros(shapes.sum(dim=0), axis=0, epsilon=1e-10)

    colors_normalized = colors.sum(dim=0) / colors.shape[0]
    shapes_normalized = shapes.sum(dim=0) / shapes.shape[0]

    colors_normalized[colors_normalized < 0.99] = 0
    conic_tensor[group_tensor_index['red']] = colors_normalized[0]
    conic_tensor[group_tensor_index['green']] = colors_normalized[1]
    conic_tensor[group_tensor_index['blue']] = colors_normalized[2]

    shapes_normalized[shapes_normalized < 0.99] = 0
    conic_tensor[group_tensor_index['sphere']] = shapes_normalized[0]
    conic_tensor[group_tensor_index['cube']] = shapes_normalized[1]

    conic_tensor[group_tensor_index["line"]] = 0
    conic_tensor[group_tensor_index["circle"]] = 0
    conic_tensor[group_tensor_index["conic"]] = 1 - conic_error.sum() / conic_error.shape[0]

    conic_tensor[group_tensor_index["x_length"]] = objs[:, 0].max() - objs[:, 0].min()
    conic_tensor[group_tensor_index["y_length"]] = objs[:, 1].max() - objs[:, 1].min()
    conic_tensor[group_tensor_index["z_length"]] = objs[:, 2].max() - objs[:, 2].min()
    conic_tensor[group_tensor_index["x_center_screen"]] = conics_sc["center"][0]
    conic_tensor[group_tensor_index["y_center_screen"]] = conics_sc["center"][1]
    conic_tensor[group_tensor_index["screen_left_x"]] = 0
    conic_tensor[group_tensor_index["screen_left_y"]] = 0
    conic_tensor[group_tensor_index["screen_right_x"]] = 0
    conic_tensor[group_tensor_index["screen_right_y"]] = 0

    conic_tensor[group_tensor_index["axis_x"]] = conics["axis"][0]
    conic_tensor[group_tensor_index["axis_z"]] = conics["axis"][1]

    conic_tensor[group_tensor_index["screen_axis_x"]] = conics_sc["axis"][0]
    conic_tensor[group_tensor_index["screen_axis_z"]] = conics_sc["axis"][1]

    conic_tensor[group_tensor_index["size"]] = objs.shape[0]
    conic_tensor = conic_tensor.reshape(-1)

    return conic_tensor


def to_obj_tensor(objs):
    obj_tensor_index = config.obj_tensor_index
    group_tensor_index = config.group_tensor_index
    obj_tensor = torch.zeros(len(group_tensor_index.keys()))

    colors = objs[:, [obj_tensor_index[i] for i in config.obj_color]]
    shapes = objs[:, [obj_tensor_index[i] for i in config.obj_shapes]]

    colors_normalized = colors.sum(dim=0) / colors.shape[0]
    shapes_normalized = shapes.sum(dim=0) / shapes.shape[0]
    # 0:2 center_x, center_z
    # 2 slope
    # 3 x_length
    # 4 z_length
    # 5 is_line
    # 6 is_circle
    # 7 probability

    obj_tensor[group_tensor_index["x"]] = objs[:, 0].mean()
    obj_tensor[group_tensor_index["y"]] = objs[:, 1].mean()
    obj_tensor[group_tensor_index["z"]] = objs[:, 2].mean()

    obj_tensor[group_tensor_index["color_counter"]] = op_count_nonzeros(colors.sum(dim=0), axis=0, epsilon=1e-10)
    obj_tensor[group_tensor_index["shape_counter"]] = op_count_nonzeros(shapes.sum(dim=0), axis=0, epsilon=1e-10)

    colors_normalized[colors_normalized < 0.99] = 0
    obj_tensor[group_tensor_index['red']] = colors_normalized[0]
    obj_tensor[group_tensor_index['green']] = colors_normalized[1]
    obj_tensor[group_tensor_index['blue']] = colors_normalized[2]

    shapes_normalized[shapes_normalized < 0.99] = 0
    obj_tensor[group_tensor_index['sphere']] = shapes_normalized[0]
    obj_tensor[group_tensor_index['cube']] = shapes_normalized[1]
    obj_tensor[group_tensor_index['cone']] = shapes_normalized[2]
    obj_tensor[group_tensor_index['cylinder']] = shapes_normalized[3]
    obj_tensor[group_tensor_index["obj"]] = 1
    obj_tensor[group_tensor_index["line"]] = 0
    obj_tensor[group_tensor_index['circle']] = 0
    obj_tensor[group_tensor_index["x_length"]] = objs[:, 0].max() - objs[:, 0].min()
    obj_tensor[group_tensor_index["y_length"]] = objs[:, 1].max() - objs[:, 1].min()
    obj_tensor[group_tensor_index["z_length"]] = objs[:, 2].max() - objs[:, 2].min()

    obj_tensor[group_tensor_index["x_center_screen"]] = objs[:, obj_tensor_index['screen_x']]
    obj_tensor[group_tensor_index["y_center_screen"]] = objs[:, obj_tensor_index['screen_y']]

    obj_tensor[group_tensor_index["screen_left_x"]] = objs[:, obj_tensor_index['screen_x']]
    obj_tensor[group_tensor_index["screen_left_y"]] = objs[:, obj_tensor_index['screen_y']]
    obj_tensor[group_tensor_index["screen_right_x"]] = objs[:, obj_tensor_index['screen_x']]
    obj_tensor[group_tensor_index["screen_right_y"]] = objs[:, obj_tensor_index['screen_y']]

    obj_tensor[group_tensor_index["axis_x"]] = 0
    obj_tensor[group_tensor_index["axis_z"]] = 0
    obj_tensor[group_tensor_index["screen_axis_x"]] = 0
    obj_tensor[group_tensor_index["screen_axis_z"]] = 0

    obj_tensor[group_tensor_index["size"]] = objs.shape[0]

    obj_tensor = obj_tensor.reshape(-1)

    return obj_tensor
