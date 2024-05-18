# Created by jing at 09.06.23
import copy
import os
import torch

import config

from aitk.utils import eval_utils
from aitk.utils import data_utils
from aitk.utils import visual_utils
from aitk.utils import log_utils


def detect_groups(args, valid_obj_all, g_type):
    if g_type == "obj":
        init_obj_num = 1
        min_obj_num = 1
        extend_func = extend_obj_group
    elif g_type == "line":
        init_obj_num = 2
        min_obj_num = args.line_group_min_sz
        extend_func = extend_line_group
    elif g_type == "cir":
        init_obj_num = 3
        min_obj_num = args.cir_group_min_sz
        extend_func = extend_circle_group
    elif g_type == "conic":
        init_obj_num = 5
        min_obj_num = args.conic_group_min_sz
        extend_func = extend_conic_group
    else:
        raise ValueError

    all_group_indices = []
    all_group_tensors = []
    all_group_shape_data = []
    all_group_unfit_err = []

    # detect lines image by image
    for img_i in range(len(valid_obj_all)):
        group_indices_ith = []
        group_groups_ith = []
        group_shape_ith = []
        group_unfit_err_ith = []
        exist_combs = []
        img_i_obj_tensors = valid_obj_all[img_i]
        min_group_indices = data_utils.get_comb(torch.tensor(range(img_i_obj_tensors.shape[0])), init_obj_num).tolist()
        # detect lines by checking each possible combinations
        for g_i, obj_indices in enumerate(min_group_indices):
            # check duplicate
            if obj_indices in exist_combs:
                continue
            g_indices, g_shape_data, unfit_err = extend_func(args, obj_indices, img_i_obj_tensors)
            if g_shape_data is not None:
                g_obj_tensors = img_i_obj_tensors[g_indices]
                if g_obj_tensors is not None and len(g_obj_tensors) >= min_obj_num:
                    if not g_type == "obj":
                        exist_combs += data_utils.get_comb(g_indices, init_obj_num).tolist()
                    group_indices_ith.append(g_indices)
                    group_groups_ith.append(g_obj_tensors)
                    group_shape_ith.append(g_shape_data)
                    group_unfit_err_ith.append(unfit_err)
        all_group_indices.append(group_indices_ith)
        all_group_tensors.append(group_groups_ith)
        all_group_shape_data.append(group_shape_ith)
        all_group_unfit_err.append(group_unfit_err_ith)
    return all_group_indices, all_group_tensors, all_group_shape_data, all_group_unfit_err


def detect_groups_cuda(args, objs_all, g_type):
    if g_type == "obj":
        init_obj_num = 1
        min_obj_num = 1
        extend_func = extend_obj_group
    elif g_type == "line":
        init_obj_num = 2
        min_obj_num = args.line_group_min_sz
        extend_func = extend_line_group
    elif g_type == "cir":
        init_obj_num = 3
        min_obj_num = args.cir_group_min_sz
        extend_func = extend_circle_group
    elif g_type == "conic":
        init_obj_num = 5
        min_obj_num = args.conic_group_min_sz
        extend_func = extend_conic_group
    else:
        raise ValueError

    all_group_indices = []
    all_group_tensors = []
    all_group_shape_data = []
    all_group_unfit_err = []

    # detect lines image by image
    max_obj_per_img = max([img_obj.shape[0] for img_obj in objs_all])
    obj_tensor_len = len(config.obj_tensor_index)
    objs_all_tensors = torch.zeros((len(objs_all), max_obj_per_img, obj_tensor_len))

    # align object tensors for each image

    for img_i in range(len(objs_all)):
        if len(objs_all[img_i]) < max_obj_per_img:
            suppliment_tensors = torch.tensor([[0] * obj_tensor_len] * (max_obj_per_img - len(objs_all[img_i])))
            objs_all[img_i] = torch.cat((objs_all[img_i], suppliment_tensors), 0).tolist()
        else:
            objs_all[img_i] = objs_all[img_i].tolist()

    objs_all_tensors = torch.tensor(objs_all)
    min_group_indices = data_utils.get_comb(torch.tensor(range(max_obj_per_img)), init_obj_num).tolist()

    for min_indices in min_group_indices:
        g_indices, g_shape_data, unfit_err = extend_line_group_cuda(args, min_indices, objs_all_tensors)

    for img_i in range(len(objs_all)):
        group_indices_ith = []
        group_groups_ith = []
        group_shape_ith = []
        group_unfit_err_ith = []
        exist_combs = []
        img_i_obj_tensors = objs_all[img_i]
        min_group_indices = data_utils.get_comb(torch.tensor(range(img_i_obj_tensors.shape[0])), init_obj_num).tolist()
        # detect lines by checking each possible combinations
        for g_i, obj_indices in enumerate(min_group_indices):
            # check duplicate
            if obj_indices in exist_combs:
                continue
            g_indices, g_shape_data, unfit_err = extend_func(args, obj_indices, img_i_obj_tensors)
            if g_shape_data is not None:
                g_obj_tensors = img_i_obj_tensors[g_indices]
                if g_obj_tensors is not None and len(g_obj_tensors) >= min_obj_num:
                    if not g_type == "obj":
                        exist_combs += data_utils.get_comb(g_indices, init_obj_num).tolist()
                    group_indices_ith.append(g_indices)
                    group_groups_ith.append(g_obj_tensors)
                    group_shape_ith.append(g_shape_data)
                    group_unfit_err_ith.append(unfit_err)
        all_group_indices.append(group_indices_ith)
        all_group_tensors.append(group_groups_ith)
        all_group_shape_data.append(group_shape_ith)
        all_group_unfit_err.append(group_unfit_err_ith)
    return all_group_indices, all_group_tensors, all_group_shape_data, all_group_unfit_err


def detect_cir(args, obj_tensors):
    point_data = obj_tensors[:, :, config.indices_position]
    color_data = obj_tensors[:, :, config.indices_color]
    shape_data = obj_tensors[:, :, config.indices_shape]
    point_screen_data = obj_tensors[:, :, config.indices_screen_position]

    used_objs = torch.zeros(point_data.shape[0], args.group_e, point_data.shape[1], dtype=torch.bool)
    circle_tensors = torch.zeros(point_data.shape[0], args.e, len(config.group_tensor_index.keys()))
    for data_i in range(point_data.shape[0]):
        exist_combs = []
        tensor_counter = 0
        group_indices = data_utils.get_comb(torch.tensor(range(point_data[data_i].shape[0])), 3).tolist()
        circle_tensor_candidates = torch.zeros(args.e, len(config.group_tensor_index.keys()))
        groups_used_objs = torch.zeros(args.group_e, point_data.shape[1], dtype=torch.bool)

        for g_i, group_index in enumerate(group_indices):
            # check duplicate
            if group_index not in exist_combs:
                p_groups, p_indices, center, r = extend_circle_group(group_index, point_data[data_i], args)
                if p_groups is not None and p_groups.shape[0] >= args.cir_group_min_sz:
                    colors = color_data[data_i][p_indices]
                    shapes = shape_data[data_i][p_indices]
                    p_groups_screen = point_screen_data[data_i][p_indices]
                    circle_tensor = data_utils.to_circle_tensor(p_groups).reshape(1, -1)
                    circle_tensor_candidates = torch.cat([circle_tensor_candidates, circle_tensor], dim=0)

                    cir_used_objs = torch.zeros(1, point_data.shape[1], dtype=torch.bool)
                    cir_used_objs[0, p_indices] = True
                    groups_used_objs = torch.cat([groups_used_objs, cir_used_objs], dim=0)

                    exist_combs += data_utils.get_comb(p_indices, 3).tolist()
                    tensor_counter += 1
                    # print(f'circle group: {p_indices}')
        # print(f'\n')
        _, prob_indices = circle_tensor_candidates[:, config.group_tensor_index["circle"]].sort(descending=True)
        prob_indices = prob_indices[:args.e]
        circle_tensors[data_i] = circle_tensor_candidates[prob_indices]
        used_objs[data_i] = groups_used_objs[prob_indices]
    return circle_tensors, used_objs


def encode_groups(args, valid_obj_all, obj_indices, obj_tensors, group_type):
    """
    Encode groups to tensors.
    """

    obj_tensor_index = config.obj_tensor_index
    group_tensor = None
    all_group_tensors = []
    all_group_tensors_indices = []
    for img_i in range(len(obj_tensors)):
        img_group_tensors = []
        img_group_tensors_indices = []
        for obj_i, objs in enumerate(obj_tensors[img_i]):
            group_used_objs = torch.zeros(len(valid_obj_all[img_i]), dtype=torch.bool)
            if group_type == "obj":
                group_tensor = data_utils.to_obj_tensor(objs)
            elif group_type == "line":
                lines = eval_utils.fit_line(objs[:, [0, 2]])
                line_sc = eval_utils.fit_line(objs[:, [obj_tensor_index[i] for i in config.obj_screen_positions]])
                line_error = eval_utils.get_line_error(lines["slope"], lines["intercept"], objs[:, [0, 2]])
                group_tensor = data_utils.to_line_tensor(objs, line_sc, line_error)
            elif group_type == "cir":
                cir = eval_utils.fit_circle(objs[:, [0, 2]], args)
                cir_sc = eval_utils.fit_circle(objs[:, [obj_tensor_index[i] for i in config.obj_screen_positions]],
                                               args)
                if cir_sc is not None and cir_sc["center"] is not None:
                    cir_error = eval_utils.get_circle_error(cir["center"], cir["radius"], objs[:, [0, 2]])
                    if cir_error.sum() > 1:
                        print("break")
                    group_tensor = data_utils.to_circle_tensor(objs, cir, cir_sc, cir_error)
                else:
                    continue

            elif group_type == "conic":
                conics = eval_utils.fit_conic(objs[:, [0, 2]])
                conics_sc = eval_utils.fit_conic(objs[:, [obj_tensor_index[i] for i in config.obj_screen_positions]])
                if conics_sc['axis'] is not None:
                    conic_error = eval_utils.get_conic_error(conics["coef"], conics["center"], objs[:, [0, 2]])
                    group_tensor = data_utils.to_conic_tensor(objs, conics, conics_sc, conic_error)
                else:
                    continue
            else:
                raise ValueError

            group_used_objs[obj_indices[img_i][obj_i]] = True
            img_group_tensors.append(group_tensor.tolist())
            img_group_tensors_indices.append(group_used_objs.tolist())

        all_group_tensors.append(img_group_tensors)
        all_group_tensors_indices.append(img_group_tensors_indices)

    # for i in range(len(all_group_tensors_indices[0])):
    #     obj_tensors = torch.tensor(valid_obj_all[0])
    #     poly_group_indices = all_group_tensors_indices[0][i]
    #     group_objs = obj_tensors[poly_group_indices]
    #     conics = eval_utils.fit_conic(group_objs[:, [0, 2]])['coef']
    #     conics_2 = group_data[0][i]['coef']
    #     print(f"diff {i}: {conics_2 - conics}")

    return all_group_tensors, all_group_tensors_indices


def prune_groups(args, OBJ_N, groups_all, group_indices, group_data, unfit_error_all, g_type):
    """ sort and select the high scoring groups"""
    pruned_tensors = []
    pruned_tensors_indices = []
    pruned_data = []
    unfit_error_pruned = []
    for img_i in range(len(groups_all)):
        img_groups = torch.tensor(groups_all[img_i])
        img_group_indices = torch.tensor(group_indices[img_i])
        img_scores = []
        # align group number to max e
        if len(img_groups) >= args.group_max_e:

            # d = args.group_max_e - len(img_groups)
            # img_groups += [[0] * len(config.group_tensor_index)] * d
            # group_indices[img_i] += [False] * d
            # group_data[img_i] += [None] * d
            # group_error[img_i] += [None] * d
            # else:
            for group in img_groups:
                group_tensor = group
                group_scores = group_tensor[config.group_tensor_index[g_type]]
                img_scores.append(group_scores.tolist())
            img_scores = torch.tensor(img_scores)

            _, prob_indices = img_scores.sort(descending=True)
            img_groups = img_groups[prob_indices.tolist()]
            img_group_indices = img_group_indices[prob_indices.tolist()]
            group_data[img_i] = [group_data[img_i][ind] for ind in prob_indices.tolist()]
            unfit_error_all[img_i] = [unfit_error_all[img_i][ind] for ind in prob_indices.tolist()]
        # else:
        #     d = args.group_max_e - len(img_groups)
        #     img_groups = torch.cat((img_groups, torch.tensor([[0] * len(config.group_tensor_index)] * d)), 0)
        #     # img_group_indices = torch.cat((img_group_indices, torch.tensor([[False] * OBJ_N] * d)), 0)
        #     img_group_indices = img_group_indices.tolist() + [[False] * OBJ_N] * d
        #
        #     group_data[img_i] += [None] * d
        #     unfit_error_all[img_i] += [None] * d

        if len(img_groups) == 0:
            # no groups detected in the image
            # pruned_tensors.append(torch.zeros(size=(args.group_max_e, len(config.group_tensor_index))).tolist())
            pruned_tensors.append(None)
            pruned_tensors_indices.append(None)
            pruned_data.append(None)
            unfit_error_pruned.append(None)
        else:
            pruned_tensors.append(img_groups[:args.group_max_e].tolist())
            pruned_tensors_indices.append(img_group_indices[:args.group_max_e])
            pruned_data.append(group_data[img_i][:args.group_max_e])
            unfit_error_pruned.append(unfit_error_all[img_i][:args.group_max_e])

    return pruned_tensors, pruned_tensors_indices, pruned_data, unfit_error_pruned


def prune_cirs(args, cir_tensors, cir_tensors_indices):
    pass


# def visual_selected_points(args, g_indices, valid_obj_all):
#     for img_i in range(args.top_data):
#         # if no groups in this image, continue
#         if g_indices[img_i] is None:
#             continue
#
#         # visual image name
#         vis_file = args.analysis_output_path / f"{args.dataset}_img_{img_i}_groups.png"
#         for g_i, obj_indices in enumerate(g_indices[img_i]):
#
#
#             g_in_indices = g_indices[img_i][g_i]
#             g_in_objs = valid_obj_all[img_i][g_in_indices]
#
#             if isinstance(g_in_indices, list):
#                 indices_rest = list(set(list(range(len(valid_obj_all[img_i])))) - set(
#                     [i for i, e in enumerate(g_in_indices) if e == True]))
#             elif g_in_indices.dtype == torch.int64:
#                 indices_rest = list(set(list(range(len(valid_obj_all[img_i])))) - set(g_in_indices.tolist()))
#             elif g_in_indices.dtype == torch.bool:
#                 in_indices = ((g_in_indices == True).nonzero(as_tuple=True)[0])
#                 indices_rest = list(set(list(range(len(valid_obj_all[img_i])))) - set(in_indices.tolist()))
#             else:
#                 raise ValueError
#             g_out_objs = valid_obj_all[img_i][indices_rest]
#             g_out_objs = torch.tensor(
#                 [obj.tolist() for obj in g_out_objs if obj[config.obj_tensor_index['prob']] > 0.5])
#
#             visual_utils.visual_points(vis_file, g_in_objs, g_out_objs, show=False, save=True)

def visual_group_analysis(args, g_indices, valid_obj_all, g_type, g_shape_data, unfit_error_all):
    # detect lines image by image
    for img_i in range(len(g_indices)):
        if g_indices[img_i] is None:
            continue
        for g_i, obj_indices in enumerate(g_indices[img_i]):
            vis_file = args.analysis_output_path / f"{args.dataset}_img_{img_i}_{g_type}_g_{g_i}.png"
            data = g_shape_data[img_i][g_i]
            g_in_indices = g_indices[img_i][g_i]
            unfit_error = unfit_error_all[img_i][g_i]
            # if len(valid_obj_all[img_i]) != len(g_in_indices):
            #     continue
            g_in_objs = valid_obj_all[img_i][g_in_indices]

            if isinstance(g_in_indices, list):
                indices_rest = list(set(list(range(len(valid_obj_all[img_i])))) - set(
                    [i for i, e in enumerate(g_in_indices) if e == True]))
            elif g_in_indices.dtype == torch.int64:
                indices_rest = list(set(list(range(len(valid_obj_all[img_i])))) - set(g_in_indices.tolist()))
            elif g_in_indices.dtype == torch.bool:
                in_indices = ((g_in_indices == True).nonzero(as_tuple=True)[0])
                indices_rest = list(set(list(range(len(valid_obj_all[img_i])))) - set(in_indices.tolist()))
            else:
                raise ValueError
            g_out_objs = valid_obj_all[img_i][indices_rest]
            g_out_objs = torch.tensor(
                [obj.tolist() for obj in g_out_objs if obj[config.obj_tensor_index['prob']] > 0.5])

            if data is not None:
                visual_utils.visual_group(g_type, vis_file, data, g_in_objs, g_out_objs, unfit_error)


def detect_line_groups(args, valid_obj_all, visual):
    """
    input: object tensors
    output: group tensors
    """
    OBJ_N = len(valid_obj_all[0])

    if not args.line_group:
        return None, None
    # line_indices, line_groups, line_data, line_error = detect_groups_cuda(args, valid_obj_all, "line")
    line_indices, line_groups, line_data, line_error = detect_groups(args, valid_obj_all, "line")
    line_tensors, line_tensors_indices = encode_groups(args, valid_obj_all, line_indices, line_groups, "line")
    # logic: prune strategy
    g_tensors, indices, data, error = prune_groups(args, OBJ_N, line_tensors, line_tensors_indices, line_data,
                                                   line_error, "line")
    if visual:
        visual_group_analysis(args, indices, valid_obj_all, "line", data, error)
    return g_tensors, indices


def detect_conic_groups(args, valid_obj_all, is_visual):
    """
    input: object tensors
    output: group tensors
    """

    OBJ_N = len(valid_obj_all[0])

    if not args.conic_group:
        return None, None

    conic_obj_indices, conic_groups, conic_data, conic_error = detect_groups(args, valid_obj_all, "conic")

    # ################### Debug ####################
    # obj_tensors = torch.tensor(valid_obj_all[0])
    # poly_group_indices = conic_obj_indices[0][10]
    # group_objs = obj_tensors[poly_group_indices]
    # conics = eval_utils.fit_conic(group_objs[:, [0, 2]])
    #
    # conics = conic_data[0][0]
    # leaf_indices = torch.tensor(sorted(set(list(range(obj_tensors.shape[0]))) - set(
    #     ((poly_group_indices == True).nonzero(as_tuple=True)[0]).tolist())))
    # poly_error = eval_utils.get_conic_error(conics["coef"], conics["center"], group_objs[:, [0, 2]])
    # visual_utils.visual_conic("test.png", conics["coef"], conics["center"],
    #                           [group_objs, obj_tensors[leaf_indices]],
    #                           errors=poly_error, labels=["base", "rest"], labels_2="detect", show=True)
    #
    ########################################

    conic_tensors, conic_tensors_indices = encode_groups(args, valid_obj_all, conic_obj_indices,
                                                         conic_groups, "conic")
    # visual_group_analysis(args, conic_obj_indices, valid_obj_all, "conic", conic_data, conic_error)
    tensors, g_in_indices, data, error = prune_groups(args, OBJ_N, conic_tensors, conic_tensors_indices,
                                                      conic_data, conic_error, "conic")

    if is_visual:
        visual_group_analysis(args, g_in_indices, valid_obj_all, "conic", data, error)
    return tensors, g_in_indices


def detect_circle_groups(args, valid_obj_all, is_visual):
    """
    input: object tensors
    output: group tensors
    """
    OBJ_N = len(valid_obj_all[0])

    if not args.circle_group:
        return None, None
    cir_indices, cir_groups, cir_data, unfit_error_all = detect_groups(args, valid_obj_all, "cir")

    visual_group_analysis(args, cir_indices, valid_obj_all, "cir", cir_data, unfit_error_all)

    cir_tensors, cir_tensors_indices = encode_groups(args, valid_obj_all, cir_indices, cir_groups, "cir")
    tensors, indices, data, unfit_err_pruned = prune_groups(args, OBJ_N, cir_tensors, cir_tensors_indices, cir_data,
                                                            unfit_error_all, "circle")
    if is_visual:
        visual_group_analysis(args, indices, valid_obj_all, "cir", data, unfit_err_pruned)
    return tensors, indices


def extend_obj_group(args, obj_indices, obj_tensors):
    obj_group_index = obj_indices
    obj = obj_tensors[obj_group_index]
    obj_error = 0
    return obj_group_index, obj, obj_error


def extend_line_group(args, obj_indices, obj_tensors):
    colinearities = None
    # points = obj_tensors[:, :, config.indices_position]
    has_new_element = True
    line_groups = copy.deepcopy(obj_tensors)
    line_group_indices = copy.deepcopy(obj_indices)

    while has_new_element:
        line_objs = obj_tensors[line_group_indices]
        extra_index = torch.tensor(sorted(set(list(range(obj_tensors.shape[0]))) - set(line_group_indices)))
        if len(extra_index) == 0:
            break
        extra_objs = obj_tensors[extra_index]
        line_objs_extended = extend_groups(line_objs, extra_objs)
        tensor_index = config.group_tensor_index
        colinearities = eval_utils.calc_colinearity(line_objs_extended,
                                                    [tensor_index[i] for i in config.obj_positions])
        avg_distances_error, avg_dist = eval_utils.calc_avg_dist(line_objs_extended,
                                                                 [tensor_index[i] for i in config.obj_positions])

        is_line = colinearities < args.error_th
        is_even_dist = avg_distances_error < args.line_even_error
        is_valid_dist = avg_dist < args.avg_dist_penalty
        passed_indices = is_line * is_even_dist * is_valid_dist
        has_new_element = passed_indices.sum() > 0
        passed_objs = extra_objs[passed_indices]
        passed_indices = extra_index[passed_indices]
        line_groups = torch.cat([line_objs, passed_objs], dim=0)
        line_group_indices += passed_indices.tolist()

    line_group_indices = torch.tensor(line_group_indices)
    line = eval_utils.fit_line(obj_tensors[line_group_indices][:, [0, 2]])

    even_dist_error = eval_utils.even_dist_error_on_line(line_groups, line)
    if even_dist_error > args.line_even_error:
        line_group_indices = None
        line = None
        colinearities = None

    return line_group_indices, line, colinearities


def extend_line_group_cuda(args, obj_indices, obj_tensors):
    colinearities = None
    # points = obj_tensors[:, :, config.indices_position]
    has_new_element = True
    tensor_index = config.group_tensor_index
    line_groups = copy.deepcopy(obj_tensors)
    line_group_indices = copy.deepcopy(obj_indices)

    line_objs = obj_tensors[:, obj_indices]
    extra_index = torch.tensor(sorted(set(list(range(obj_tensors.shape[1]))) - set(obj_indices)))
    extra_objs = obj_tensors[:, extra_index]
    line_objs_extended = extend_groups_cuda(line_objs, extra_objs)

    pos_indices = [tensor_index[i] for i in config.obj_positions]
    colinearities = eval_utils.calc_colinearity_cuda(line_objs_extended, pos_indices)
    avg_distances = eval_utils.calc_avg_dist_cuda(line_objs_extended)

    is_line = torch.lt(colinearities, args.error_th)
    is_even_dist = torch.lt(avg_distances, args.line_even_error)
    passed_indices = is_line * is_even_dist
    # has_new_element = passed_indices.sum() > 0
    has_new_element = torch.gt(torch.sum(passed_indices, dim=-1), 0)

    extra_obj_scores = avg_distances * colinearities
    extra_obj_scores_sorted, extra_obj_scores_indices = torch.sort(extra_obj_scores, dim=-1)

    obj_add_num = 1
    passed_obj_th = 1e-4
    passed_obj_scores = extra_obj_scores_sorted[:, :obj_add_num]

    passed_obj_mask = torch.lt(passed_obj_scores, passed_obj_th)
    passed_obj_index = extra_obj_scores_indices[:, :obj_add_num]
    passed_objs = torch.tensor(
        [extra_objs[img_i][passed_obj_index[img_i]].tolist() for img_i in range(obj_tensors.shape[0])])

    # add passed objects to line groups
    line_objs = torch.cat([line_objs, passed_objs], dim=1)
    # add mask and index to corresponding groups

    passed_indices = extra_index[passed_indices]
    line_groups = torch.cat([line_objs, passed_objs], dim=0)
    line_group_indices += passed_indices.tolist()

    while has_new_element:
        line_objs = obj_tensors[line_group_indices]
        extra_index = torch.tensor(sorted(set(list(range(obj_tensors.shape[0]))) - set(line_group_indices)))
        if len(extra_index) == 0:
            break
        extra_objs = obj_tensors[extra_index]
        line_objs_extended = extend_groups(line_objs, extra_objs)
        tensor_index = config.group_tensor_index
        colinearities = eval_utils.calc_colinearity(line_objs_extended,
                                                    [tensor_index[i] for i in config.obj_positions])
        avg_distances = eval_utils.calc_avg_dist(line_objs_extended, [tensor_index[i] for i in config.obj_positions])

        is_line = colinearities < args.error_th
        is_even_dist = avg_distances < args.line_even_error
        passed_indices = is_line * is_even_dist
        has_new_element = passed_indices.sum() > 0
        passed_objs = extra_objs[passed_indices]
        passed_indices = extra_index[passed_indices]
        line_groups = torch.cat([line_objs, passed_objs], dim=0)
        line_group_indices += passed_indices.tolist()

    line_group_indices = torch.tensor(line_group_indices)
    line = eval_utils.fit_line(obj_tensors[line_group_indices][:, [0, 2]])

    even_dist_error = eval_utils.even_dist_error_on_line(line_groups, line)
    if even_dist_error > args.line_even_error:
        line_group_indices = None
        line = None
        colinearities = None

    return line_group_indices, line, colinearities


def extend_conic_group(args, group_indices, obj_tensors):
    passed_objs_error = None
    has_new_element = True
    group_indices_ = copy.deepcopy(group_indices)
    group_objs = obj_tensors[group_indices_]
    conics = eval_utils.fit_conic(group_objs[:, [0, 2]])
    if conics["axis"] is None:
        return None, None, None

    # extend the group
    while has_new_element and conics is not None:
        leaf_indices = torch.tensor(sorted(set(list(range(obj_tensors.shape[0]))) - set(group_indices_)))
        if len(leaf_indices) == 0:
            break

        leaf_objs = obj_tensors[leaf_indices]
        leaf_error = eval_utils.get_conic_error(conics["coef"], conics["center"], leaf_objs[:, [0, 2]])
        if leaf_error is None:
            break
        is_poly = leaf_error < args.poly_error_th
        has_new_element = is_poly.sum() > 0
        passed_leaf_objs = leaf_objs[is_poly]
        passed_leaf_indices = leaf_indices[is_poly]
        passed_objs_error = leaf_error[is_poly]

        group_objs = torch.cat([group_objs, passed_leaf_objs], dim=0)
        group_indices_ = group_indices_ + passed_leaf_indices.tolist()
        conics = eval_utils.fit_conic(group_objs[:, [0, 2]])
        combined_error = eval_utils.get_conic_error(conics["coef"], conics["center"], group_objs[:, [0, 2]])
        if conics["axis"] is None:
            return None, None, None

    group_indices_ = torch.tensor(group_indices_)

    # even_dist_error = eval_utils.even_dist_error_on_conic(group_objs, conics)
    # if even_dist_error > args.conic_even_error:
    #     poly_group_indices = None
    #     conics = None
    #     poly_error = None

    return group_indices_, conics, passed_objs_error


def extend_circle_group(args, obj_indices, obj_tensors):
    unfit_err = torch.tensor([])
    has_new_element = True
    # cir_groups = copy.deepcopy(obj_tensors)
    cir_group_indices = copy.deepcopy(obj_indices)

    group_objs = obj_tensors[cir_group_indices]
    cir = eval_utils.fit_circle(group_objs[:, [0, 2]], args)
    while has_new_element and cir is not None:
        leaf_indices = torch.tensor(sorted(set(list(range(obj_tensors.shape[0]))) - set(cir_group_indices)))
        if len(leaf_indices) == 0:
            break

        leaf_objs = obj_tensors[leaf_indices]
        cir_error = eval_utils.get_circle_error(cir["center"], cir["radius"], leaf_objs[:, [0, 2]])
        is_circle = cir_error < args.cir_error_th
        has_new_element = is_circle.sum() > 0
        passed_leaf_objs = leaf_objs[is_circle]
        passed_leaf_indices = leaf_indices[is_circle]
        group_objs = torch.cat([group_objs, passed_leaf_objs], dim=0)
        cir_group_indices += passed_leaf_indices.tolist()

        cir = eval_utils.fit_circle(group_objs[:, [0, 2]], args)

    cir_group_indices = torch.tensor(cir_group_indices)

    if cir is not None and cir["center"] is not None:
        # if points are not evenly distributed, return None
        even_dist_error = eval_utils.even_dist_error_on_cir(group_objs, cir)
        if even_dist_error > args.cir_even_error:
            return None, None, None

        unfit_indices = torch.tensor(sorted(set(list(range(obj_tensors.shape[0]))) - set(cir_group_indices.tolist())))
        if len(unfit_indices) > 0:
            unfit_err = eval_utils.get_circle_error(cir["center"], cir["radius"], obj_tensors[unfit_indices][:, [0, 2]])
    else:
        return None, None, None
    return cir_group_indices, cir, unfit_err

    # point_groups = obj_tensors[obj_indices]
    # c, r = eval_utils.calc_circles(point_groups, args.error_th)
    # if c is None or r is None:
    #     return None, None
    # extra_index = torch.tensor(sorted(set(list(range(obj_tensors.shape[0]))) - set(obj_indices)))
    # extra_points = obj_tensors[extra_index]
    #
    # group_error = eval_utils.get_circle_error(c, r, extra_points)
    # passed_points = extra_points[group_error < args.error_th]
    # if len(passed_points) == 0:
    #     return None, None
    # point_groups_extended = extend_groups(point_groups, passed_points)
    # group_distribution = eval_utils.get_group_distribution(point_groups_extended, c)
    # passed_indices = extra_index[group_error < args.error_th][group_distribution]
    # passed_points = extra_points[group_error < args.error_th][group_distribution]
    # point_groups_new = torch.cat([point_groups, passed_points], dim=0)
    # point_groups_indices_new = torch.cat([torch.tensor(obj_indices), passed_indices])
    # # if not is_even_distributed_points(args, point_groups_new, shape="circle"):
    # #     return None, None, None, None
    # cir_data = {
    #     "groups": point_groups_new,
    #     "centers": c,
    #     "radius": r
    # }
    # return cir_data, point_groups_indices_new


def extend_groups(group_objs, extend_objs):
    group_points_duplicate_all = group_objs.unsqueeze(0).repeat(extend_objs.shape[0], 1, 1)
    group_points_candidate = torch.cat([group_points_duplicate_all, extend_objs.unsqueeze(1)], dim=1)

    return group_points_candidate


def extend_groups_cuda(group_objs, extend_objs):
    group_points_duplicate_all = group_objs.unsqueeze(1).repeat(1, extend_objs.shape[1], 1, 1)
    group_points_candidate = torch.cat([group_points_duplicate_all, extend_objs.unsqueeze(2)], dim=2)

    return group_points_candidate


def detect_dot_groups(args, percept_dict, valid_obj_indices):
    point_data = percept_dict[:, valid_obj_indices, config.indices_position]
    color_data = percept_dict[:, valid_obj_indices, config.indices_color]
    shape_data = percept_dict[:, valid_obj_indices, config.indices_shape]
    point_screen_data = percept_dict[:, valid_obj_indices, config.indices_screen_position]
    dot_tensors = torch.zeros(point_data.shape[0], args.e, len(config.group_tensor_index.keys()))
    for data_i in range(point_data.shape[0]):
        exist_combs = []
        tensor_counter = 0
        group_indices = data_utils.get_comb(torch.tensor(range(point_data[data_i].shape[0])), 3).tolist()
        dot_tensor_candidates = torch.zeros(args.e, len(config.group_tensor_index.keys()))
        for g_i, group_index in enumerate(group_indices):
            # check duplicate
            if group_index not in exist_combs:
                p_groups, p_indices, center, r = extend_circle_group(group_index, point_data[data_i], args)
                if p_groups is not None and p_groups.shape[0] >= args.cir_group_min_sz:
                    colors = color_data[data_i][p_indices]
                    shapes = shape_data[data_i][p_indices]
                    p_groups_screen = point_screen_data[data_i][p_indices]
                    circle_tensor = data_utils.to_circle_tensor(p_groups).reshape(1, -1)
                    dot_tensor_candidates = torch.cat([dot_tensor_candidates, circle_tensor], dim=0)
                    exist_combs += data_utils.get_comb(p_indices, 3).tolist()
                    tensor_counter += 1
                    print(f'dot group: {p_indices}')
        print(f'\n')
        _, prob_indices = dot_tensor_candidates[:, config.group_tensor_index["circle"]].sort(descending=True)
        prob_indices = prob_indices[:args.e]
        dot_tensors[data_i] = dot_tensor_candidates[prob_indices]
    return dot_tensors


def gen_single_obj_groups(args, valid_obj_all, visual):
    """
    input: object tensors
    output: group tensors
    """
    OBJ_N = len(valid_obj_all[0])

    if not args.obj_group:
        return None, None

    obj_indices, obj_groups, obj_data, obj_error = detect_groups(args, valid_obj_all, "obj")
    line_tensors, line_tensors_indices = encode_groups(args, valid_obj_all, obj_indices, obj_groups, "obj")
    # logic: prune strategy
    g_tensors, indices, data, error = prune_groups(args, OBJ_N, line_tensors, line_tensors_indices, obj_data, obj_error,
                                                   "obj")
    if visual:
        visual_group_analysis(args, indices, valid_obj_all, "obj", data, error)
    return g_tensors, indices


def detect_obj_groups(args, percept_dict_single, data_type):
    log_utils.add_lines(f"- grouping {data_type} objects ...", args.log_file)
    save_path = config.buffer_path / args.dataset_type / args.dataset / "buffer_groups"
    save_file = save_path / f"{args.dataset}_group_res_{data_type}.pth.tar"
    if save_file.is_file() and args.re_eval_groups:
        os.remove(str(save_file))

    args.analysis_output_path = args.analysis_path / data_type
    is_visual = args.is_visual
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if os.path.exists(save_file):
        group_res = torch.load(save_file)
        group_tensors = group_res["tensors"]
        group_obj_index_tensors = group_res['used_objs']
    else:
        # valid_objs_all = []
        # for img_i in range(len(percept_dict_single)):
        #     # valid_objs = percept_dict_single[img_i][percept_dict_single[img_i, :, config.obj_tensor_index['prob']] > 0]
        #     valid_objs_all.append(percept_dict_single[img_i])
        single_tensors, single_tensors_indices = gen_single_obj_groups(args, percept_dict_single, visual=is_visual)
        line_tensors, line_tensors_indices = detect_line_groups(args, percept_dict_single, visual=is_visual)
        conic_tensors, conic_tensors_indices = detect_conic_groups(args, percept_dict_single, is_visual=is_visual)
        cir_tensors, cir_tensors_indices = detect_circle_groups(args, percept_dict_single, is_visual=is_visual)

        group_tensors, group_obj_index_tensors = merge_groups(args, percept_dict_single,
                                                              single_tensors, single_tensors_indices,
                                                              line_tensors, cir_tensors, conic_tensors,
                                                              line_tensors_indices, cir_tensors_indices,
                                                              conic_tensors_indices)
        group_tensors, group_obj_index_tensors = align_groups(args, group_tensors, group_obj_index_tensors)
        # visual_utils.visual_groups(args, group_tensors, percept_dict_single, group_obj_index_tensors, data_type)
        group_res = {"tensors": group_tensors, "used_objs": group_obj_index_tensors}
        torch.save(group_res, save_file)

    return group_tensors, group_obj_index_tensors,


def get_group_tree(args, obj_groups, group_by):
    """ root node corresponds the scene """
    """ Used for very complex scene """
    pos_groups, neg_groups = obj_groups[0], obj_groups[1]

    # cluster by color

    # cluster by shape

    # cluster by line and circle
    if args.neural_preds[group_by][0].name == 'group_shape':
        pos_groups = None

    return None


def test_groups(test_positive, test_negative, groups):
    accuracy = 0
    for i in range(test_positive.shape[0]):
        accuracy += test_groups_on_one_image(test_positive[i:i + 1], groups)
    for i in range(test_negative.shape[0]):
        neg_score = test_groups_on_one_image(test_negative[i:i + 1], groups)
        accuracy += 1 - neg_score
    accuracy = accuracy / (test_positive.shape[0] + test_negative.shape[0])
    print(f"test acc: {accuracy}")
    return accuracy


def test_groups_on_one_image(data, val_result):
    test_score = 1
    score_color_1, score_color_2 = eval_utils.eval_data(data[:, :, config.indices_color])
    eval_shape_positive_res, score_shape_1, score_shape_2 = eval_utils.eval_data(data[:, :, config.indices_shape])

    return test_score


def select_top_k_groups(args, object_groups, obj_indices):
    # obj_indices = []
    # for i in range(len(used_objs[0])):
    #     obj_indices_img = []
    #     for shape_i in range(len(used_objs)):
    #         if used_objs[shape_i][i] is not None:
    #             obj_indices_img += used_objs[shape_i][i]
    #     obj_indices.append(obj_indices_img)
    obj_groups_top = []
    obj_groups_top_indices = []

    for img_i in range(len(object_groups)):
        # TODO: deal with group number less than group_e
        actual_group_num = len(object_groups[img_i])

        if actual_group_num < args.group_e:
            objs_max_selection = []
            groups = []
        else:
            group_indices = data_utils.get_comb(torch.tensor(range(actual_group_num)), args.group_e)
            groups = object_groups[img_i]
            # select groups including as many objects as possible
            objs_max_selection = group_indices[0]
            used_objs_max = 0
            for group_index in group_indices:
                if len(obj_indices[img_i]) > 0:
                    used_obj_in_img_group = [obj_indices[img_i][g_i] for g_i in group_index]
                    # comb_used_objs = torch.sum(used_obj_in_img_group, dim=0)
                    obj_num = sum(
                        [used_obj_in_img_group[g_i].sum().tolist() for g_i in range(len(used_obj_in_img_group))])
                    if obj_num > used_objs_max:
                        objs_max_selection = group_index
                        used_objs_max = obj_num

        obj_groups_img_top = [groups[_g_i] for _g_i in objs_max_selection[:args.group_e]]
        # obj_groups_img_top_indices = used_objs[img_i, objs_max_selection.tolist()[:args.group_e]]
        # obj_groups_img_top_indices = obj_indices[img_i][objs_max_selection]
        obj_groups_img_top_indices = [obj_indices[img_i][g_i].tolist() for g_i in objs_max_selection]

        obj_groups_top.append(obj_groups_img_top)
        obj_groups_top_indices.append(obj_groups_img_top_indices)

        # log
        for g_i, obj_group in enumerate(obj_groups_img_top):
            if obj_group[config.group_tensor_index["circle"]] != 0:
                group_name = "circle"
                group_msg = f""
            elif obj_group[config.group_tensor_index["line"]] != 0:
                group_name = "line"
                group_msg = f""
            elif obj_group[config.group_tensor_index["conic"]] != 0:
                group_name = "conic"
                axis_a = obj_group[config.group_tensor_index["screen_axis_x"]]
                axis_b = obj_group[config.group_tensor_index["screen_axis_z"]]
                group_msg = f"axis a: {axis_a}, axis b: {axis_b}"
            else:
                group_name = "unknown"
                group_msg = f""
            group_obj_indices = ((torch.tensor(obj_groups_img_top_indices[g_i]) == True).nonzero(as_tuple=True)[0])
            if args.show_process:
                print(f'(img {img_i}) {group_name} {group_obj_indices} {group_msg}')

    return obj_groups_top, obj_groups_top_indices


def merge_groups(args, percept_dict_single, single_groups, single_used_objs,
                 line_groups, cir_groups, conic_groups, line_used_objs, cir_used_objs, conic_used_objs):
    final_groups = []
    final_used_objs = []

    if single_groups is not None:
        final_groups.append(single_groups)
        final_used_objs.append(single_used_objs)

    if line_groups is not None and line_groups[0] is not None:
        final_groups.append(line_groups)
        final_used_objs.append(line_used_objs)

    if cir_groups is not None and cir_groups[0] is not None:
        final_groups.append(cir_groups)
        final_used_objs.append(cir_used_objs)

    if conic_groups is not None and conic_groups[0] is not None:
        final_groups.append(conic_groups)
        final_used_objs.append(conic_used_objs)

    # object_groups = torch.cat(final_groups, dim=1)

    groups_all = []
    group_indices = []
    if len(final_groups) == 0:
        return [], []
    for img_i in range(len(final_groups[0])):
        img_groups = []
        img_group_indices = []
        for type_i in range(len(final_groups)):
            if final_groups[type_i][img_i] is not None:
                img_groups += final_groups[type_i][img_i]
                img_group_indices += final_used_objs[type_i][img_i]
        groups_all.append(img_groups)
        group_indices.append(img_group_indices)

    # select top-k groups
    top_k_groups, top_k_group_indices = select_top_k_groups(args, groups_all, group_indices)
    # visualize selected groups
    if args.is_visual:
        visual_utils.visual_selected_groups(args, top_k_group_indices, percept_dict_single)
        # visual_selected_points(args, top_k_group_indices, percept_dict_single)

    return top_k_groups, top_k_group_indices


def align_groups(args, group_tensors, group_obj_index_tensors):
    for i in range(len(group_tensors)):
        if len(group_tensors[i]) < args.group_e:
            diff_num = args.group_e - len(group_tensors[i])
            group_tensors[i] += diff_num * [[0] * len(config.group_tensor_index)]
            group_obj_index_tensors[i] += diff_num * [None]
    return group_tensors, group_obj_index_tensors
