# Created by shaji on 21-Apr-23
import torch

import aitk.utils.logic_utils
from aitk.utils import log_utils
from aitk.utils import data_utils

import config
import logic_utils
from aitk.utils.eval_utils import eval_data


def remove_duplicate_clauses(refs_i, unused_args, used_args, args):
    non_duplicate_c = []
    for clause in refs_i:
        is_duplicate = False
        for body in clause.body:
            if "in" != body.pred.name:
                if len(body.terms) == 2 and "O" not in body.terms[1].name:
                    # predicate with 1 object arg
                    if len(unused_args) > 0:
                        if not (body.terms[0] == unused_args[0] or body.terms[0] in used_args):
                            is_duplicate = True
                            break
                    # predicate with 2 object args
                elif len(body.terms) == 2 and body.terms[0] in unused_args and body.terms[1] in unused_args:
                    if body.terms[0] not in unused_args[:2] and body.terms[1] not in unused_args:
                        is_duplicate = True
                        break
                elif len(body.terms) == 1 and body.terms[0] in unused_args:
                    if body.terms[0] not in unused_args[:1]:
                        is_duplicate = True
                        break

        if not is_duplicate:
            non_duplicate_c.append(clause)
    return non_duplicate_c


# def remove_trivial_clauses(refs_non_conflict, args):
#     non_trivial_clauses = []
#     for ref in refs_non_conflict:
#         preds = get_pred_names_from_clauses(ref)
#
#         if not logic_utils.is_trivial_preds(preds):
#             non_trivial_clauses.append(ref)
#         # else:
#         #     log_utils.add_lines(f"(trivial clause) {ref}", args.log_file)
#     return non_trivial_clauses


def remove_conflict_clauses(clauses, pi_clauses, args):
    # print("\nCheck for conflict clauses...")
    clause_ordered = []
    non_conflict_clauses = []
    for clause in clauses:
        is_conflict = False
        with_pi = False
        if len(pi_clauses) > 0:
            for cb in clause.body:
                if "inv_pred" in cb.pred.name:
                    with_pi = True
            if not with_pi:
                is_conflict = False
        if with_pi or len(pi_clauses) == 0:
            for i in range(len(clause.body)):
                if is_conflict:
                    break
                for j in range(len(clause.body)):
                    if i == j:
                        continue
                    if "inv_pred" in clause.body[j].pred.name and not is_conflict:
                        pi_name = clause.body[j].pred.name
                        pi_bodies = logic_utils.get_pi_bodies_by_name(pi_clauses, pi_name)
                        is_conflict = logic_utils.is_conflict_bodies(pi_bodies, clause.body)
                        if is_conflict:
                            break
                    if "inv_pred" in clause.body[i].pred.name and not is_conflict:
                        pi_name = clause.body[i].pred.name
                        pi_bodies = logic_utils.get_pi_bodies_by_name(pi_clauses, pi_name)
                        is_conflict = logic_utils.is_conflict_bodies(pi_bodies, clause.body)
                        if is_conflict:
                            break
                    # if "at_are_6" in clause.body[i].pred.name or "at_are_6" in clause.body[j].pred.name:
                    #     print("conflict")

        if not is_conflict:
            non_conflict_clauses.append(clause)
        # else:
        #     log_utils.add_lines(f"(conflict clause) {clause}", args.log_file)

    return non_conflict_clauses


# def remove_same_semantic_clauses(clauses):
#     semantic_diff_clauses = []
#     for c in clauses:
#         c_equiv_cs = get_equivalent_clauses(c)
#         c.equiv_c_preds = []
#         for c_equiv in c_equiv_cs:
#             c_equiv_cs_preds = get_pred_names_from_clauses(c_equiv, exclude_objects=True)
#             if c_equiv_cs_preds not in c.equiv_c_preds:
#                 c.equiv_c_preds.append(c_equiv_cs_preds)
#
#     for c in clauses:
#         is_same = False
#         for added_c in semantic_diff_clauses:
#             c_preds = get_pred_names_from_clauses(c)
#             added_c_preds = get_pred_names_from_clauses(added_c)
#             if c_preds == added_c_preds:
#                 is_same = True
#                 break
#             elif semantic_same_pred_lists(added_c.equiv_c_preds, c.equiv_c_preds):
#                 is_same = True
#                 break
#         if not is_same:
#             semantic_diff_clauses.append(c)
#
#     return semantic_diff_clauses


def semantic_same_pred_lists(added_pred_list, new_pred_list):
    is_same = True
    for new_pred in new_pred_list:
        for added_pred in added_pred_list:
            if not new_pred == added_pred:
                is_same = False
                break
    if is_same:
        print("break")
    return is_same


def extract_pi(lang, all_pi_clauses, args):
    for index, new_p in enumerate(lang.invented_preds):
        if new_p in lang.invented_preds:
            continue
        is_duplicate = False
        for self_p in lang.invented_preds:
            if new_p.body == self_p.body:
                is_duplicate = True
                log_utils.add_lines(f"duplicate pi body {new_p.name} {new_p.body}", args.log_file)
                break
        if not is_duplicate:
            print(f"add new predicate: {new_p.name}")
            lang.invented_preds.append(new_p)
        else:
            log_utils.add_lines(f"duplicate pi: {new_p}", args.log_file)

    new_p_names = [self_p.name for self_p in lang.invented_preds]
    new_all_pi_clausese = []
    for pi_c in all_pi_clauses:
        pi_c_head_name = pi_c.head.pred.name
        if pi_c_head_name in new_p_names:
            new_all_pi_clausese.append(pi_c)
    return new_all_pi_clausese


def update_refs(clause_with_scores, args):
    refs = []
    nc_clauses = aitk.utils.logic_utils.extract_clauses_from_bs_clauses(clause_with_scores, "clause", args)
    refs += nc_clauses

    return refs


def eval_groups(args, img_i):
    log_utils.add_lines("- group evaluation", args.log_file)
    # extract indices

    shape_group_res = eval_single_group(args, config.group_group_shapes, "group", img_i)
    color_res = eval_single_group(args, config.group_color, "object", img_i)
    shape_res = eval_single_group(args, config.group_shapes, "object", img_i)

    result = {'shape_group': shape_group_res, 'color': color_res, 'shape': shape_res}

    # The patterns is too simple. Print the reason.
    # if False and is_done:
    # Dataset is too simple. Finish the program.
    # eval_result_test = eval_groups(test_pattern_pos, test_pattern_neg, clu_result)
    # is_done = check_group_result(args, eval_result_test)
    # log_utils.print_dataset_simple(args, is_done, eval_result_test)

    return result


def eval_single_group(args, props, g_type, img_i):
    """ evaluate single group prop on single image """
    indices = data_utils.prop2index(props, g_type)
    # extract data
    if g_type == "group":
        data_pos = torch.tensor(args.val_group_pos[img_i])[:, indices]
        data_neg = torch.tensor(args.val_group_pos[img_i])[:, indices]
    elif g_type == "object":
        data_pos = args.val_pos[img_i][:, indices]
        data_neg = args.val_neg[img_i][:, indices]
    else:
        raise ValueError

    # evaluation
    log_utils.add_lines(f"{props}", args.log_file)
    score_pos_1, score_pos_2 = eval_data(data_pos)
    score_neg_1, score_neg_2 = eval_data(data_neg)

    res_dict = {
        "score_pos_1": score_pos_1, "score_pos_2": score_pos_2,
        "score_neg_1": score_neg_1, "score_neg_2": score_neg_2
    }

    return res_dict


