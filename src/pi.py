import datetime
import torch

import config
from aitk.utils.fol import bk

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def generate_explain_pred(args, lang, atom_terms, unclear_pred):
    focused_obj_props = bk.pred_obj_mapping[unclear_pred.name]
    if focused_obj_props is None:
        return None
    # check if it is a clause with group term
    if not has_term(unclear_pred, 'group'):
        return None

    val_pos_obj = args.val_pos
    val_pos_group = args.val_group_pos
    val_pos_avail = args.obj_avail_val_pos
    group_objs_all_data = torch.zeros(
        size=(val_pos_group.shape[0], val_pos_group.shape[1], val_pos_obj.shape[1], val_pos_obj.shape[2]))

    focused_obj_prop_indices = [config.obj_tensor_index[prop] for prop in focused_obj_props]
    focused_obj_values = torch.zeros(size=(val_pos_obj.shape[0], val_pos_group.shape[1], len(focused_obj_prop_indices)))
    for image_i, all_objs in enumerate(val_pos_obj):
        for g_i in range(val_pos_avail.shape[1]):
            group_objs = all_objs[val_pos_avail[image_i][g_i]]
            if len(group_objs) == 0:
                continue
            group_objs_all_data[image_i, g_i, :group_objs.shape[0]] = group_objs
            objs_mean_value = group_objs.mean(dim=0)
            focused_obj_values[image_i, g_i] = objs_mean_value[focused_obj_prop_indices]

    # raw data for explanation
    min_value_set = find_minimum_common_values(focused_obj_values)

    # p_args = logic_utils.count_arity_from_clause_cluster(clause_cluster)
    dtypes = terms_to_dtypes(atom_terms)

    new_predicate = lang.inv_pred(args, arity=len(atom_terms), pi_dtypes=dtypes, p_args=atom_terms,
                                  pi_type=config.pi_type["exp"])

    # define atoms
    # extend the clause with new atoms
    new_predicate.obj_indices = focused_obj_prop_indices
    new_predicate.value_set = min_value_set

    return new_predicate

    # # to generate a set of new predicates, we need
    # # bk predicates (mapping from)
    # # convert min common indices to predicates
    #
    # NSFR = nsfr_utils.get_nsfr_model(args, lang, FC)
    # # evaluate new clauses
    # score_all = eval_clause_infer.eval_clause_on_scenes(NSFR, args, eval_pred)
    # scores = eval_clause_infer.eval_clauses(score_all[:, :, index_pos], score_all[:, :, index_neg], args, step)
    # # classify clauses
    # clause_with_scores = eval_clause_infer.prune_low_score_clauses(refs_extended, score_all, scores, args)
    # # print best clauses that have been found...
    # clause_with_scores = logic_utils.sorted_clauses(clause_with_scores, args)
    #
    # # explain the unclear predicates by extending with new predicates


def has_term(pred, term_name):
    for dtype in pred.dtypes:
        if dtype.name == term_name:
            return True
    return False


def find_minimum_common_values(focused_obj_values):
    minimum_set = []
    for indices in focused_obj_values:
        indices_list = indices.reshape(-1).tolist()
        if indices_list not in minimum_set:
            minimum_set.append(indices_list)

    return minimum_set


def terms_to_dtypes(terms):
    dtypes = []

    for term in terms:
        if "O" == term.name[0]:
            dtypes.append("group")
        elif "number" in term.name:
            dtypes.append("number")
        else:
            raise ValueError
    return dtypes
