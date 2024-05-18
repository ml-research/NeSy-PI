import numpy as np
import torch
import itertools


import config
import ilp

import aitk.utils.fol.language
from aitk.utils import log_utils
from aitk.utils.fol.data_utils import DataUtils
from aitk.infer import ClauseInferModule
from aitk.tensor_encoder import TensorEncoder
from aitk.utils.fol import logic


def build_pi_clause_infer_module(args, clauses, pi_clauses, atoms, lang, device, m=3, infer_step=3, train=False):
    te = TensorEncoder(lang, atoms, clauses, device=device)
    I = te.encode()

    te_bk = None
    I_bk = None

    te_pi = None
    I_pi = None
    if len(pi_clauses) > 0:
        te_pi = TensorEncoder(lang, atoms, pi_clauses, device=device)
        I_pi = te_pi.encode()

    im = ClauseInferModule(I, m=m, infer_step=infer_step, device=device, train=train, I_bk=I_bk, I_pi=I_pi)
    return im


def generate_atoms(lang):
    spec_atoms = [aitk.utils.fol.language.false, aitk.utils.fol.language.true]
    atoms = []
    for pred in lang.preds:
        dtypes = pred.dtypes
        consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
        args_list = list(set(itertools.product(*consts_list)))
        for args in args_list:
            if len(args) == 1 or len(set(args)) == len(args):
                atoms.append(logic.Atom(pred, args))
    pi_atoms = []
    for pred in lang.invented_preds:
        dtypes = pred.dtypes
        consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
        args_list = list(set(itertools.product(*consts_list)))
        for args in args_list:
            if len(args) == 1 or len(set(args)) == len(args):
                pi_atoms.append(logic.Atom(pred, args))
    bk_pi_atoms = []
    for pred in lang.bk_inv_preds:
        dtypes = pred.dtypes
        consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
        args_list = list(set(itertools.product(*consts_list)))
        for args in args_list:
            # check if args and pred correspond are in the same area
            if pred.dtypes[0].name == 'area':
                if pred.name[0] + pred.name[5:] != args[0].name:
                    continue
            if len(args) == 1 or len(set(args)) == len(args):
                pi_atoms.append(logic.Atom(pred, args))
    return spec_atoms + sorted(atoms) + sorted(pi_atoms) + sorted(bk_pi_atoms)


def generate_bk(lang):
    atoms = []
    for pred in lang.preds:
        if pred.name in ['diff_color', 'diff_shape']:
            dtypes = pred.dtypes
            consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
            args_list = itertools.product(*consts_list)
            for args in args_list:
                if len(args) == 1 or (args[0] != args[1] and args[0].mode == args[1].mode):
                    atoms.append(logic.Atom(pred, args))
    return atoms


def parse_clauses(lang, clause_strs):
    du = DataUtils(lang)
    return [du.parse_clause(c) for c in clause_strs]


def get_pi_bodies_by_name(pi_clauses, pi_name):
    pi_bodies_all = []
    # name_change_dict = {"A": "O1", "B": "O2", "X": "X", "O2": "O2", "O1": "O1"}
    for pi_c in pi_clauses:
        if pi_name == pi_c.head.pred.name:
            pi_bodies = []
            for b in pi_c.body:
                p_name = b.pred.name
                if "inv_pred" in p_name:
                    body_names = get_pi_bodies_by_name(pi_clauses, p_name)
                    pi_bodies += body_names
                else:
                    # b.terms[0].name = name_change_dict[b.terms[0].name]
                    # b.terms[1].name = name_change_dict[b.terms[1].name]
                    pi_bodies.append(b)

            pi_bodies_all += pi_bodies

    return pi_bodies_all


def change_pi_body_names(pi_clauses):
    name_change_dict = {"A": "O1", "B": "O2", "X": "X"}
    for pi_c in pi_clauses:
        pi_c.head.terms[0] = name_change_dict[pi_c.head.terms[0].name]
        if len(pi_c.head.terms) > 1:
            pi_c.head.terms[1] = name_change_dict[pi_c.head.terms[1].name]
        for b in pi_c.body:
            b.terms[0].name = name_change_dict[b.terms[0].name]
            if len(b.terms) > 1:
                b.terms[1].name = name_change_dict[b.terms[1].name]

    return pi_clauses


def conflict_pred(p1, p2, t1, t2):
    non_confliect_dict = {
        "at_area_0": ["at_area_2"],
        "at_area_1": ["at_area_3"],
        "at_area_2": ["at_area_0"],
        "at_area_3": ["at_area_1"],
        "at_area_4": ["at_area_6"],
        "at_area_5": ["at_area_7"],
        "at_area_6": ["at_area_4"],
        "at_area_7": ["at_area_5"],
    }
    if p1 in non_confliect_dict.keys():
        if "at_area" in p2 and p2 not in non_confliect_dict[p1]:
            if t1[0] == t2[1] and t2[0] == t1[1]:
                return True
    return False


def is_repeat_clu(clu, clu_list):
    is_repeat = False
    for ref_clu, clu_score in clu_list:
        if clu == ref_clu:
            is_repeat = True
            break
    return is_repeat


def sub_clause_of(clause_a, clause_b):
    """
    Check if clause a is a sub-clause of clause b
    Args:
        clause_a:
        clause_b:

    Returns:

    """
    for body_a in clause_a.body:
        if body_a not in clause_b.body:
            return False

    return True


def eval_clause_clusters(clause_clusters, clause_scores_full):
    """
    Scoring each clause cluster, ranking them, return them.
    Args:
        clause_clusters:
        clause_scores_full:

    Returns:

    """
    cluster_candidates = []
    for c_index, clause_cluster in enumerate(clause_clusters):
        clause = list(clause_cluster.keys())[0]
        clause_full_score = clause_scores_full[c_index]
        total_score = np.sum(clause_full_score)
        complementary_clauses = [list(ic_dict.keys())[0] for ic_dict in list(clause_cluster.values())[0]]
        complementary_clauses_index = [list(ic_dict.values())[0] for ic_dict in list(clause_cluster.values())[0]]
        complementary_clauses_full_score = [clause_scores_full[index] for index in complementary_clauses_index]

        complementary_clauses_full_score_no_repeat = []
        complementary_clauses_no_repeat = []
        for i, full_scores_i in enumerate(complementary_clauses_full_score):
            for j, full_scores_j in enumerate(complementary_clauses_full_score):
                if i == j:
                    continue
                if full_scores_j == full_scores_i:
                    if full_scores_j in complementary_clauses_full_score_no_repeat:
                        continue
                    else:
                        complementary_clauses_full_score_no_repeat.append(full_scores_j)
                        complementary_clauses_no_repeat.append(complementary_clauses[j])

        sum_pos_clause = clause_full_score[1] + clause_full_score[3]
        sum_pos_clause_ind = [fs[1] + fs[3] for fs in complementary_clauses_full_score_no_repeat]

        if (sum_pos_clause + np.sum(sum_pos_clause_ind)) == total_score:
            cluster_candidates.append(clause_cluster)
        elif (sum_pos_clause + np.sum(sum_pos_clause_ind)) >= total_score:
            sum_ind = total_score - sum_pos_clause
            # find a subset of sum_pos_clause_ind, so that the sum of the subset equal to sum_ind
            sub_sets_candidates = []
            subsets = []
            for i in range(0, len(sum_pos_clause_ind) + 1):  # to get all lengths: 0 to 3
                for subset in itertools.combinations(sum_pos_clause_ind, i):
                    subsets.append(subset)
            sum_pos_clause_ind_index = [i for i in range(len(sum_pos_clause_ind))]
            subset_index = []
            for i in range(0, len(sum_pos_clause_ind_index) + 1):  # to get all lengths: 0 to 3
                for subset in itertools.combinations(sum_pos_clause_ind_index, i):
                    subset_index.append(subset)

            for subset_i, subset in enumerate(subsets):
                if np.sum(list(subset)) == sum_ind:
                    indices = subset_index[subset_i]
                    clauses_set = [clause_cluster[clause][c_i] for c_i in indices]
                    cluster_candidates.append({
                        "clause": clause, "clause_score": sum_pos_clause, "clause_set": clauses_set,
                        "clause_set_score": subset,
                    })
    new_pi_clauses = []

    for cluster in cluster_candidates:
        cluster_ind = [list(c.keys())[0] for c in cluster["clause_set"]]
        new_pi_clauses.append([cluster["clause"]] + cluster_ind)
    return new_pi_clauses


# def get_four_scores(predicate_scores):
#     return eval_clause_sign(predicate_scores)[0][1]


def eval_predicates_slow(NSFR, args, pred_names, pos_pred, neg_pred):
    bz = args.batch_size
    device = args.device
    pos_img_num = pos_pred.shape[0]
    neg_img_num = neg_pred.shape[0]
    eval_pred_num = len(pred_names)
    clause_num = len(NSFR.clauses)
    score_positive = torch.zeros((bz, pos_img_num, clause_num, eval_pred_num)).to(device)
    score_negative = torch.zeros((bz, neg_img_num, clause_num, eval_pred_num)).to(device)
    # get predicates that need to be evaluated.
    # pred_names = ['kp']
    # for pi_c in pi_clauses:
    #     for body_atom in pi_c.body:
    #         if "inv_pred" in body_atom.pred.name:
    #             pred_names.append(body_atom.pred.name)

    for image_index in range(pos_img_num):
        V_T_list = NSFR.clause_eval_quick(pos_pred[image_index].unsqueeze(0)).detach()
        A = V_T_list.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG

        C_score = torch.zeros((bz, clause_num, eval_pred_num)).to(device)
        # clause loop
        for clause_index, V_T in enumerate(V_T_list):
            for pred_index, pred_name in enumerate(pred_names):
                predicted = ilp.ilp_predict(v=V_T_list[clause_index, 0:1, :], predname=pred_name).detach()
                C_score[:, clause_index, pred_index] = predicted
        # sum over positive prob
        score_positive[:, image_index, :] = C_score

    # negative image loop
    for image_index in range(neg_img_num):
        V_T_list = NSFR.clause_eval_quick(neg_pred[image_index].unsqueeze(0)).detach()

        C_score = torch.zeros((bz, clause_num, eval_pred_num)).to(device)
        for clause_index, V_T in enumerate(V_T_list):
            for pred_index, pred_name in enumerate(pred_names):
                predicted = ilp.ilp_predict(v=V_T_list[clause_index, 0:1, :], predname=pred_name).detach()
                C_score[:, clause_index, pred_index] = predicted
            # C
            # C_score = PI.clause_eval(C_score)
            # sum over positive prob
        score_negative[:, image_index, :] = C_score

    # axis: batch_size, pred_names, pos_neg_labels, clauses, images
    score_positive = score_positive.permute(0, 3, 2, 1).unsqueeze(2)
    score_negative = score_negative.permute(0, 3, 2, 1).unsqueeze(2)
    all_predicates_scores = torch.cat((score_negative, score_positive), 2)

    return all_predicates_scores


def eval_predicates_sign(c_score):
    resolution = 2

    clause_sign_list = []
    clause_high_scores = []
    clause_score_full_list = []
    for clause_image_score in c_score:
        data_map = np.zeros(shape=[resolution, resolution])
        for index in range(len(clause_image_score[0][0])):
            x_index = int(clause_image_score[0][0][index] * resolution)
            y_index = int(clause_image_score[1][0][index] * resolution)
            data_map[x_index, y_index] += 1

        pos_low_neg_low_area = data_map[0, 0]
        pos_high_neg_low_area = data_map[0, 1]
        pos_low_neg_high_area = data_map[1, 0]
        pos_high_neg_high_area = data_map[1, 1]

        # TODO: find a better score evaluation function
        clause_score = pos_high_neg_low_area + pos_high_neg_high_area * 0.8
        clause_high_scores.append(clause_score)
        clause_score_full_list.append(
            [pos_low_neg_low_area, pos_high_neg_low_area, pos_low_neg_high_area, pos_high_neg_high_area])

        data_map[0, 0] = 0
        if np.max(data_map) == data_map[0, 1] and data_map[0, 1] > data_map[1, 1]:
            clause_sign_list.append(True)
        else:
            clause_sign_list.append(False)

    return clause_sign_list, clause_high_scores, clause_score_full_list


def check_repeat_conflict(atom1, atom2):
    if atom1.terms[0].name == atom2.terms[0].name and atom1.terms[1].name == atom2.terms[1].name:
        return True
    if atom1.terms[0].name == atom2.terms[1].name and atom1.terms[1].name == atom2.terms[0].name:
        return True
    return False


def is_conflict_bodies(pi_bodies, clause_bodies):
    is_conflict = False
    # check for pi_bodies confliction
    # for i, bs_1 in enumerate(pi_bodies):
    #     for j, bs_2 in enumerate(pi_bodies):
    #         if i == j:
    #             continue
    #         is_conflict = check_conflict_body(bs_1, bs_2)
    #         if is_conflict:
    #             return True

    # check for pi_bodies and clause_bodies confliction
    for i, p_b in enumerate(pi_bodies):
        for j, c_b in enumerate(clause_bodies):
            if p_b == c_b and p_b.pred.name != "in":
                is_conflict = True
            elif p_b.pred.name == c_b.pred.name:
                if p_b.pred.name == "rho":
                    is_conflict = check_repeat_conflict(p_b, c_b)
                elif p_b.pred.name == "phi":
                    is_conflict = check_repeat_conflict(p_b, c_b)
            if is_conflict:
                return True
            # if "at_area" in p_b.pred.name and "at_area" in c_b.pred.name:
            #     if p_b.terms == c_b.terms:
            #         return True
            #     elif conflict_pred(p_b.pred.name,
            #                        c_b.pred.name,
            #                        list(p_b.terms),
            #                        list(c_b.terms)):
            #         return True

    return False


def check_conflict_body(b1, b2):
    if "phi" in b1.pred.name and "phi" in b2.pred.name:
        if list(b1.terms) == list(b2.terms):
            return True
        elif conflict_pred(b1.pred.name,
                           b2.pred.name,
                           list(b1.terms),
                           list(b2.terms)):
            return True
    return False


def get_inv_body_preds(inv_body):
    preds = []
    for atom_list in inv_body:
        for atom in atom_list:
            preds.append(atom.pred.name)
            for t in atom.terms:
                if "O" not in t.name:
                    preds.append(t.name)
    return preds


def remove_duplicate_predicates(new_predicates, args):
    non_duplicate_pred = []
    for a_i, [p_a, a_score] in enumerate(new_predicates):
        is_duplicate = False
        for b_i, [p_b, b_score] in enumerate(new_predicates[a_i + 1:]):
            if p_a.name == p_b.name:
                continue
            p_a.body.sort()
            p_b.body.sort()
            p_a_body_preds = get_inv_body_preds(p_a.body)
            p_b_body_preds = get_inv_body_preds(p_b.body)
            if p_a_body_preds == p_b_body_preds:
                is_duplicate = True
        if not is_duplicate:
            non_duplicate_pred.append([p_a, a_score])
        # else:
        #     log_utils.add_lines(f"(remove duplicate predicate) {p_a} {a_score}", args.log_file)
    return non_duplicate_pred


def remove_unaligned_predicates(new_predicates):
    non_duplicate_pred = []
    for a_i, [p_a, p_score] in enumerate(new_predicates):
        b_lens = [len(b) - len(p_a.body[0]) for b in p_a.body]
        if sum(b_lens) == 0:
            non_duplicate_pred.append([p_a, p_score])
    return non_duplicate_pred


def remove_extended_clauses(clauses, p_score):
    clauses = list(clauses)
    non_duplicate_pred = []
    long_clauses_indices = []
    for a_i, c_a in enumerate(clauses):
        for b_i, c_b in enumerate(clauses):
            if a_i == b_i:
                continue
            if set(c_a.body) <= set(c_b.body):
                if b_i not in long_clauses_indices:
                    long_clauses_indices.append(b_i)

    short_clauses_indices = list(set([i for i in range(len(clauses))]) - set(long_clauses_indices))
    clauses = [clauses[i] for i in short_clauses_indices]
    # p_score = [p_score[i] for i in short_clauses_indices]

    return set(clauses)


def get_terms_from_atom(atom):
    terms = []
    for t in atom.terms:
        if t.name not in terms and "O" in t.name:
            terms.append(t.name)
    return terms


def check_accuracy(clause_scores_full, pair_num):
    accuracy = clause_scores_full[:, 1] / pair_num

    return accuracy


def get_best_clauses(clauses, clause_scores, step, args, max_clause):
    target_has_been_found = False
    higher = False
    # clause_accuracy = check_accuracy(clause_scores, total_score)
    sn_scores = clause_scores[config.score_type_index["sn"]].to("cpu")
    if sn_scores.max() == 1.0:
        log_utils.add_lines(f"(BS Step {step}) max clause accuracy: {sn_scores.max()}", args.log_file)
        target_has_been_found = True
        c_indices = [np.argmax(sn_scores)]
        for c_i in c_indices:
            log_utils.add_lines(f"{clauses[c_i]}", args.log_file)
    else:
        new_max_score = sn_scores.max()
        c_indices = [np.argmax(sn_scores)]
        # if len(clauses) != len(clause_scores):
        #     print("break")
        max_scoring_clauses = [[clauses[c_i], clause_scores[:, c_i]] for c_i in c_indices]
        new_max_clause = [new_max_score, max_scoring_clauses]

        if new_max_clause[0] > max_clause[0] and str(new_max_clause[1]) != str(max_clause[1]):
            max_clause = new_max_clause
            higher = True
            log_utils.add_lines(f"(BS Step {step}) (global) max clause accuracy: {sn_scores.max()}",
                                args.log_file)
            for c_i in c_indices:
                log_utils.add_lines(f"{clauses[c_i]}, {clause_scores[:, c_i]}", args.log_file)

        else:
            max_clause = [0.0, []]
            log_utils.add_lines(f"(BS Step {step}) (local) max clause accuracy: {sn_scores.max()}", args.log_file)
            for c_i in c_indices:
                log_utils.add_lines(f"{clauses[c_i]}, {clause_scores[:, c_i]}", args.log_file)
    return max_clause, higher


# def select_top_x_clauses(clause_candidates, c_type, args, threshold=None):
#     top_clauses_with_scores = []
#     clause_candidates_with_scores_sorted = []
#     if threshold is None:
#         top_clauses_with_scores = clause_candidates
#     else:
#         clause_candidates_with_scores = []
#         for c_i, c in enumerate(clause_candidates):
#             four_scores = get_four_scores(clause_candidates[c_i][1].unsqueeze(0))
#             clause_candidates_with_scores.append([c, four_scores])
#         clause_candidates_with_scores_sorted = sorted(clause_candidates_with_scores, key=lambda x: x[1][0][1],
#                                                       reverse=True)
#         clause_candidates_with_scores_sorted = clause_candidates_with_scores_sorted[:threshold]
#         for c in clause_candidates_with_scores_sorted:
#             top_clauses_with_scores.append(c[0])
#     for t_i, t in enumerate(top_clauses_with_scores):
#         log_utils.add_lines(f'TOP {(c_type)} {t[0]}, {clause_candidates_with_scores_sorted[t_i][1]}', args.log_file)
#
#     return top_clauses_with_scores


def is_trivial_preds(preds_terms):
    term_0 = preds_terms[0][1]
    for [pred, terms] in preds_terms:
        if terms != term_0:
            return False
    preds = [pt[0] for pt in preds_terms]

    # for trivial_set in config.trivial_preds_dict:
    #     is_trivial = True
    #     for pred in trivial_set:
    #         if pred not in preds:
    #             is_trivial = False
    #     if is_trivial:
    #         return True

    return False


def remove_3_zone_only_predicates(new_predicates, args):
    passed_predicates = []
    if len(new_predicates) > 0:
        if len(new_predicates[0]) == 0:
            return []
    for predicate in new_predicates:
        if torch.sum(predicate[1][:3]) > 0:
            passed_predicates.append(predicate)
        # else:
        #     log_utils.add_lines(f"(remove 3 zone only predicate) {predicate[0]} {predicate[1]}", args.log_file)
    return passed_predicates


def keep_1_zone_max_predicates(new_predicates):
    passed_predicates = []
    for predicate in new_predicates:
        if torch.max(predicate[1]) == predicate[1][1]:
            passed_predicates.append(predicate)
    return passed_predicates


def remove_same_four_score_predicates(new_predicates, args):
    passed_predicates = []
    passed_scores = []
    for predicate in new_predicates:
        if predicate[1].tolist() not in passed_scores:
            passed_scores.append(predicate[1].tolist())
            passed_predicates.append(predicate)
        # else:
        #     log_utils.add_lines(f"(remove same four score predicate) {predicate[0]} {predicate[1]}", args.log_file)
    return passed_predicates


def get_clause_unused_args(clause):
    used_args = []
    unused_args = []
    all_args = []
    for body in clause.body:
        if body.pred.name == "in":
            for term in body.terms:
                if term.name != "X" and term not in all_args:
                    all_args.append(term)
        else:
            for term in body.terms:
                if term.name != "X" and term not in used_args:
                    used_args.append(term)
    for arg in all_args:
        if arg not in used_args:
            unused_args.append(arg)
    return unused_args


def replace_inv_to_equiv_preds(inv_pred):
    equiv_preds = []
    for atom_list in inv_pred.body:
        equiv_preds.append(atom_list)
    return equiv_preds


def get_equivalent_clauses(c):
    equivalent_clauses = [c]
    inv_preds = []
    usual_preds = []
    for atom in c.body:
        if "inv_pred" in atom.pred.name:
            inv_preds.append(atom)
        else:
            usual_preds.append(atom)

    if len(inv_preds) == 0:
        return equivalent_clauses
    else:
        for inv_atom in inv_preds:
            inv_pred_equiv_bodies = replace_inv_to_equiv_preds(inv_atom.pred)
            for equiv_inv_body in inv_pred_equiv_bodies:
                equiv_body = sorted(list(set(equiv_inv_body + usual_preds)))
                equiv_c = logic.Clause(head=c.head, body=equiv_body)
                equivalent_clauses.append(equiv_c)

    return equivalent_clauses


def count_arity_from_clause_cluster(clause_cluster):
    arity_list = []
    for [c_i, clause, c_score] in clause_cluster:
        for b in clause.body:
            if "in" == b.pred.name:
                continue
            for t in b.terms:
                if t.name not in arity_list and "O" in t.name:
                    arity_list.append(t.name)
    arity_list.sort()
    return arity_list


def count_arity_from_clause(clause):
    arity_list = []
    for b in clause.body:
        if "in" == b.pred.name:
            continue
        for t in b.terms:
            if t.name not in arity_list and "O" in t.name:
                arity_list.append(t.name)
    arity_list.sort()
    return arity_list


