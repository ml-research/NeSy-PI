# Created by jing at 30.05.23

"""
Root utils file, only import modules that don't belong to this project.
"""
import itertools
import os
import torch

import config
from aitk.utils import log_utils, file_utils
from aitk.utils.fol import bk


def get_index_by_predname(pred_str, atoms):
    indices = []
    for p_i, p_str in enumerate(pred_str):
        p_indices = []
        for i, atom in enumerate(atoms):
            if atom.pred.name == p_str:
                p_indices.append(i)
        indices.append(p_indices)
    return indices


def data_ordering(data):
    data_ordered = torch.zeros(data.shape)
    delta = data[:, :, :3].max(dim=1, keepdims=True)[0] - data[:, :, :3].min(dim=1, keepdims=True)[0]
    order_axis = torch.argmax(delta, dim=2)
    for data_i in range(len(data)):
        data_order_i = data[data_i, :, order_axis[data_i]].sort(dim=0)[1].squeeze(1)
        data_ordered[data_i] = data[data_i, data_order_i, :]

    return data_ordered


def convert_data_to_tensor(args, od_res):
    if os.path.exists(od_res):
        pm_res = torch.load(od_res)
        pos_pred = pm_res['pos_res']
        neg_pred = pm_res['neg_res']
    else:
        raise ValueError
    # data_files = glob.glob(str(pos_dataset_folder / '*.json'))
    # data_tensors = torch.zeros((len(data_files), args.e, 9))
    # for d_i, data_file in enumerate(data_files):
    #     with open(data_file) as f:
    #         data = json.load(f)
    #     data_tensor = torch.zeros(1, args.e, 9)
    #     for o_i, obj in enumerate(data["objects"]):
    #
    #         data_tensor[0, o_i, 0:3] = torch.tensor(obj["position"])
    #         if "blue" in obj["material"]:
    #             data_tensor[0, o_i, 3:6] = torch.tensor([0, 0, 1])
    #         elif "green" in obj["material"]:
    #             data_tensor[0, o_i, 3:6] = torch.tensor([0, 1, 0])
    #         else:
    #             data_tensor[0, o_i, 3:6] = torch.tensor([1, 0, 0])
    #         if "sphere" in obj["material"]:
    #             data_tensor[0, o_i, 6] = 0.99
    #         if "cube" in obj["material"]:
    #             data_tensor[0, o_i, 7] = 0.99
    #         data_tensor[0, o_i, 8] = 0.99
    #     data_tensors[d_i] = data_tensor[0]

    return pos_pred, neg_pred


def vertex_normalization(data):
    return data

    if len(data.shape) != 3:
        raise ValueError

    ax = 0
    min_value = data[:, :, ax:ax + 1].min(axis=1, keepdims=True)[0].repeat(1, data.shape[1], 3)
    max_value = data[:, :, ax:ax + 1].max(axis=1, keepdims=True)[0].repeat(1, data.shape[1], 3)
    data[:, :, :3] = (data[:, :, :3] - min_value) / (max_value - min_value + 1e-10)

    ax = 2
    data[:, :, ax] = data[:, :, ax] - data[:, :, ax].min(axis=1, keepdims=True)[0]
    # for i in range(len(data)):
    #     data_plot = np.zeros(shape=(5, 2))
    #     data_plot[:, 0] = data[i, :5, 0]
    #     data_plot[:, 1] = data[i, :5, 2]
    #     chart_utils.plot_scatter_chart(data_plot, config.buffer_path / "hide", show=True, title=f"{i}")
    return data


def sorted_clauses(clause_with_scores, args):
    if len(clause_with_scores) > 0:
        c_sorted = sorted(clause_with_scores, key=lambda x: x[1][2], reverse=True)
        # for c in c_sorted:
        #     log_utils.add_lines(f"clause: {c[0]} {c[1]}", args.log_file)
        return c_sorted
    else:
        return []


def extract_clauses_from_max_clause(bs_clauses, args):
    clauses = []
    if len(bs_clauses) == 0:
        return clauses

    for bs_clause in bs_clauses:
        clauses.append(bs_clause[0])
        log_utils.add_lines(f"add max clause: {bs_clause[0]}", args.log_file)
    return clauses


def top_select(bs_clauses, args):
    # all_c = bs_clauses['sn'] + bs_clauses['nc'] + bs_clauses['sc'] + bs_clauses['nc_good'] + bs_clauses['sc_good'] + \
    #         bs_clauses['uc'] + bs_clauses['uc_good']

    top_clauses = sorted_clauses(bs_clauses, args)
    top_clauses = top_clauses[:args.c_top]
    top_clauses = extract_clauses_from_max_clause(top_clauses, args)
    return top_clauses


def extract_clauses_from_bs_clauses(bs_clauses, c_type, args):
    clauses = []
    if len(bs_clauses) == 0:
        return clauses

    for bs_clause in bs_clauses:
        clauses.append(bs_clause[0])
        if args.show_process:
            if len(bs_clause) == 3:
                positive_score = bs_clause[2][:, config.score_example_index["pos"]]
                failed_img_index = ((positive_score < 0.9).nonzero(as_tuple=True)[0]).tolist()
                log_utils.add_lines(
                    f"({c_type}): {bs_clause[0]} {bs_clause[1].reshape(-1)} "
                    f"Failed Positive Image: ({len(failed_img_index)}/{bs_clause[2].shape[0]}) {failed_img_index} ",
                    args.log_file)
    return clauses


def get_pred_names_from_clauses(clause, exclude_objects=False):
    preds = []
    for atom in clause.body:
        pred = atom.pred.name
        if "in" == pred:
            continue
        if exclude_objects:
            terms = [t.name for t in atom.terms if "O" not in t.name]
        else:
            terms = [t.name for t in atom.terms]
        if pred not in preds:
            preds.append([pred, terms])
    return preds


def get_semantic_from_c(clause):
    semantic = []
    semantic += get_pred_names_from_clauses(clause)
    return semantic


def get_independent_clusters(args, lang, clauses):
    clause_with_scores = lang.clause_with_scores

    if args.show_process:
        print(f"- searching for independent clauses from {len(clause_with_scores)} clauses...")

    clauses_with_score = []
    for clause_i, [clause, four_scores, c_scores] in enumerate(clause_with_scores):
        clauses_with_score.append([clause_i, clause, c_scores])

    clause_clusters = sub_lists(clauses_with_score, min_len=args.min_cluster_size, max_len=args.max_cluster_size)

    return clause_clusters


def check_trivial_clusters(clause_clusters):
    clause_clusters_untrivial = []
    for c_clu in clause_clusters:
        is_trivial = False
        if len(c_clu) > 1:
            for c_i, c in enumerate(c_clu):
                clause = c[1]
                clause.body = sorted(clause.body)
                if c_i > 0:
                    if has_same_preds_and_atoms(clause, c_clu[0][1]):
                        is_trivial = True
                        break
                    if not has_same_preds(clause, c_clu[0][1]):
                        is_trivial = True
                        break
        if not is_trivial:
            clause_clusters_untrivial.append(c_clu)
    return clause_clusters_untrivial


def has_same_preds_and_atoms(c1, c2):
    if len(c1.body) != len(c2.body):
        return False
    same_preds = True
    for i in range(len(c1.body)):
        if not same_preds:
            break
        if not c1.body[i].pred.name == c2.body[i].pred.name:
            same_preds = False
        else:
            for j, term in enumerate(c1.body[i].terms):
                if "O" not in term.name:
                    if not term.name == c2.body[i].terms[j].name:
                        same_preds = False
    if same_preds:
        return True
    else:
        return False


def has_same_preds(c1, c2):
    if len(c1.body) != len(c2.body):
        return False
    same_preds = True
    for i in range(len(c1.body)):
        if not c1.body[i].pred.name == c2.body[i].pred.name:
            same_preds = False
    if same_preds:
        return True
    else:
        return False


def sub_lists(l, min_len=0, max_len=None):
    # initializing empty list
    comb = []

    # Iterating till length of list
    if max_len is None:
        max_len = len(l) + 1
    for i in range(min_len, max_len):
        # Generating sub list
        comb += [list(j) for j in itertools.combinations(l, i)]
    # Returning list
    return comb


def update_args(args, pm_prediction_dict, obj_groups, obj_avail):
    args.val_pos = pm_prediction_dict["val_pos"]#.to(args.device)
    args.val_neg = pm_prediction_dict["val_neg"]#.to(args.device)
    args.test_pos = pm_prediction_dict["test_pos"]#.to(args.device)
    args.test_neg = pm_prediction_dict["test_neg"]#.to(args.device)
    args.train_pos = pm_prediction_dict["train_pos"]#.to(args.device)
    args.train_neg = pm_prediction_dict["train_neg"]#.to(args.device)


    args.val_group_pos = obj_groups['group_val_pos']
    args.val_group_neg = obj_groups['group_val_neg']
    args.train_group_pos = obj_groups['group_train_pos']
    args.train_group_neg = obj_groups['group_train_neg']
    args.test_group_pos = obj_groups['group_test_pos']
    args.test_group_neg = obj_groups['group_test_neg']

    args.obj_avail_val_pos = obj_avail['obj_avail_val_pos']
    args.obj_avail_val_neg = obj_avail['obj_avail_val_neg']
    args.obj_avail_train_pos = obj_avail['obj_avail_train_pos']
    args.obj_avail_train_neg = obj_avail['obj_avail_train_neg']
    args.obj_avail_test_pos = obj_avail['obj_avail_test_pos']
    args.obj_avail_test_neg = obj_avail['obj_avail_test_neg']

    args.data_size = len(args.val_pos)
    args.invented_pred_num = 0
    args.last_refs = []
    args.found_ns = False
    args.d = len(config.group_tensor_index)

    # clause generation and predicate invention
    lang_data_path = args.lang_base_path / args.dataset_type / args.dataset

    pi_type = config.pi_type['bk']
    bk_preds = [bk.neural_predicate_2[bk_pred_name] for bk_pred_name in args.bk_pred_names.split(",")]
    neural_preds = file_utils.load_neural_preds(bk_preds, pi_type)

    args.neural_preds = [neural_pred for neural_pred in neural_preds]
    args.p_inv_counter = 0



def update_eval_args(args, pm_prediction_dict, obj_groups, obj_avail):
    args.train_pos = pm_prediction_dict["train_pos"]#.to(args.device)
    args.train_neg = pm_prediction_dict["train_neg"]#.to(args.device)
    args.train_group_pos = obj_groups['group_train_pos']
    args.train_group_neg = obj_groups['group_train_neg']
    args.obj_avail_train_pos = obj_avail['obj_avail_train_pos']
    args.obj_avail_train_neg = obj_avail['obj_avail_train_neg']


    args.data_size = len(args.train_pos)
    args.invented_pred_num = 0
    args.last_refs = []
    args.found_ns = False
    args.d = len(config.group_tensor_index)

    # clause generation and predicate invention
    lang_data_path = args.lang_base_path / args.dataset_type / args.dataset

    pi_type = config.pi_type['bk']
    bk_preds = [bk.neural_predicate_2[bk_pred_name] for bk_pred_name in args.bk_pred_names.split(",")]
    neural_preds = file_utils.load_neural_preds(bk_preds, pi_type)

    args.neural_preds = [neural_pred for neural_pred in neural_preds]
    args.p_inv_counter = 0
