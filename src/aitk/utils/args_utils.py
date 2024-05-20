# Created by shaji on 26-May-2023

import argparse
import json
import os


def get_args(data_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int,
                        default=1, help="Batch size in beam search")
    parser.add_argument("--batch-size-train", type=int,
                        default=20, help="Batch size in nsfr train")
    parser.add_argument("--group_e", type=int, default=2,
                        help="The number of groups in one image")
    parser.add_argument("--group_max_e", type=int, default=5,
                        help="The maximum number of groups in one image")
    parser.add_argument("--dataset", default="red-triangle",
                        help="Use kandinsky patterns dataset")
    parser.add_argument("--is_visual", action="store_true",
                        help="Analysis visualization for grouping results.")
    parser.add_argument("--dataset-type", default="kandinsky",
                        help="kandinsky or clevr")
    parser.add_argument('--device_id', default='cpu',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument("--with_pi", action="store_true",
                        help="Generate Clause with predicate invention.")
    parser.add_argument("--with_explain", action="store_true",
                        help="Explain Clause with predicate invention.")

    parser.add_argument("--small-data", action="store_true",
                        help="Use small training data.")
    parser.add_argument("--score_unique", action="store_false",
                        help="prune same score clauses.")
    parser.add_argument("--semantic_unique", action="store_false",
                        help="prune same semantic clauses.")
    parser.add_argument("--no-xil", action="store_true",
                        help="Do not use confounding labels for clevr-hans.")
    parser.add_argument("--small_data", action="store_false",
                        help="Use small portion of valuation data.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.001, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--plot", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--t-beam", type=int, default=4,
                        help="Number of rule expantion of clause generation.")
    parser.add_argument("--min-beam", type=int, default=0,
                        help="The size of the minimum beam.")
    parser.add_argument("--n-beam", type=int, default=5,
                        help="The size of the beam.")
    parser.add_argument("--cim-step", type=int, default=5,
                        help="The steps of clause infer module.")
    parser.add_argument("--n-max", type=int, default=50,
                        help="The maximum number of clauses.")
    parser.add_argument("--m", type=int, default=1,
                        help="The size of the logic program.")
    parser.add_argument("--max_group_num", type=int, default=6,
                        help="The max number of groups to be clustered.")
    # parser.add_argument("--n-obj", type=int, default=5, help="The number of objects to be focused.")
    parser.add_argument("--epochs", type=int, default=101,
                        help="The number of epochs.")
    parser.add_argument("--pi_epochs", type=int, default=3,
                        help="The number of epochs for predicate invention.")
    parser.add_argument("--nc_max_step", type=int, default=3,
                        help="The number of max steps for nc searching.")
    parser.add_argument("--max_step", type=int, default=5,
                        help="The number of max steps for clause searching.")
    parser.add_argument("--weight_tp", type=float, default=0.95,
                        help="The weight of true positive in evaluation equation.")
    parser.add_argument("--weight_length", type=float, default=0.05,
                        help="The weight of length in evaluation equation.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="The learning rate.")
    parser.add_argument("--suff_min", type=float, default=0.1,
                        help="The minimum accept threshold for sufficient clauses.")
    parser.add_argument("--sn_th", type=float, default=0.9,
                        help="The accept threshold for sufficient and necessary clauses.")
    parser.add_argument("--nc_th", type=float, default=0.9,
                        help="The accept threshold for necessary clauses.")
    parser.add_argument("--uc_th", type=float, default=0.8,
                        help="The accept threshold for unclassified clauses.")
    parser.add_argument("--sc_th", type=float, default=0.9,
                        help="The accept threshold for sufficient clauses.")
    parser.add_argument("--sn_min_th", type=float, default=0.2,
                        help="The accept sn threshold for sufficient or necessary clauses.")
    parser.add_argument("--similar_th", type=float, default=1e-3,
                        help="The minimum different requirement between any two clauses.")
    parser.add_argument("--semantic_th", type=float, default=0.75,
                        help="The minimum semantic different requirement between any two clauses.")
    parser.add_argument("--conflict_th", type=float, default=0.9,
                        help="The accept threshold for conflict clauses.")
    parser.add_argument("--length_weight", type=float, default=0.05,
                        help="The weight of clause length for clause evaluation.")
    parser.add_argument("--c_top", type=int, default=20,
                        help="The accept number for clauses.")
    parser.add_argument("--uc_good_top", type=int, default=10,
                        help="The accept number for unclassified good clauses.")
    parser.add_argument("--sc_good_top", type=int, default=20,
                        help="The accept number for sufficient good clauses.")
    parser.add_argument("--sc_top", type=int, default=20,
                        help="The accept number for sufficient clauses.")
    parser.add_argument("--nc_top", type=int, default=10,
                        help="The accept number for necessary clauses.")
    parser.add_argument("--nc_good_top", type=int, default=30,
                        help="The accept number for necessary good clauses.")
    parser.add_argument("--pi_top", type=int, default=20,
                        help="The accept number for pi on each classes.")
    parser.add_argument("--max_cluster_size", type=int, default=5,
                        help="The max size of clause cluster.")
    parser.add_argument("--min_cluster_size", type=int, default=2,
                        help="The min size of clause cluster.")
    parser.add_argument("--n-data", type=float, default=200,
                        help="The number of data to be used.")
    parser.add_argument("--pre-searched", action="store_true",
                        help="Using pre searched clauses.")
    parser.add_argument("--top_data", type=int, default=20,
                        help="The maximum number of training data.")
    parser.add_argument("--with_bk", action="store_true",
                        help="Using background knowledge by PI.")
    parser.add_argument("--error_th", type=float, default=0.001,
                        help="The threshold for MAE of line group fitting.")
    parser.add_argument("--line_even_error", type=float, default=0.001,
                        help="The threshold for MAE of  point distribution in a line group.")
    parser.add_argument("--cir_error_th", type=float, default=0.05,
                        help="The threshold for MAE of circle group fitting.")
    parser.add_argument("--cir_even_error", type=float, default=0.001,
                        help="The threshold for MAE of point distribution in a circle group.")
    parser.add_argument("--poly_error_th", type=float, default=0.1,
                        help="The threshold for error of poly group fitting.")
    parser.add_argument("--line_group_min_sz", type=int, default=3,
                        help="The minimum objects allowed to fit a line.")
    parser.add_argument("--cir_group_min_sz", type=int, default=5,
                        help="The minimum objects allowed to fit a circle.")
    parser.add_argument("--conic_group_min_sz", type=int, default=5,
                        help="The minimum objects allowed to fit a conic section.")
    parser.add_argument("--group_conf_th", type=float, default=0.98,
                        help="The threshold of group confidence.")
    parser.add_argument("--re_eval_groups", action="store_true",
                        help="Overwrite the evaluated group detection files.")
    parser.add_argument("--maximum_obj_num", type=int, default=5,
                        help="The maximum number of objects/groups to deal with in a single image.")
    parser.add_argument("--distribute_error_th", type=float, default=0.0005,
                        help="The threshold for group points forming a shape that evenly distributed on the whole shape.")
    parser.add_argument("--show_process", action="store_false",
                        help="Print process to the logs and screen.")
    parser.add_argument("--obj_group", action="store_false",
                        help="Treat a single object as a group.")
    parser.add_argument("--line_group", action="store_false",
                        help="Treat a line of objects as a group.")
    parser.add_argument("--circle_group", action="store_false",
                        help="Treat a circle of objects as a group.")
    parser.add_argument("--conic_group", action="store_false",
                        help="Treat a conic of objects as a group.")
    parser.add_argument("--bk_pred_names", type=str,
                        help="BK predicates used in this exp.")
    parser.add_argument("--phi_num", type=int, default=20,
                        help="The number of directions for direction predicates.")
    parser.add_argument("--rho_num", type=int, default=20,
                        help="The number of distance for distance predicates.")
    parser.add_argument("--slope_num", type=int, default=10,
                        help="The number of directions for direction predicates.")
    parser.add_argument("--avg_dist_penalty", type=float, default=0.2,
                        help="The number of directions for direction predicates.")
    args = parser.parse_args()

    args_file = data_path / "lang" / "exp_args" / f"{str(args.dataset)}.json"
    load_args_from_file(str(args_file), args)

    return args


def load_args_from_file(args_file_path, given_args):
    if os.path.isfile(args_file_path):
        with open(args_file_path, 'r') as fp:
            loaded_args = json.load(fp)

        # Replace given_args with the loaded default values
        for key, value in loaded_args.items():
            # if key not in ['conflict_th', 'sc_th','nc_th']:  # Do not overwrite these keys
            setattr(given_args, key, value)

        print('\n==> Args were loaded from file "{}".'.format(args_file_path))
    else:
        print('\n==> Args file "{}" was not found!'.format(args_file_path))
    return None
