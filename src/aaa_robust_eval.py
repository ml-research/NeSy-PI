# Created by shaji on 18-Sep-23
import os
import time
import datetime
from pathlib import Path
import torch
from rtpt import RTPT

from aitk import percept
from aitk.percept_group import detect_obj_groups
from aitk.utils import log_utils, args_utils, file_utils
import semantic as se
import config

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def init(args):
    if "alphabet" in args.dataset_type:
        name = str(Path("A") / f"{args.dataset}")
    else:
        if not args.no_xil:
            name = str(Path('CH') / Path(f"/aILP_{args.dataset}"))
        else:
            name = str(Path('CH') / f"aILP-noXIL_{args.dataset}")

    log_utils.add_lines(f"============================= Group Num: {args.group_e} =======================",
                        args.log_file)
    # log_utils.add_lines(f"args: {args}", log_file)

    img_output_path = config.buffer_path / args.dataset_type / args.dataset / "image_output"
    if not os.path.exists(img_output_path):
        os.mkdir(img_output_path)
    args.image_output_path = img_output_path

    analysis_path = config.buffer_path / args.dataset_type / args.dataset / "analysis"
    if not os.path.exists(analysis_path):
        os.mkdir(analysis_path)
        os.mkdir(analysis_path / "train_pos")
        os.mkdir(analysis_path / "train_neg")
        os.mkdir(analysis_path / "test_pos")
        os.mkdir(analysis_path / "test_neg")
        os.mkdir(analysis_path / "val_pos")
        os.mkdir(analysis_path / "val_neg")

    args.analysis_path = analysis_path

    if args.no_cuda:
        args.device = torch.device('cpu')
    elif len(str(args.device_id).split(',')) > 1:
        # multi gpu
        args.device = torch.device('cuda')
    else:
        args.device = torch.device(int(args.device_id))

    log_utils.add_lines(f"- device: {args.device}", args.log_file)

    # Create RTPT object
    rtpt = RTPT(name_initials='JS', experiment_name=name, max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()
    torch.set_printoptions(precision=4)
    torch.autograd.set_detect_anomaly(True)

    file_path = config.buffer_path / args.dataset_type / f"{args.dataset}"
    pm_prediction_dict = percept.get_perception_predictions(args, file_path)
    pm_prediction_dict = percept.remove_zero_tensors(pm_prediction_dict, config.obj_tensor_index['prob'])

    args.n_obj = pm_prediction_dict["train_pos"][0].shape[0]
    args.n_obj = 100
    # grouping objects to reduce the problem complexity
    # group_val_pos, obj_avail_val_pos = detect_obj_groups(args, pm_prediction_dict["val_pos"], "val_pos")
    # group_val_neg, obj_avail_val_neg = detect_obj_groups(args, pm_prediction_dict["val_neg"], "val_neg")
    group_train_pos, obj_avail_train_pos = detect_obj_groups(args, pm_prediction_dict["train_pos"], "train_pos")
    group_train_neg, obj_avail_train_neg = detect_obj_groups(args, pm_prediction_dict["train_neg"], "train_neg")
    # group_test_pos, obj_avail_test_pos = detect_obj_groups(args, pm_prediction_dict["test_pos"], "test_pos")
    # group_test_neg, obj_avail_test_neg = detect_obj_groups(args, pm_prediction_dict["test_neg"], "test_neg")

    group_tensors = {
        # 'group_val_pos': group_val_pos, 'group_val_neg': group_val_neg,
        'group_train_pos': group_train_pos, 'group_train_neg': group_train_neg,
        # 'group_test_pos': group_test_pos, 'group_test_neg': group_test_neg,
    }

    group_tensors_indices = {
        # 'obj_avail_val_pos': obj_avail_val_pos, 'obj_avail_val_neg': obj_avail_val_neg,
        'obj_avail_train_pos': obj_avail_train_pos, 'obj_avail_train_neg': obj_avail_train_neg,
        # 'obj_avail_test_pos': obj_avail_test_pos, 'obj_avail_test_neg': obj_avail_test_neg,
    }

    # load logical representations
    args.lark_path = str(config.root / 'src' / 'lark' / 'exp.lark')
    args.lang_base_path = config.root / 'data' / 'lang'

    args.index_pos = config.score_example_index["pos"]
    args.index_neg = config.score_example_index["neg"]
    NSFR = None
    return args, rtpt, pm_prediction_dict, group_tensors, group_tensors_indices, NSFR


def load_inv_preds(file_path):
    bk_pred_files = str(file_path)
    f = open(bk_pred_files)
    clauses = f.readlines()
    arity = 1
    target_clauses = []
    inv_p_clauses = []

    for clause_str in clauses:
        # find its arity
        while ("O" + str(arity + 1)) in clause_str:
            arity += 1
        clause_str = clause_str.replace("\n", "")
        if "kp(X)" not in clause_str:
            inv_p_clauses.append(clause_str)
        else:
            target_clauses.append(clause_str)

    return arity, target_clauses, inv_p_clauses


def main():
    exp_start = time.time()
    args = args_utils.get_args(config.data_path)
    group_round_time = []
    train_round_time = []
    train_end = 0
    eval_end = 0

    # get images names
    file_utils.get_image_names(args)
    exp_output_path = config.buffer_path / args.dataset_type / args.dataset / "logs"
    if not os.path.exists(exp_output_path):
        os.mkdir(exp_output_path)
    log_file = log_utils.create_log_file(exp_output_path)

    print(f"- log_file_path:{log_file}")
    args.log_file = log_file

    if args.with_pi:
        bk_file = config.buffer_path / args.dataset_type / args.dataset / "learned_p_and_c.txt"
    else:
        bk_file = config.buffer_path / args.dataset_type / args.dataset / "learned_c.txt"

    arity, target_clauses, inv_p_clauses = load_inv_preds(bk_file)
    args.group_e = arity

    args, rtpt, percept_dict, obj_groups, obj_avail, nsfr = init(args)
    # set up the environment, load the dataset and results from perception models

    # ILP and PI system
    lang = se.init_eval_ilp(args, percept_dict, obj_groups, obj_avail, config.pi_type['bk'], "group",
                            target_clauses, inv_p_clauses)
    # evaluation
    se.ilp_robust_eval(args, lang)
    eval_end = time.time()

    # log
    log_utils.add_lines(f"=============================", args.log_file)
    log_utils.add_lines(f"+ Grouping round time: {(sum(group_round_time) / 60):.2f} minute(s)", args.log_file)
    log_utils.add_lines(f"+ Training round time: {(sum(train_round_time) / 60):.2f} minute(s)", args.log_file)
    log_utils.add_lines(f"+ Evaluation round time: {((eval_end - train_end) / 60):.2f} minute(s)", args.log_file)
    log_utils.add_lines(f"+ Running time: {((eval_end - exp_start) / 60):.2f} minute(s)", args.log_file)
    log_utils.add_lines(f"=============================", args.log_file)


if __name__ == "__main__":
    main()
