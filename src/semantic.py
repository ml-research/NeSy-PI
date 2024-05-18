# Created by jing at 30.05.23

"""
semantic implementation
"""
from aitk.utils import logic_utils
from aitk.utils.fol.language import Language

import ilp


def init_ilp(args, percept_dict, obj_groups, obj_avail, pi_type, level):
    logic_utils.update_args(args, percept_dict, obj_groups, obj_avail)
    lang = Language(args, [], pi_type, level)
    return lang

def init_eval_ilp(args, percept_dict, obj_groups, obj_avail, pi_type, level,  target_clauses, inv_p_clauses):
    logic_utils.update_eval_args(args, percept_dict, obj_groups, obj_avail)
    lang = Language(args, [], pi_type, level)

    reset_lang(lang, args, level, args.neural_preds, full_bk=True)
    lang.load_invented_preds(inv_p_clauses, target_clauses)
    reset_args(args)
    import aitk.utils.lang_utils as lang_utils
    lang.mode_declarations = lang_utils.get_mode_declarations(args.group_e, lang)

    return lang
# def run_ilp(args, lang, level):
#     success = ilp.ilp_main(args, lang, level)
#     return success

def clause2scores():
    pass


def scene2clauses():
    pass


# to be removed
def run_ilp_predict(args, NSFR, th, split):
    acc_val, rec_val, th_val = ilp.ilp_predict(NSFR, args, th=th, split=split)
    return acc_val, rec_val, th_val


# ---------------------------- ilp api --------------------------------------
def extend_clause():
    pass


def reset_args(args):
    ilp.reset_args(args)


def reset_lang(lang, args, level, neural_pred, full_bk):
    init_clause, e = ilp.reset_lang(lang, args, level, neural_pred, full_bk)
    return init_clause, e


def search_clauses(args, lang, init_clauses, FC, level):
    clauses = ilp.ilp_search(args, lang, init_clauses, FC, level)
    return clauses


def explain_clauses(args, lang, clauses):
    ilp.explain_scenes(args, lang, clauses)


def run_ilp(args, lang, level):
    # print all the invented predicates
    return success, clauses


def predicate_invention(args, lang, clauses, e):
    ilp.ilp_pi(args, lang, clauses, e)


def keep_best_preds(args, lang):
    ilp.keep_best_preds(args, lang)


def run_ilp_train(args, lang, level):
    ilp.ilp_train(args, lang, level)
    success, clauses = ilp.ilp_test(args, lang, level)
    return success, clauses


def run_ilp_train_explain(args, lang, level):
    ilp.ilp_train_explain(args, lang, level)


def ilp_eval(success, args, lang, clauses, g_data):
    scores = ilp.ilp_eval(success, args, lang, clauses, g_data)
    return scores

def ilp_robust_eval(args, lang):
    scores = ilp.ilp_robust_eval(args, lang)
    return scores

def train_nsfr(args, rtpt, lang, clauses):
    ilp.train_nsfr(args, rtpt, lang, clauses)
