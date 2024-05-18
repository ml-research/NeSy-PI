# Created by J.Sha
# Create Date: 31.01.2023
from lark import Lark

from aitk.utils.fol.exp_parser import ExpTree
from aitk.utils.fol.language import DataType

import logic_utils


def generate_new_predicate(args, lang, clause_clusters, pi_type=None):
    new_predicates = []
    # cluster predicates
    for pi_index, [clause_cluster, cluster_score] in enumerate(clause_clusters):
        p_args = logic_utils.count_arity_from_clause_cluster(clause_cluster)
        dtypes = [DataType("group")] * len(p_args)
        new_predicate = lang.inv_pred(args, arity=len(p_args), pi_dtypes=dtypes, p_args=p_args, pi_type=pi_type)
        new_predicate.body = []
        for [c_i, clause, c_score] in clause_cluster:
            atoms = []
            for atom in clause.body:
                terms = logic_utils.get_terms_from_atom(atom)
                terms = sorted(terms)
                if "X" in terms:
                    terms.remove("X")
                obsolete_term = [t for t in terms if t not in p_args]
                if len(obsolete_term) == 0:
                    atoms.append(atom)
            new_predicate.body.append(atoms)
        if len(new_predicate.body) > 1:
            new_predicates.append([new_predicate, cluster_score])
        elif len(new_predicate.body) == 1:
            body = (new_predicate.body)[0]
            if len(body) > new_predicate.arity + 1:
                new_predicates.append([new_predicate, cluster_score])
    return new_predicates


def generate_new_explain_predicate__(args, lang, clause_clusters, pi_type=None):
    new_predicates = []
    # cluster predicates
    for pi_index, [clause_cluster, cluster_score] in enumerate(clause_clusters):
        p_args = logic_utils.count_arity_from_clause(clause_cluster)
        dtypes = [DataType("group")] * len(p_args)
        new_predicate = lang.inv_pred(args, arity=len(p_args), pi_dtypes=dtypes, p_args=p_args, pi_type=pi_type)
        new_predicate.body = []
        for [c_i, clause, c_score] in clause_cluster:
            atoms = []
            for atom in clause.body:
                terms = logic_utils.get_terms_from_atom(atom)
                terms = sorted(terms)
                if "X" in terms:
                    terms.remove("X")
                obsolete_term = [t for t in terms if t not in p_args]
                if len(obsolete_term) == 0:
                    atoms.append(atom)
            new_predicate.body.append(atoms)
        if len(new_predicate.body) > 1:
            new_predicates.append([new_predicate, cluster_score])
        elif len(new_predicate.body) == 1:
            body = (new_predicate.body)[0]
            if len(body) > new_predicate.arity + 1:
                new_predicates.append([new_predicate, cluster_score])
    return new_predicates


# def generate_new_explain_predicate(args, lang, head_terms, min_value_set, obj_indices):
#     # shape_counter(G, number1), shape_counter_explain_1(G, number1),
#     # shape_counter_explain_1(G, number1) :- cube_percentage(G, 1.0), sphere_percentage(G, 0.0) ||
#     # cube_percentage(G, 0.0), sphere_percentage(G, 1.0).
#     # cluster predicates
#     p_args = head_terms
#     # new predicate
#     new_predicate = lang.inv_pred(args, arity=len(p_args), pi_dtypes=None, p_args=p_args, pi_type=config.pi_type["exp"])
#     # define atoms
#     # extend the clause with new atoms
#     new_predicate.obj_indices = obj_indices
#     new_predicate.value_set = min_value_set
#     return new_predicate


def gen_clu_pi_clauses(args, lang, new_predicates, clause_str_list_with_score, kp_str_list):
    """Read lines and parse to Atom objects.
    """

    with open(args.lark_path, encoding="utf-8") as grammar:
        lp_clause = Lark(grammar.read(), start="clause")

    for n_p in new_predicates:
        lang.invented_preds.append(n_p[0])
    clause_str_list = []
    for c_str, c_score in clause_str_list_with_score:
        clause_str_list += c_str

    clauses = []
    for clause_str in clause_str_list:
        tree = lp_clause.parse(clause_str)
        clause = ExpTree(lang).transform(tree)
        clauses.append(clause)

    # for str in kp_str_list:
    #     print(str)
    kp_clause = []
    for clause_str in kp_str_list:
        tree = lp_clause.parse(clause_str)
        clause = ExpTree(lang).transform(tree)
        kp_clause.append(clause)

    return clauses, kp_clause


def gen_exp_pi_clauses(args, lang, clause_str_list_with_score):
    """ Read lines and parse to Atom objects. """

    clauses = []
    for n_p in clause_str_list_with_score:
        clauses.append(n_p[0])

    return clauses
