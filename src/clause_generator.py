import torch

import aitk.utils.logic_utils
import ilp
import nsfr_utils
from aitk.utils.fol.refinement import RefinementGenerator
import logic_utils
from aitk.utils.fol import DataType
from aitk.utils import log_utils
from aitk.utils.fol import DataUtils
from src.logic_utils import count_arity_from_clause_cluster


class ClauseGenerator(object):
    """
    clause generator by refinement and beam search
    Parameters
    ----------
    ilp_problem : .ilp_problem.ILPProblem
    infer_step : int
        number of steps in forward inference
    max_depth : int
        max depth of nests of function symbols
    max_body_len : int
        max number of atoms in body of clauses
    """

    def __init__(self, args, NSFR, PI, lang, mode_declarations, no_xil=False):
        self.args = args
        self.NSFR = NSFR
        self.PI = PI
        self.lang = lang
        self.mode_declarations = mode_declarations
        self.bk_clauses = None
        self.device = args.device
        self.no_xil = no_xil
        self.rgen = RefinementGenerator(lang=lang, mode_declarations=mode_declarations)
        self.bce_loss = torch.nn.BCELoss()

    # def extend_clauses(self, clauses, args, pi_clauses):
    #     refs = []
    #     B_ = []
    #     is_done = False
    #     for c in clauses:
    #         refs_i = self.rgen.refinement_clause(c)
    #         unused_args, used_args = log_utils.get_unused_args(c)
    #         refs_i_removed = logic_utils.remove_duplicate_clauses(refs_i, unused_args, used_args, args)
    #         # remove invalid clauses
    #         ###refs_i = [x for x in refs_i if self._is_valid(x)]
    #         # remove already appeared refs
    #         refs_i_removed = list(set(refs_i_removed).difference(set(B_)))
    #         B_.extend(refs_i_removed)
    #         refs.extend(refs_i_removed)
    #
    #     # remove semantic conflict clauses
    #     refs_no_conflict = self.remove_conflict_clauses(refs, pi_clauses, args)
    #     if len(refs_no_conflict) == 0:
    #         is_done = True
    #     return refs_no_conflict, is_done

    # def clause_extension(self, init_clauses, pi_clauses, args, max_clause):
    #     index_pos = config.score_example_index["pos"]
    #     index_neg = config.score_example_index["neg"]
    #
    #     eval_pred = ['kp']
    #     clause_with_scores = []
    #     # extend clauses
    #     is_done = False
    #     # if args.no_new_preds:
    #     step = args.iteration
    #     refs = args.last_refs
    #     if args.pi_top == 0:
    #         step = args.iteration
    #         if len(args.last_refs) > 0:
    #             refs = args.last_refs
    #     while step <= args.iteration:
    #         # log
    #         log_utils.print_time(args, args.iteration, step, args.iteration)
    #         # clause extension
    #         refs_extended, is_done = self.extend_clauses(refs, args, pi_clauses)
    #         if is_done:
    #             break
    #         # self.lang.preds += self.lang.invented_preds
    #         # update NSFR
    #         self.NSFR = get_nsfr_model(args, self.lang, refs_extended, self.NSFR.atoms, pi_clauses, self.NSFR.fc)
    #         # evaluate new clauses
    #         score_all = eval_clause_infer.eval_clause_on_scenes(self.NSFR, args, eval_pred)
    #         scores = eval_clause_infer.eval_clauses(score_all[:, :, index_pos], score_all[:, :, index_neg], args, step)
    #         # classify clauses
    #         clause_with_scores = eval_clause_infer.prune_low_score_clauses(refs_extended, score_all, scores, args)
    #         # print best clauses that have been found...
    #         clause_with_scores = logic_utils.sorted_clauses(clause_with_scores, args)
    #
    #         new_max, higher = logic_utils.get_best_clauses(refs_extended, scores, step, args, max_clause)
    #         max_clause, found_sn = self.check_result(clause_with_scores, higher, max_clause, new_max)
    #
    #         if args.pi_top > 0:
    #             refs, clause_with_scores, is_done = self.prune_clauses(clause_with_scores, args)
    #         else:
    #             refs = logic_utils.top_select(clause_with_scores, args)
    #         step += 1
    #
    #         if found_sn or len(refs) == 0:
    #             is_done = True
    #             break
    #
    #     args.is_done = is_done
    #     args.last_refs = refs
    #     return clause_with_scores, max_clause, step, args

    # def remove_conflict_clauses(self, refs, pi_clauses, args):
    #     # remove conflict clauses
    #     refs_non_conflict = logic_utils.remove_conflict_clauses(refs, pi_clauses, args)
    #     refs_non_trivial = logic_utils.remove_trivial_clauses(refs_non_conflict, args)
    #
    #     log_utils.add_lines(f"after removing conflict clauses: {len(refs_non_trivial)} clauses left", args.log_file)
    #     return refs_non_trivial

    # def print_clauses(self, clause_dict, args):
    #     log_utils.add_lines('\n======= BEAM SEARCHED CLAUSES ======', args.log_file)
    #
    #     if len(clause_dict["sn"]) > 0:
    #         for c in clause_dict["sn"]:
    #             log_utils.add_lines(f"sufficient and necessary clause: {c[0]}", args.log_file)
    #     if len(clause_dict["sn_good"]) > 0:
    #         for c in clause_dict["sn_good"]:
    #             score = logic_utils.get_four_scores(c[1].unsqueeze(0))
    #             log_utils.add_lines(
    #                 f"sufficient and necessary clause with {args.sn_th * 100}% accuracy: {c[0]}, {score}",
    #                 args.log_file)
    #     if len(clause_dict["sc"]) > 0:
    #         for c in clause_dict["sc"]:
    #             score = logic_utils.get_four_scores(c[1].unsqueeze(0))
    #             log_utils.add_lines(f"sufficient clause: {c[0]}, {score}", args.log_file)
    #     if len(clause_dict["sc_good"]) > 0:
    #         for c in clause_dict["sc_good"]:
    #             score = logic_utils.get_four_scores(c[1].unsqueeze(0))
    #             log_utils.add_lines(f"sufficient clause with {args.sc_th * 100}%: {c[0]}, {score}", args.log_file)
    #     if len(clause_dict["nc"]) > 0:
    #         for c in clause_dict["nc"]:
    #             score = logic_utils.get_four_scores(c[1].unsqueeze(0))
    #             log_utils.add_lines(f"necessary clause: {c[0]}, {score}", args.log_file)
    #     if len(clause_dict["nc_good"]) > 0:
    #         for c in clause_dict["nc_good"]:
    #             score = logic_utils.get_four_scores(c[1].unsqueeze(0))
    #             log_utils.add_lines(f"necessary clause with {args.nc_th * 100}%: {c[0]}, {score}", args.log_file)
    #     log_utils.add_lines('============= Beam search End ===================\n', args.log_file)

    # def update_refs(self, clause_with_scores, args):
    #     refs = []
    #     nc_clauses = logic_utils.extract_clauses_from_bs_clauses(clause_with_scores, "clause", args)
    #     refs += nc_clauses
    #
    #     return refs

    # def prune_clauses(self, clause_with_scores, args):
    #     refs = []
    #
    #     # prune score similar clauses
    #     log_utils.add_lines(f"=============== score pruning ==========", args.log_file)
    #     # for c in clause_with_scores:
    #     #     log_utils.add_lines(f"(clause before pruning) {c[0]} {c[1].reshape(3)}", args.log_file)
    #     if args.score_unique:
    #         score_unique_c = []
    #         score_repeat_c = []
    #         appeared_scores = []
    #         for c in clause_with_scores:
    #             if not eval_clause_infer.eval_score_similarity(c[1][2], appeared_scores, args.similar_th):
    #                 score_unique_c.append(c)
    #                 appeared_scores.append(c[1][2])
    #             else:
    #                 score_repeat_c.append(c)
    #         c_score_pruned = score_unique_c
    #     else:
    #         c_score_pruned = clause_with_scores
    #
    #     # prune predicate similar clauses
    #     log_utils.add_lines(f"=============== semantic pruning ==========", args.log_file)
    #     if args.semantic_unique:
    #         semantic_unique_c = []
    #         semantic_repeat_c = []
    #         appeared_semantics = []
    #         for c in c_score_pruned:
    #             c_semantic = logic_utils.get_semantic_from_c(c[0])
    #             if not eval_clause_infer.eval_semantic_similarity(c_semantic, appeared_semantics, args):
    #                 semantic_unique_c.append(c)
    #                 appeared_semantics.append(c_semantic)
    #             else:
    #                 semantic_repeat_c.append(c)
    #         c_semantic_pruned = semantic_unique_c
    #     else:
    #         c_semantic_pruned = c_score_pruned
    #
    #     c_score_pruned = c_semantic_pruned
    #     # select top N clauses
    #     if args.c_top is not None and len(c_score_pruned) > args.c_top:
    #         c_score_pruned = c_score_pruned[:args.c_top]
    #     log_utils.add_lines(f"after top select: {len(c_score_pruned)}", args.log_file)
    #
    #     refs += self.update_refs(c_score_pruned, args)
    #
    #     return refs, c_score_pruned, False

    # def check_result(self, clause_with_scores, higher, max_clause, new_max_clause):
    #
    #     if higher:
    #         best_clause = new_max_clause
    #     else:
    #         best_clause = max_clause
    #
    #     if len(clause_with_scores) == 0:
    #         return best_clause, False
    #     elif clause_with_scores[0][1][2] == 1.0:
    #         return best_clause, True
    #     elif clause_with_scores[0][1][2] > self.args.sn_th:
    #         return best_clause, True
    #     return best_clause, False


class PIClauseGenerator(object):
    """
    clause generator by refinement and beam search
    Parameters
    ----------
    ilp_problem : .ilp_problem.ILPProblem
    infer_step : int
        number of steps in forward inference
    max_depth : int
        max depth of nests of function symbols
    max_body_len : int
        max number of atoms in body of clauses
    """

    def __init__(self, args, NSFR, PI, lang, no_xil=False):
        self.args = args
        self.lang = lang
        self.NSFR = NSFR
        self.PI = PI
        self.bk_clauses = None
        self.device = args.device

    def invent_predicate(self, lang, pi_clauses, args, neural_pred, invented_p, e):

        new_predicates, is_done = self.cluster_invention(args, lang, e)
        log_utils.add_lines(f"new PI: {len(new_predicates)}", args.log_file)
        for new_c, new_c_score in new_predicates:
            log_utils.add_lines(f"{new_c} {new_c_score.reshape(3)}", args.log_file)

        # convert to strings
        new_clauses_str_list, kp_str_list = self.generate_new_clauses_str_list(new_predicates)
        # convert clauses from strings to objects
        # pi_languages = logic_utils.get_pi_clauses_objs(self.args, self.lang, new_clauses_str_list, new_predicates)
        du = DataUtils(lark_path=args.lark_path, lang_base_path=args.lang_base_path, dataset_type=args.dataset_type,
                       dataset=args.dataset)

        lang, vars, init_clauses, atoms = nsfr_utils.get_lang(args)
        if neural_pred is not None:
            lang.preds += neural_pred
        lang.invented_preds = invented_p
        all_pi_clauses, all_pi_kp_clauses = du.gen_pi_clauses(lang, new_predicates, new_clauses_str_list, kp_str_list)

        all_pi_clauses = self.extract_pi(lang, all_pi_clauses, args) + pi_clauses
        all_pi_kp_clauses = self.extract_kp_pi(lang, all_pi_kp_clauses, args)

        new_p = self.lang.invented_preds
        new_c = all_pi_clauses

        log_utils.add_lines(f"======  Total PI Number: {len(new_p)}  ======", args.log_file)
        for p in new_p:
            log_utils.add_lines(f"{p}", args.log_file)
        log_utils.add_lines(f"========== Total {len(new_c)} PI Clauses ============= ", args.log_file)
        for c in new_c:
            log_utils.add_lines(f"{c}", args.log_file)

        return new_c, new_p, new_predicates, is_done

    def generate_new_predicate(self, args, clause_clusters, clause_type=None):
        new_predicate = None
        # positive_clauses_exchange = [(c[1], c[0]) for c in positive_clauses]
        # no_hn_ = [(c[0], c[1]) for c in positive_clauses_exchange if c[0][2] == 0 and c[0][3] == 0]
        # no_hnlp = [(c[0], c[1]) for c in positive_clauses_exchange if c[0][2] == 0]
        # score clauses properly

        new_predicates = []
        # cluster predicates
        for pi_index, [clause_cluster, cluster_score] in enumerate(clause_clusters):
            p_args = count_arity_from_clause_cluster(clause_cluster)
            dtypes = [DataType("object")] * len(p_args)
            new_predicate = self.lang.inv_pred(args, arity=len(p_args), pi_dtypes=dtypes,
                                               p_args=p_args, pi_type=clause_type)
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

    def generate_new_clauses_str_list(self, new_predicates):
        pi_str_lists = []
        kp_str_lists = []
        for [new_predicate, p_score] in new_predicates:
            single_pi_str_list = []
            # head_args = "(O1,O2)" if new_predicate.arity == 2 else "(X)"
            kp_clause = "kp(X):-"
            head_args = "("

            for arg in new_predicate.args:
                head_args += arg + ","
                kp_clause += f"in({arg},X),"
            head_args = head_args[:-1]
            head_args += ")"
            kp_clause += f"{new_predicate.name}{head_args}."
            kp_str_lists.append(kp_clause)

            head = new_predicate.name + head_args + ":-"
            for body in new_predicate.body:
                body_str = ""
                for atom_index in range(len(body)):
                    atom_str = str(body[atom_index])
                    end_str = "." if atom_index == len(body) - 1 else ","
                    body_str += atom_str + end_str
                new_clause = head + body_str
                single_pi_str_list.append(new_clause)
            pi_str_lists.append([single_pi_str_list, p_score])

        return pi_str_lists, kp_str_lists

    def extract_pi(self, new_lang, all_pi_clauses, args):
        for index, new_p in enumerate(new_lang.invented_preds):
            if new_p in self.lang.invented_preds:
                continue
            is_duplicate = False
            for self_p in self.lang.invented_preds:
                if new_p.body == self_p.body:
                    is_duplicate = True
                    log_utils.add_lines(f"duplicate pi body {new_p.name} {new_p.body}", args.log_file)
                    break
            if not is_duplicate:
                print(f"add new predicate: {new_p.name}")
                self.lang.invented_preds.append(new_p)
            else:
                log_utils.add_lines(f"duplicate pi: {new_p}", args.log_file)

        new_p_names = [self_p.name for self_p in self.lang.invented_preds]
        new_all_pi_clausese = []
        for pi_c in all_pi_clauses:
            pi_c_head_name = pi_c.head.pred.name
            if pi_c_head_name in new_p_names:
                new_all_pi_clausese.append(pi_c)
        return new_all_pi_clausese

    def extract_kp_pi(self, new_lang, all_pi_clauses, args):
        new_all_pi_clausese = []
        for pi_c in all_pi_clauses:
            pi_c_head_name = pi_c.head.pred.name
            new_all_pi_clausese.append(pi_c)
        return new_all_pi_clausese

    def cluster_invention(self, args, lang, e):
        found_ns = False

        clu_lists = ilp.search_independent_clauses_parallel(args, lang, e)
        new_predicates = self.generate_new_predicate(args, clu_lists)
        new_predicates = new_predicates[:args.pi_top]

        return new_predicates, found_ns
