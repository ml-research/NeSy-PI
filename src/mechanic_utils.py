from aitk.utils import log_utils

import logic_utils


# def clause_extension(pi_clauses, args, max_clause, lang, mode_declarations):
#     log_utils.add_lines(f"\n=== beam search iteration {args.iteration}/{args.max_step} ===", args.log_file)
#     index_pos = config.score_example_index["pos"]
#     index_neg = config.score_example_index["neg"]
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
#         refs_extended, is_done = extend_clauses(args, lang, mode_declarations, refs, pi_clauses)
#         if is_done:
#             break
#
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
#         max_clause, found_sn = check_result(args, clause_with_scores, higher, max_clause, new_max)
#
#         if args.pi_top > 0:
#             refs, clause_with_scores, is_done = prune_clauses(clause_with_scores, args)
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


def remove_conflict_clauses(refs, pi_clauses, args):
    # remove conflict clauses
    refs_non_conflict = logic_utils.remove_conflict_clauses(refs, pi_clauses, args)
    refs_non_trivial = logic_utils.remove_trivial_clauses(refs_non_conflict, args)

    log_utils.add_lines(f"after removing conflict clauses: {len(refs_non_trivial)} clauses left", args.log_file)
    return refs_non_trivial


def check_result(args, clause_with_scores, higher, max_clause, new_max_clause):
    if higher:
        best_clause = new_max_clause
    else:
        best_clause = max_clause

    if len(clause_with_scores) == 0:
        return best_clause, False
    elif clause_with_scores[0][1][2] == 1.0:
        return best_clause, True
    elif clause_with_scores[0][1][2] > args.sn_th:
        return best_clause, True
    return best_clause, False

# def get_lang_model(args, percept_dict, obj_groups):
#     clauses = []
#     # load language module
#     lang = Language(args, [])
# update language with neural predicate: shape/color/dir/dist


# PM = get_perception_module(args)
# VM = get_valuation_module(args, lang)
# PI_VM = PIValuationModule(lang=lang, device=args.device, dataset=args.dataset, dataset_type=args.dataset_type)
# FC = FactsConverter(lang=lang, perception_module=PM, valuation_module=VM,
#                                     pi_valuation_module=PI_VM, device=args.device)
# # Neuro-Symbolic Forward Reasoner for clause generation
# NSFR_cgen = get_nsfr_model(args, lang, clauses, atoms, pi_clauses, FC)
# PI_cgen = pi_utils.get_pi_model(args, lang, clauses, atoms, pi_clauses, FC)
#
# mode_declarations = get_mode_declarations(args, lang)
# clause_generator = ClauseGenerator(args, NSFR_cgen, PI_cgen, lang, mode_declarations,
#                                    no_xil=args.no_xil)  # torch.device('cpu'))
#
# # pi_clause_generator = PIClauseGenerator(args, NSFR_cgen, PI_cgen, lang,
# #                                         no_xil=args.no_xil)  # torch.device('cpu'))


# def update_system(args, category, ):
#     # update arguments
#     clauses = []
#     p_inv_with_scores = []
#     # load language module
#     lang, vars, init_clauses, atoms = get_lang(args)
#     # update language with neural predicate: shape/color/dir/dist
#
#     if (category < len(args.neural_preds) - 1):
#         lang.preds = lang.preds[:2]
#         lang.invented_preds = []
#         lang.preds.append(args.neural_preds[category][0])
#         pi_clauses = []
#         pi_p = []
#     else:
#         print('last round')
#         lang.preds = lang.preds[:2] + args.neural_preds[-1]
#         lang.invented_preds = invented_preds
#         pi_clauses = all_pi_clauses
#         pi_p = invented_preds
#
#     atoms = logic_utils.get_atoms(lang)
#
#     args.is_done = False
#     args.iteration = 0
#     args.max_clause = [0.0, None]
#     args.no_new_preds = False
#     args.last_refs = init_clauses
#
#     PM = get_perception_module(args)
#     VM = get_valuation_module(args, lang)
#     PI_VM = PIValuationModule(lang=lang, device=args.device, dataset=args.dataset, dataset_type=args.dataset_type)
#     FC = facts_converter.FactsConverter(lang=lang, perception_module=PM, valuation_module=VM,
#                                         pi_valuation_module=PI_VM, device=args.device)
#     # Neuro-Symbolic Forward Reasoner for clause generation
#     NSFR_cgen = get_nsfr_model(args, lang, FC)
#     PI_cgen = pi_utils.get_pi_model(args, lang, clauses, atoms, pi_clauses, FC)
#
#     mode_declarations = get_mode_declarations(args, lang)
#     clause_generator = ClauseGenerator(args, NSFR_cgen, PI_cgen, lang, mode_declarations,
#                                        no_xil=args.no_xil)  # torch.device('cpu'))
#
#     # pi_clause_generator = PIClauseGenerator(args, NSFR_cgen, PI_cgen, lang,
#     #                                         no_xil=args.no_xil)  # torch.device('cpu'))
#
#     return atoms, pi_clauses, pi_p
