import torch


# def eval_clause_sign(p_scores):
#     p_clauses_signs = []
#
#     # p_scores axis: batch_size, pred_names, clauses, pos_neg_labels, images
#     p_scores[p_scores == 1] = 0.98
#     resolution = 2
#     ps_discrete = (p_scores * resolution).int()
#     four_zone_scores = torch.zeros((p_scores.size(0), 4))
#     img_total = p_scores.size(1)
#
#     # low pos, low neg
#     four_zone_scores[:, 0] = img_total - ps_discrete.sum(dim=2).count_nonzero(dim=1)
#
#     # high pos, low neg
#     four_zone_scores[:, 1] = img_total - (ps_discrete[:, :, 0] - ps_discrete[:, :, 1] + 1).count_nonzero(dim=1)
#
#     # low pos, high neg
#     four_zone_scores[:, 2] = img_total - (ps_discrete[:, :, 0] - ps_discrete[:, :, 1] - 1).count_nonzero(dim=1)
#
#     # high pos, high neg
#     four_zone_scores[:, 3] = img_total - (ps_discrete.sum(dim=2) - 2).count_nonzero(dim=1)
#
#     clause_score = four_zone_scores[:, 1] + four_zone_scores[:, 3]
#
#     # four_zone_scores[:, 0] = 0
#
#     # clause_sign_list = (four_zone_scores.max(dim=-1)[1] - 1) == 0
#
#     # TODO: find a better score evaluation function
#     p_clauses_signs.append([clause_score, four_zone_scores])
#
#     return p_clauses_signs


def eval_ness(positive_scores):
    positive_scores[positive_scores == 1] = 0.98
    ness_scores = positive_scores.sum(dim=2).sum(dim=1) / positive_scores.shape[1]

    return ness_scores


def eval_suff(negative_scores):
    negative_scores[negative_scores == 1] = 0.98

    # negative scores are inversely proportional to sufficiency scores
    negative_scores_inv = 1 - negative_scores
    suff_scores = negative_scores_inv.sum(dim=2).sum(dim=1) / negative_scores_inv.shape[1]
    return suff_scores


def eval_sn(positive_scores, negative_scores):
    negative_scores[negative_scores == 1] = 0.98
    positive_scores[positive_scores == 1] = 0.98

    # negative scores are inversely proportional to sufficiency scores
    negative_scores_inv = 1 - negative_scores
    ness_scores = positive_scores.sum(dim=2).sum(dim=1) / positive_scores.shape[1]
    suff_scores = negative_scores_inv.sum(dim=2).sum(dim=1) * positive_scores.sum(dim=2).sum(dim=1) / \
                  negative_scores_inv.shape[1]
    return suff_scores


def eval_score_similarity(score, appeared_scores, threshold):
    is_repeat = False
    for appeared_score in appeared_scores:
        if score > 0.9:
            is_repeat = False
        elif torch.abs(score - appeared_score) / appeared_score < threshold:
            is_repeat = True

    return is_repeat


def eval_semantic_similarity(semantic, appeared_semantics, args):
    is_repeat = False
    for appeared_semantic in appeared_semantics:
        similar_counter = 0
        for p_i in range(len(appeared_semantic)):
            if p_i < len(semantic):
                for a_i in range(len(appeared_semantic[p_i])):
                    if a_i < len(semantic[p_i]):
                        if semantic[p_i][a_i] == appeared_semantic[p_i][a_i]:
                            similar_counter += 1
                        elif isinstance(semantic[p_i][a_i], list):
                            if semantic[p_i][a_i][-1] == appeared_semantic[p_i][a_i][-1]:
                                similar_counter += 1
        similarity = similar_counter / (len(semantic) * 2)
        if similarity > args.semantic_th:
            is_repeat = True
    return is_repeat
