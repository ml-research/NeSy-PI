import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to infer with")
    parser.add_argument("--e", type=int, default=4,
                        help="The maximum number of objects in one image")
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--dataset", choices=["twopairs", "threepairs", "red-triangle", "closeby",
                                              "online", "online-pair", "nine-circles"],
                        help="Use kandinsky patterns dataset")
    parser.add_argument("--dataset-type", default="kandinsky",
                        help="kandinsky or clevr")
    parser.add_argument("--small-data", action="store_true", help="Use small training data.")
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--plot", action="store_true",
                        help="Plot images with captions.")
    args = parser.parse_args()
    return args

# def predict(NSFR, loader, args, device, writer, th=None, split='train'):
#     predicted_list = []
#     target_list = []
#     count = 0
#     for i, sample in tqdm(enumerate(loader, start=0)):
#         if i > 100:
#             break
#         # to cuda
#         imgs, target_set = map(lambda x: x.to(device), sample)
#
#         # infer and predict the target probability
#         V_T = NSFR(imgs)
#         predicted = get_prob(V_T, NSFR, args)
#         predicted_list.append(predicted)
#         target_list.append(target_set)
#         if args.plot:
#             imgs = to_plot_images_kandinsky(imgs)
#             captions = generate_captions(
#                 V_T, NSFR.atoms, args.e, th=0.3)
#             save_images_with_captions(
#                 imgs, captions, folder='result/kandinsky/' + args.dataset + '/' + split + '/', img_id_start=count, dataset=args.dataset)
#         count += V_T.size(0)  # batch size
#
#     predicted = torch.cat(predicted_list, dim=0).detach().cpu().numpy()
#     target_set = torch.cat(target_list, dim=0).to(
#         torch.int64).detach().cpu().numpy()
#
#     if th == None:
#         fpr, tpr, thresholds = roc_curve(target_set, predicted, pos_label=1)
#         accuracy_scores = []
#         print('ths', thresholds)
#         for thresh in thresholds:
#             accuracy_scores.append(accuracy_score(
#                 target_set, [m > thresh for m in predicted]))
#
#         accuracies = np.array(accuracy_scores)
#         max_accuracy = accuracies.max()
#         max_accuracy_threshold = thresholds[accuracies.argmax()]
#         rec_score = recall_score(
#             target_set,  [m > thresh for m in predicted], average=None)
#
#         print('target_set: ', target_set, target_set.shape)
#         print('predicted: ', predicted, predicted.shape)
#         print('accuracy: ', max_accuracy)
#         print('threshold: ', max_accuracy_threshold)
#         print('recall: ', rec_score)
#
#         return max_accuracy, rec_score, max_accuracy_threshold
#     else:
#         accuracy = accuracy_score(target_set, [m > th for m in predicted])
#         rec_score = recall_score(
#             target_set,  [m > th for m in predicted], average=None)
#         return accuracy, rec_score, th
#
