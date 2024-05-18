import argparse

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from nsfr_utils import denormalize_clevr
from aitk.utils.nsfr_utils import get_prob
from nsfr_utils import save_images_with_captions, to_plot_images_clevr, generate_captions


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to infer with")
    parser.add_argument("--e", type=int, default=10,
                        help="The maximum number of objects in one image")
    parser.add_argument(
        "--dataset", choices=["clevr-hans3", "clevr-hans7"], help="Use clevr-hans dataset.")
    parser.add_argument("--dataset-type", default="clevr",
                        help="kandinsky or clevr")
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
    parser.add_argument("--plot-cam", action="store_true",
                        help="Plot images cam.")
    args = parser.parse_args()
    return args


def predict(NSFR, loader, args, device, writer, split='train'):
    predicted_list = []
    target_list = []
    count = 0
    for i, sample in tqdm(enumerate(loader, start=0)):
        # to cuda
        imgs, target_set = map(lambda x: x.to(device), sample)

        # infer and predict the target probability
        V_T = NSFR(imgs)
        #print(valuations_to_string(V_T, NSFR.atoms, NSFR.pm.e))
        predicted = get_prob(V_T, NSFR, args)

        predicted_list.extend(
            list(np.argmax(predicted.detach().cpu().numpy(), axis=1)))
        target_list.extend(
            list(np.argmax(target_set.detach().cpu().numpy(), axis=1)))

        if i < 1:
            if args.dataset_type == 'clevr':
                writer.add_images(
                    'images', denormalize_clevr(imgs).detach().cpu(), 0)
            else:
                writer.add_images(
                    'images', imgs.detach().cpu(), 0)
            writer.add_text('V_T', NSFR.get_valuation_text(V_T), 0)
        if args.plot:
            imgs = to_plot_images_clevr(imgs)
            captions = generate_captions(
                V_T, NSFR.atoms, args.e, th=0.33)
            save_images_with_captions(
                imgs, captions, folder='result/clevr/' + args.dataset + '/' + split + '/', img_id_start=count, dataset=args.dataset)
        count += V_T.size(0)  # batch size
    predicted = predicted_list
    target = target_list
    return accuracy_score(target, predicted), confusion_matrix(target, predicted)

