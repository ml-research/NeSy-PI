import os
import torch

import config

from aitk.utils.fol.language import DataType
from aitk.utils.fol.logic import NeuralPredicate


def load_neural_preds(neural_predicates, pi_type):
    preds = [parse_neural_pred(value, pi_type) for value in neural_predicates]
    return preds


def parse_neural_pred(line, pi_type):
    """Parse string to predicates.
    """
    line = line.replace('\n', '')
    pred, arity, dtype_names_str = line.split(':')
    dtype_names = dtype_names_str.split(',')
    dtypes = [DataType(dt) for dt in dtype_names]

    assert int(arity) == len(dtypes), 'Invalid arity and dtypes in ' + pred + '.'
    return NeuralPredicate(pred, int(arity), dtypes, pi_type)


def load_dataset(args):
    image_root = config.buffer_path / args.dataset
    image_name_dict = {}
    for data_mode in ['test', 'train', 'val']:
        tar_file = image_root / f"{args.dataset}_pm_res_{data_mode}.pth.tar"
        if not os.path.exists(tar_file):
            raise FileNotFoundError('OD result is not found.')
        tensor_dict = torch.load(tar_file)

        image_name_dict[data_mode] = {}
        image_name_dict[data_mode]["true"] = tensor_dict["pos_names"]
        image_name_dict[data_mode]["false"] = tensor_dict["neg_names"]

    args.image_name_dict = image_name_dict
