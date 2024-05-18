# Created by jing at 30.05.23
import torch
from torch import nn as nn

from aitk.utils import log_utils
from aitk.utils import logic_utils

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression


def get_perception_module(args):
    PM = FCNNPerceptionModule(e=args.e, d=13, device=args.device)
    return PM


class FCNNPerceptionModule(nn.Module):
    """A perception module using YOLO.

    Attrs:
        e (int): The maximum number of entities.
        d (int): The dimension of the object-centric vector.
        device (device): The device where the model and tensors are loaded.
        train (bool): The flag if the parameters are trained.
        preprocess (tensor->tensor): Reshape the yolo output into the unified format of the perceptiom module.
    """

    def __init__(self, e, d, device, train=False):
        super().__init__()
        self.e = e  # num of entities
        self.d = d  # num of dimension
        self.device = device
        self.train_ = train  # the parameters should be trained or not
        # self.model = self.load_model(path=str(config.root) + '/src/weights/yolov5/best.pt', device=device)
        # function to transform e * d shape, YOLO returns class labels,
        # it should be decomposed into attributes and the probabilities.
        # self.preprocess = FCNNPreprocess(device)

    def load_model(self, path, device):
        # print("Loading YOLO model...")
        yolo_net = attempt_load(weights=path)
        yolo_net.to(device)
        if not self.train_:
            for param in yolo_net.parameters():
                param.requires_grad = False
        return yolo_net

    def forward(self, imgs):
        pred = self.model(imgs)[0]  # yolo model returns tuple
        # yolov5.utils.general.non_max_supression returns List[tensors]
        # with lengh of batch size
        # the number of objects can vary image to iamge
        yolo_output = self.pad_result(non_max_suppression(pred, max_det=self.e))
        return self.preprocess(yolo_output)

    def pad_result(self, output):
        """Padding the result by zeros.
            (batch, n_obj, 6) -> (batch, n_max_obj, 6)
        """
        padded_list = []
        for objs in output:
            if objs.size(0) < self.e:
                diff = self.e - objs.size(0)
                zero_tensor = torch.zeros((diff, self.e)).to(self.device)
                padded = torch.cat([objs, zero_tensor], dim=0)
                padded_list.append(padded)
            else:
                padded_list.append(objs)
        return torch.stack(padded_list)


def get_perception_predictions(args, file_path):
    # train_pos_loader, val_pos_loader, test_pos_loader = get_data_pos_loader(args)
    # train_neg_loader, val_neg_loader, test_neg_loader = get_data_neg_loader(args)
    # if args.dataset_type == "kandinsky":
    #     pm_val_res_file = str(config.buffer_path / f"{args.dataset}_pm_res_val.pth.tar")
    #     pm_train_res_file = str(config.buffer_path / f"{args.dataset}_pm_res_train.pth.tar")
    #     pm_test_res_file = str(config.buffer_path / f"{args.dataset}_pm_res_test.pth.tar")
    #
    #     val_pos_pred, val_neg_pred = percept.eval_images(args, pm_val_res_file, args.device, val_pos_loader,
    #                                                      val_neg_loader)
    #     train_pos_pred, train_neg_pred = percept.eval_images(args, pm_train_res_file, args.device, train_pos_loader,
    #                                                          train_neg_loader)
    #     test_pos_pred, test_neg_pred = percept.eval_images(args, pm_test_res_file, args.device, test_pos_loader,
    #                                                        test_neg_loader)

    train_pos_pred, train_neg_pred = get_pred_res(args, "train", file_path)
    test_pos_pred, test_neg_pred = get_pred_res(args, "test", file_path)
    val_pos_pred, val_neg_pred = get_pred_res(args, "val", file_path)

    log_utils.add_lines(f"- positive image number: {len(val_pos_pred)}", args.log_file)
    log_utils.add_lines(f"- negative image number: {len(val_neg_pred)}", args.log_file)
    pm_prediction_dict = {
        'val_pos': val_pos_pred,
        'val_neg': val_neg_pred,
        'train_pos': train_pos_pred,
        'train_neg': train_neg_pred,
        'test_pos': test_pos_pred,
        'test_neg': test_neg_pred
    }
    return pm_prediction_dict


def remove_zero_tensors(pm_prediction_dict, prob_index):
    for data_name, data in pm_prediction_dict.items():
        objs_all = []
        for img_i in range(data.shape[0]):
            objs_img_all = []
            valid_objs = data[img_i][data[img_i, :, prob_index] > 0]
            objs_all.append(valid_objs)
        pm_prediction_dict[data_name] = objs_all

    return pm_prediction_dict


def get_pred_res(args, data_type, file_path):
    od_res = file_path / f"{args.dataset}_pm_res_{data_type}.pth.tar"
    pred_pos, pred_neg = logic_utils.convert_data_to_tensor(args, od_res)

    # normalize the position
    pred_pos_norm = logic_utils.vertex_normalization(pred_pos)
    pred_neg_norm = logic_utils.vertex_normalization(pred_neg)

    # order the data by vertices (align the axis with higher delta)
    # pred_pos_ordered = logic_utils.data_ordering(pred_pos_norm)
    # pred_neg_ordered = logic_utils.data_ordering(pred_neg_norm)

    if args.top_data < len(pred_neg_norm):
        pred_pos_norm = pred_pos_norm[:args.top_data]
        pred_neg_norm = pred_neg_norm[:args.top_data]

    return pred_pos_norm, pred_neg_norm
