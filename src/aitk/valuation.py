# Created by jing at 30.05.23
import torch
from torch import nn as nn
from torch.nn import functional as F

import config
from aitk.utils.fol import bk
from aitk.utils.neural_utils import LogisticRegression


class FCNNValuationModule(nn.Module):
    """A module to call valuation functions.
        Attrs:
            lang (language): The language.
            device (device): The device.
            layers (list(nn.Module)): The list of valuation functions.
            vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
            attrs (dic(term->tensor)): The dictionary that maps an attribute term to the corresponding one-hot encoding.
            dataset (str): The dataset.
    """

    def __init__(self, lang, device, dataset, dataset_type):
        super().__init__()
        self.lang = lang
        self.device = device
        self.layers, self.vfs = self.init_valuation_functions(device, dataset_type)
        # attr_term -> vector representation dic
        self.attrs = self.init_attr_encodings(device)
        self.dataset = dataset

    def init_valuation_functions(self, device, dataset_type=None):
        """
            Args:
                device (device): The device.
                dataset (str): The dataset.

            Retunrs:
                layers (list(nn.Module)): The list of valuation functions.
                vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
        """
        layers = []
        vfs = {}  # a dictionary: pred_name -> valuation function

        v_color = FCNNColorValuationFunction()
        vfs['color'] = v_color
        layers.append(v_color)

        v_shape = FCNNShapeValuationFunction()
        vfs['shape'] = v_shape
        layers.append(v_shape)

        v_in = FCNNInValuationFunction()
        vfs['in'] = v_in
        layers.append(v_in)

        v_rho = FCNNRhoValuationFunction(device)
        vfs['rho'] = v_rho
        layers.append(v_rho)

        v_phi = FCNNPhiValuationFunction(device)
        vfs['phi'] = v_phi
        layers.append(v_phi)

        v_slope = FCNNSlopeValuationFunction(device)
        vfs['slope'] = v_slope
        layers.append(v_slope)

        v_shape_counter = FCNNShapeCounterValuationFunction()
        vfs['shape_counter'] = v_shape_counter
        layers.append(v_shape_counter)

        v_color_counter = FCNNColorCounterValuationFunction()
        vfs['color_counter'] = v_color_counter
        layers.append(v_color_counter)

        return nn.ModuleList(layers), vfs

    def init_attr_encodings(self, device):
        """Encode color and shape into one-hot encoding.

            Args:
                device (device): The device.

            Returns:
                attrs (dic(term->tensor)): The dictionary that maps an attribute term to the corresponding one-hot encoding.
        """
        attr_names = bk.attr_names
        attrs = {}
        for dtype_name in attr_names:
            for term in self.lang.get_by_dtype_name(dtype_name):
                term_index = self.lang.term_index(term)
                num_classes = len(self.lang.get_by_dtype_name(dtype_name))
                one_hot = F.one_hot(torch.tensor(
                    term_index).to(device), num_classes=num_classes)
                one_hot.to(device)
                attrs[term] = one_hot
        return attrs

    def forward(self, zs, atom):
        """Convert the object-centric representation to a valuation tensor.

            Args:
                zs (tensor): The object-centric representaion (the output of the YOLO model).
                atom (atom): The target atom to compute its proability.

            Returns:
                A batch of the probabilities of the target atom.
        """
        if atom.pred.name in self.vfs:
            args = [self.ground_to_tensor(term, zs) for term in atom.terms]
            # call valuation function
            return self.vfs[atom.pred.name](*args)
        else:
            return torch.zeros((zs.size(0),)).to(
                torch.float32).to(self.device)

    def ground_to_tensor(self, term, zs):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.

            Return:
                The tensor representation of the input term.
        """
        term_index = self.lang.term_index(term)
        if term.dtype.name == 'group':
            return zs[:, term_index].to(self.device)
        elif term.dtype.name in bk.attr_names:
            return self.attrs[term].unsqueeze(0).repeat(zs.shape[0], 1).to(self.device)
        elif term.dtype.name == 'image':
            return None
        else:
            assert 0, "Invalid datatype of the given term: " + str(term) + ':' + term.dtype.name


class FCNNColorValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self):
        super(FCNNColorValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [x,y,z, (0:3)
                color1, color2, color3, (3:6)
                sphere, 6
                cube, 7
                ]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        # z_color = torch.zeros(size=z[:, 3:6].shape).to(z.device)
        # colors = z[:, 3:6]
        # for c in range(colors.shape[0]):
        #     c_index = torch.argmax(colors[c])
        #     z_color[c, c_index] = 1
        return (a * z[:, 3:6]).sum(dim=1)


class FCNNShapeValuationFunction(nn.Module):
    """The function v_shape.
    """

    def __init__(self):
        super(FCNNShapeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                [x,y,z, (0:3)
                color1, color2, color3, (3:6)
                sphere, 6
                cube, 7
                ]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        shape_indices = [config.group_tensor_index[_shape] for _shape in config.group_pred_shapes]
        z_shape = z[:, shape_indices]
        # a_batch = a.repeat((z.size(0), 1))  # one-hot encoding for batch
        return (a * z_shape).sum(dim=1)


class FCNNShapeCounterValuationFunction(nn.Module):
    def __init__(self):
        super(FCNNShapeCounterValuationFunction, self).__init__()

    def forward(self, z, a):
        attr_index = config.group_tensor_index["shape_counter"]
        z_shapeCounter = torch.zeros(a.shape)
        tensor_index = z[:, attr_index].to(torch.long)
        for i in range(len(tensor_index)):
            z_shapeCounter[i, tensor_index[i]] = 0.999
        return (a * z_shapeCounter).sum(dim=1)


class FCNNColorCounterValuationFunction(nn.Module):
    def __init__(self):
        super(FCNNColorCounterValuationFunction, self).__init__()

    def forward(self, z, a):
        attr_index = config.group_tensor_index["color_counter"]
        z_color_counter = torch.zeros(a.shape)
        tensor_index = z[:, attr_index].to(torch.long)
        for i in range(len(tensor_index)):
            z_color_counter[i, tensor_index[i]] = 0.999
        return (a * z_color_counter).sum(dim=1)


class FCNNInValuationFunction(nn.Module):
    """The function v_in.
    """

    def __init__(self):
        super(FCNNInValuationFunction, self).__init__()

    def forward(self, z, x):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            x (none): A dummy argment to represent the input constant.

        Returns:
            A batch of probabilities.
        """

        prob, _ = z[:, 6:14].max(dim=-1)
        return prob


class FCNNRhoValuationFunction(nn.Module):
    """The function v_area.
    """

    def __init__(self, device):
        super(FCNNRhoValuationFunction, self).__init__()
        self.device = device
        self.logi = LogisticRegression(input_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2, dist_grade):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            z_2 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]

        Returns:
            A batch of probabilities.
        """
        c_1 = self.to_center(z_1)
        c_2 = self.to_center(z_2)

        dir_vec = c_2 - c_1
        dir_vec[1] = -dir_vec[1]
        rho, phi = self.cart2pol(dir_vec[0], dir_vec[1])
        dist_id = torch.zeros(rho.shape)

        dist_grade_num = dist_grade.shape[1]
        grade_weight = 1 / dist_grade_num
        for i in range(1, dist_grade_num):
            threshold = grade_weight * i
            dist_id[rho >= threshold] = i

        dist_pred = torch.zeros(dist_grade.shape).to(dist_grade.device)
        for i in range(dist_pred.shape[0]):
            dist_pred[i, int(dist_id[i])] = 1

        return (dist_grade * dist_pred).sum(dim=1)

    def cart2pol(self, x, y):
        rho = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y, x)
        phi = torch.rad2deg(phi)
        return (rho, phi)

    def to_center(self, z):
        return torch.stack((z[:, 0], z[:, 2]))


class FCNNPhiValuationFunction(nn.Module):
    """The function v_area.
    """

    def __init__(self, device):
        super(FCNNPhiValuationFunction, self).__init__()
        self.device = device
        self.logi = LogisticRegression(input_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2, dir):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            z_2 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]

        Returns:
            A batch of probabilities.
        """
        c_1 = self.to_center(z_1)
        c_2 = self.to_center(z_2)

        round_divide = dir.shape[1]
        area_angle = int(360 / round_divide)
        area_angle_half = area_angle * 0.5
        # area_angle_half = 0
        dir_vec = c_2 - c_1
        dir_vec[1] = -dir_vec[1]
        rho, phi = self.cart2pol(dir_vec[0], dir_vec[1])
        phi_clock_shift = (90 - phi.long()) % 360
        zone_id = (phi_clock_shift + area_angle_half) // area_angle % round_divide

        # This is a threshold, but it can be decided automatically.
        # zone_id[rho >= 0.12] = zone_id[rho >= 0.12] + round_divide

        dir_pred = torch.zeros(dir.shape).to(dir.device)
        for i in range(dir_pred.shape[0]):
            dir_pred[i, int(zone_id[i])] = 1

        return (dir * dir_pred).sum(dim=1)

    def cart2pol(self, x, y):
        rho = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y, x)
        phi = torch.rad2deg(phi)
        return (rho, phi)

    def to_center(self, z):
        return torch.stack((z[:, 0], z[:, 2]))


class FCNNSlopeValuationFunction(nn.Module):
    """The function v_area.
    """

    def __init__(self, device):
        super(FCNNSlopeValuationFunction, self).__init__()
        self.device = device
        self.logi = LogisticRegression(input_dim=1)
        self.logi.to(device)

    def forward(self, z_1, dir):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            z_2 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]

        Returns:
            A batch of probabilities.
        """
        dir_pred = torch.zeros(dir.shape).to(dir.device)

        z_without_line = z_1[:, config.group_tensor_index["line"]] == 0
        c_1 = self.get_left(z_1)
        c_2 = self.get_right(z_1)

        round_divide = dir.shape[1]
        area_angle = int(180 / round_divide)
        area_angle_half = area_angle * 0.5
        # area_angle_half = 0
        dir_vec = c_2 - c_1
        dir_vec[1] = -dir_vec[1]
        rho, phi = self.cart2pol(dir_vec[0], dir_vec[1])
        phi[phi < 0] = 360 - torch.abs(phi[phi < 0])
        phi_clock_shift = (90 + phi.long()) % 360
        zone_id = (phi_clock_shift + area_angle_half) // area_angle % round_divide

        # This is a threshold, but it can be decided automatically.
        # zone_id[rho >= 0.12] = zone_id[rho >= 0.12] + round_divide

        for i in range(dir_pred.shape[0]):
            dir_pred[i, int(zone_id[i])] = 1

        dir_pred[z_without_line] = 0

        return (dir * dir_pred).sum(dim=1)

    def cart2pol(self, x, y):
        rho = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y, x)
        phi = torch.rad2deg(phi)
        return (rho, phi)

    def get_left(self, z):
        return torch.stack(
            (z[:, config.group_tensor_index["screen_left_x"]], z[:, config.group_tensor_index["screen_left_y"]]))

    def get_right(self, z):
        return torch.stack(
            (z[:, config.group_tensor_index["screen_right_x"]], z[:, config.group_tensor_index["screen_right_y"]]))


def get_valuation_module(args, lang):
    VM = FCNNValuationModule(lang=lang, device=args.device, dataset=args.dataset, dataset_type=args.dataset_type)
    return VM
