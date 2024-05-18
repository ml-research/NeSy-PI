import torch
import torch.nn as nn
import torch.nn.functional as F

import aitk.valuation
import aitk.valuation_yolo

from aitk.utils.fol import Atom
from aitk.utils.fol import bk


class PIValuationModule(nn.Module):
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

        if dataset_type == "kandinsky":
            v_phi = aitk.valuation_yolo.YOLOPhiValuationFunction(device)
            vfs['phi'] = v_phi
            layers.append(v_phi)

            v_rho = aitk.valuation_yolo.YOLORhoValuationFunction(device)
            vfs['rho'] = v_rho
            layers.append(v_rho)

            v_group_shape = aitk.valuation_yolo.YOLOGroupShapeValuationFunction(device)
            vfs['group_shape'] = v_group_shape
            layers.append(v_group_shape)

        elif dataset_type == "hide":
            v_phi = aitk.valuation.FCNNPhiValuationFunction(device)
            vfs['phi'] = v_phi
            layers.append(v_phi)

            v_rho = aitk.valuation.FCNNRhoValuationFunction(device)
            vfs['rho'] = v_rho
            layers.append(v_rho)

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

    def forward(self, head_atom, clauses, V, G):
        """Convert the object-centric representation to a valuation tensor.

            Args:
                zs (tensor): The object-centric representaion (the output of the YOLO model).
                atom (atom): The target atom to compute its proability.

            Returns:
                A batch of the probabilities of the target atom.
        """

        # evaluate each clauses, choose the max value
        atom_eval_values = torch.zeros(len(clauses))
        sub_dict = {}
        for term in head_atom.terms:
            sub_dict[term] = ""
        # the atom probabilities decided by the highest value of one of its bodies
        for i, clause in enumerate(clauses):
            clause_value = 0
            for body_atom in clause:
                # get evaluated value of the body atom
                if body_atom.pred.name == "in":
                    continue
                body_atom_list = list(body_atom.terms)
                for i in range(len(body_atom_list)):
                    # search for a substitution for each body terms
                    if body_atom_list[i] not in sub_dict.values():
                        for sub_key in sub_dict.keys():
                            if sub_dict[sub_key] == "":
                                sub_dict[sub_key] = body_atom_list[i]
                                break
                    for sub in sub_dict:
                        if sub_dict[sub] == body_atom_list[i]:
                            body_atom_list[i] = sub
                # body_atom_list[i] = [sub for sub in sub_dict if sub_dict[sub] == body_atom_list[i]][0]
                terms = tuple(body_atom_list)
                pred = body_atom.pred
                eval_atom = Atom(pred, terms)
                # TODO: consider the case with hidden terms in the body

                body_atom_index = G.index(eval_atom)
                body_atom_value = V[0, body_atom_index]
                clause_value += body_atom_value
            clause_value_avg = clause_value / (len(clause) - 2)
            atom_eval_values[i] = clause_value_avg

        atom_value = atom_eval_values.max()
        return atom_value

    def ground_to_tensor(self, term, zs):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.

            Return:
                The tensor representation of the input term.
        """
        term_index = self.lang.term_index(term)
        if term.dtype.name == 'object':
            return zs[:, term_index]
        elif term.dtype.name in bk.attr_names:
            return self.attrs[term]
        elif term.dtype.name == 'image':
            return None
        else:
            assert 0, "Invalid datatype of the given term: " + str(term) + ':' + term.dtype.name