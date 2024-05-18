# Created by shaji on 12-Apr-23

import torch


class Coder():
    def __init__(self, data):
        self.code = None
        self.dict_to_data = None
        self.dict_to_code = None

        self.coding(data)

    def coding(self, data):
        self.code = list(range(len(data)))
        self.dict_to_data = dict(zip(self.code, data))
        self.dict_to_code = dict(zip(data, self.code))


def get_substitutions(coder_const, coder_var):
    n_vars = len(coder_var.code)
    # e.g. if the data type is shape, then subs_consts_list = [(red,), (yellow,), (blue,)]
    const_permutations = itertools.permutations(coder_const.code, n_vars)

    theta_list = []
    # generate substitutions by combining variables to the head of subs_consts_list
    theta_list = torch.zeros(size=())
    for const_permutation in const_permutations:
        theta = []
        for i, const in enumerate(const_permutation):
            s = (coder_var.code[i], const)
            theta.append(s)
        theta_list.append(theta)

    return theta_list


def get_code_mat(args, consts, atoms, vars):
    coder_atoms = Coder(atoms)
    coder_vars = Coder(vars)
    coder_consts = Coder(consts)
    theta_list = get_substitutions(coder_consts, coder_vars)

    return None
