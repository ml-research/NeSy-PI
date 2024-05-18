import os.path
import glob

from lark import Lark
from .exp_parser import ExpTree
from .language import DataType
from .logic import Predicate, InventedPredicate, FuncSymbol


class DataUtils(object):
    """Utilities about logic.

    A class of utilities about first-order logic.

    Args:
        dataset_type (str): A dataset type (kandinsky or clevr).
        dataset (str): A dataset to be used.

    Attrs:
        base_path: The base path of the dataset.
    """

    def __init__(self, lark_path, lang_base_path, dataset_type='kandinsky', dataset='twopairs'):
        self.base_path = lang_base_path / dataset_type / dataset
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_atom = Lark(grammar.read(), start="atom")
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_clause = Lark(grammar.read(), start="clause")

    # def load_clauses(self, lang, args):
    #     """Read lines and parse to Atom objects.
    #     """
    #     clauses = []
    #     init_clause = "kp(X):-."
    #     tree = self.lp_clause.parse(init_clause)
    #     clause = ExpTree(lang).transform(tree)
    #     clauses.append(clause)
    #     clause = clauses[0]
    #     clause.body = clause.body[:args.e]
    #     print("Initial clauses: ", clause)
    #
    #     return clauses

    def load_pi_clauses(self, bk_prefix, path, lang):
        """Read lines and parse to Atom objects.
        """
        clauses = []
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    if line[-1] == '\n':
                        line = line[:-1]
                    # substitude placeholder predicates to exist predicates
                    clause_candidates = self.get_clause_candidates(lang, line)
                    for clause_str in clause_candidates:
                        tree = self.lp_clause.parse(clause_str)
                        clause = ExpTree(lang).transform(tree)
                        clauses.append(clause)

        return clauses

    def gen_pi_clauses(self, lang, new_predicates, clause_str_list_with_score, kp_str_list):
        """Read lines and parse to Atom objects.
        """
        for n_p in new_predicates:
            lang.invented_preds.append(n_p[0])
        clause_str_list = []
        for c_str, c_score in clause_str_list_with_score:
            clause_str_list += c_str

        clauses = []
        for clause_str in clause_str_list:
            tree = self.lp_clause.parse(clause_str)
            clause = ExpTree(lang).transform(tree)
            clauses.append(clause)

        for str in kp_str_list:
            print(str)
        kp_clause = []
        for clause_str in kp_str_list:
            tree = self.lp_clause.parse(clause_str)
            clause = ExpTree(lang).transform(tree)
            kp_clause.append(clause)

        return clauses, kp_clause

    def get_clause_candidates(self, lang, clause_template):
        """

        Args:
            lang: language
            clause_template: a predicate invention template

        Returns:
            all the possible clauses, which satisfy the template by replacing the template_predicates to exist predicates
            in the language.
        """

        [head, body] = clause_template.split(":-")
        body_predicates = body.split(";")

        body_candidates = []
        for body_predicate in body_predicates:
            predicate_candidates = []
            pred_arity = len(body_predicate.split(","))
            arguments = body_predicate.split("(")[1].split(")")[0]

            for p in lang.preds:
                if p.arity == pred_arity:
                    predicate_candidates.append(p.name + "(" + arguments + ")")
            body_candidates.append(predicate_candidates)

        new_clauses = []
        for invented_preds in lang.invented_preds:
            clause_head = invented_preds.name
            arity = invented_preds.arity
            arguments = "(X,Y)" if arity == 2 else "(X)"
            clause_head += arguments

            new_clauses += [clause_head + ":-" + i + "." for i in body_candidates[0]]
        return new_clauses

    def load_atoms(self, path, lang):
        """Read lines and parse to Atom objects.
        """
        atoms = []

        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    if line[-1] == '\n':
                        line = line[:-2]
                    else:
                        line = line[:-1]
                    tree = self.lp_atom.parse(line)
                    atom = ExpTree(lang).transform(tree)
                    atoms.append(atom)
        return atoms

    # def load_target_predicate(self):
    #     preds = [self.parse_pred(line) for line in target_predicate]
    #     return preds

    # def load_neural_preds(self):
    #     preds = [self.parse_neural_pred(line) for line in neural_predicate]
    #     return preds

    def load_invented_preds(self, bk_prefix, path):
        f = open(path)
        lines = f.readlines()
        lines = [self.rename_bk_preds_in_clause(bk_prefix, line) for line in lines]
        preds = [self.parse_invented_bk_pred(bk_prefix, line) for line in lines]
        return preds

    def load_invented_clauses(self, bk_prefix, path, lang):
        f = open(path)
        lines = f.readlines()
        lines = [self.rename_bk_preds_in_clause(bk_prefix, line) for line in lines]
        clauses = [self.parse_invented_bk_clause(line, lang) for line in lines]
        return clauses

    def load_invented_preds_template(self, path):
        f = open(path)
        lines = f.readlines()
        preds = {}
        for line in lines:
            new_pred = self.parse_invented_pred(line)
            if new_pred is not None:
                if new_pred.arity == 1:
                    preds["1-ary"] = new_pred
                elif new_pred.arity == 2:
                    preds["2-ary"] = new_pred
                elif new_pred.arity == 3:
                    preds["3-ary"] = new_pred
                elif new_pred.arity == 4:
                    preds["4-ary"] = new_pred
                elif new_pred.arity == 5:
                    preds["5-ary"] = new_pred
        return preds

    # def load_consts(self, args):
    #     consts_str = []
    #     for const_name, const_type in consts.items():
    #         consts_str.extend(self.parse_const(args, const_name, const_type))
    #     return consts_str

    def parse_pred(self, line):
        """Parse string to predicates.
        """
        line = line.replace('\n', '')
        pred, arity, dtype_names_str = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]

        return Predicate(pred, int(arity), dtypes)

    def parse_invented_pred(self, line):
        """Parse string to invented predicates.
        """
        line = line.replace('\n', '')

        if (len(line)) == 0:
            return None
        pred, arity, dtype_names_str = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]
        assert int(arity) == len(dtypes), 'Invalid arity and dtypes in ' + pred + '.'
        pred_with_id = pred
        invented_pred = InventedPredicate(pred_with_id, int(arity), dtypes, args=None, pi_type=None)
        return invented_pred

    def rename_bk_preds_in_clause(self, bk_prefix, line):
        """Parse string to invented predicates.
        """
        new_line = line.replace('\n', '')
        new_line = new_line.replace('inv_pred', "inv_pred_bk" + str(bk_prefix) + "_")
        return new_line

    def parse_invented_bk_clause(self, line, lang):
        """Parse string to invented predicates.
        """

        tree = self.lp_clause.parse(line)
        clause = ExpTree(lang).transform(tree)

        return clause

    def parse_invented_bk_pred(self, bk_prefix, line):
        """Parse string to invented predicates.
        """
        head, body = line.split(':-')
        arity = len(head.split(","))
        head_dtype_names = arity * ['object']
        dtypes = [DataType(dt) for dt in head_dtype_names]

        # pred_with_id = pred + f"_{i}"
        pred_with_id = head.split("(")[0]
        invented_pred = InventedPredicate(pred_with_id, int(arity), dtypes, args=None, pi_type=None)

        return invented_pred

    def parse_funcs(self, line):
        """Parse string to function symbols.
        """
        funcs = []
        for func_arity in line.split(','):
            func, arity = func_arity.split(':')
            funcs.append(FuncSymbol(func, int(arity)))
        return funcs

    # def parse_const(self, args, const, const_type):
    #     """Parse string to function symbols.
    #     """
    #     const_data_type = DataType(const)
    #     if "amount_" in const_type:
    #         _, num = const_type.split('_')
    #         if num == 'e':
    #             num = args.e
    #         const_names = []
    #         for i in range(int(num)):
    #             const_names.append(str(const) + str(i))
    #     elif "enum" in const_type:
    #         if const == 'color':
    #             const_names = color
    #         elif const == 'shape':
    #             const_names = shape
    #         # elif const == 'group_shape':
    #         #     const_names = group_shape
    #         else:
    #             raise ValueError
    #     elif 'target' in const_type:
    #         const_names = ['image']
    #     else:
    #         raise ValueError
    #
    #     return [Const(const_name, const_data_type) for const_name in const_names]

    def parse_clause(self, clause_str, lang):
        tree = self.lp_clause.parse(clause_str)
        return ExpTree(lang).transform(tree)

    def get_bk(self, lang):
        return self.load_atoms(str(self.base_path / 'bk.txt'), lang)

    # def load_language(self, args):
    #     """Load language, background knowledge, and clauses from files.
    #     """
    #     # preds = self.load_target_predicate()
    #     # preds += self.load_neural_preds()
    #     # consts = self.load_consts(args)
    #     # pi_templates = self.load_invented_preds_template(str(self.base_path / 'neural_preds.txt'))
    #
    #     bk_inv_preds = []
    #     if args.with_bk:
    #         bk_pred_files = glob.glob(str(self.base_path / ".." / "bg_predicates" / "*.txt"))
    #         for bk_i, bk_file in enumerate(bk_pred_files):
    #             bk_inv_preds += self.load_invented_preds(bk_i, bk_file)
    #
    #
    #     return lang
