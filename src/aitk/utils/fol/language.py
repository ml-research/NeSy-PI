import glob
from lark import Lark
import itertools
import re
from .exp_parser import ExpTree
from .logic import *
from . import bk


# from fol import mode_declaration

class Language(object):
    """Language of first-order logic.

    A class of languages in first-order logic.

    Args:
        preds (List[Predicate]): A set of predicate symbols.
        funcs (List[FunctionSymbol]): A set of function symbols.
        consts (List[Const]): A set of constants.

    Attrs:
        preds (List[Predicate]): A set of predicate symbols.
        funcs (List[FunctionSymbol]): A set of function symbols.
        consts (List[Const]): A set of constants.
    """

    def __init__(self, args, funcs, pi_type, level):
        self.vars = [Var(f"O{i + 1}") for i in range(args.group_e)]
        self.var_num = args.group_e
        self.atoms = []
        self.preds = []
        self.invented_preds = []
        self.invented_preds_with_scores = []
        self.funcs = funcs
        self.consts = []
        # self.clauses = []
        self.pi_clauses = []
        self.clause_with_scores = []
        # self.pi_templates = pi_templates

        ## BK
        self.bk_inv_preds = []
        self.all_invented_preds = []
        self.all_pi_clauses = []
        self.invented_preds_number = args.p_inv_counter

        self.base_path = args.lang_base_path / args.dataset_type / args.dataset
        with open(args.lark_path, encoding="utf-8") as grammar:
            self.lp_clause = Lark(grammar.read(), start="clause")
        with open(args.lark_path, encoding="utf-8") as grammar:
            self.lp_atom = Lark(grammar.read(), start="atom")

        self.load_lang(args, pi_type, level)
        # self.load_init_clauses(args.e)

    def __str__(self):
        s = "===Predicates===\n"
        for pred in self.preds:
            s += pred.__str__() + '\n'
        s += "===Function Symbols===\n"
        for func in self.funcs:
            s += func.__str__() + '\n'
        s += "===Constants===\n"
        for const in self.consts:
            s += const.__str__() + '\n'
        s += "===Invented Predicates===\n"
        for invented_predicates in self.invented_preds:
            s += invented_predicates.__str__() + '\n'
        return s

    def __repr__(self):
        return self.__str__()

    def generate_atoms(self):
        p_ = Predicate('.', 1, [DataType('spec')])
        false = Atom(p_, [Const('__F__', dtype=DataType('spec'))])
        true = Atom(p_, [Const('__T__', dtype=DataType('spec'))])

        spec_atoms = [false, true]
        atoms = []
        for pred in self.preds:
            dtypes = pred.dtypes
            consts_list = [self.get_by_dtype(dtype) for dtype in dtypes]
            args_list = list(set(itertools.product(*consts_list)))
            for args in args_list:
                if len(args) == 1 or len(set(args)) == len(args):
                    atoms.append(Atom(pred, args))
        pi_atoms = []
        for pred in self.invented_preds:
            dtypes = pred.dtypes
            consts_list = [self.get_by_dtype(dtype) for dtype in dtypes]
            args_list = list(set(itertools.product(*consts_list)))
            for args in args_list:
                if len(args) == 1 or len(set(args)) == len(args):
                    new_atom = Atom(pred, args)
                    if new_atom not in atoms:
                        pi_atoms.append(new_atom)
        bk_pi_atoms = []
        for pred in self.bk_inv_preds:
            dtypes = pred.dtypes
            consts_list = [self.get_by_dtype(dtype) for dtype in dtypes]
            args_list = list(set(itertools.product(*consts_list)))
            for args in args_list:
                # check if args and pred correspond are in the same area
                if pred.dtypes[0].name == 'area':
                    if pred.name[0] + pred.name[5:] != args[0].name:
                        continue
                if len(args) == 1 or len(set(args)) == len(args):
                    pi_atoms.append(Atom(pred, args))
        self.atoms = spec_atoms + sorted(atoms) + sorted(bk_pi_atoms) + sorted(pi_atoms)

    def load_init_clauses(self, e):
        """Read lines and parse to Atom objects.
        """
        init_clause = "kp(X):-"
        for i in range(e):
            init_clause += f"in(O{i + 1},X),"
        init_clause = init_clause[:-1]
        init_clause += "."
        tree = self.lp_clause.parse(init_clause)
        clauses = ExpTree(self).transform(tree)
        clauses = [clauses]
        return clauses

    def parse_pred(self, line, pi_type):
        """Parse string to predicates.
        """
        pred, arity, dtype_names_str = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]
        return NeuralPredicate(pred, int(arity), dtypes, pi_type)

    def parse_const(self, args, const, const_type, level):
        """Parse string to function symbols.
        """
        if level == "object":
            e = args.n_obj
        elif level == "group":
            e = args.group_e
        else:
            raise ValueError

        const_data_type = DataType(const)
        if "amount_" in const_type:
            _, num = const_type.split('_')
            if num == 'e':
                num = e
            elif num == "phi":
                num = args.phi_num
            elif num == "rho":
                num = args.rho_num
            elif num == "slope":
                num = args.slope_num

            const_names = []
            for i in range(int(num)):
                const_names.append(str(const) + str(i))
        elif "enum" in const_type:
            if const == 'color':
                const_names = bk.color
            elif const == 'shape':
                const_names = bk.shape
            # elif const == 'group_shape':
            #     const_names = group_shape
            else:
                raise ValueError
        elif 'target' in const_type:
            const_names = ['image']
        else:
            raise ValueError

        return [Const(const_name, const_data_type) for const_name in const_names]

    def load_consts(self, args, level):
        consts_str = []
        for const_name, const_type in bk.const_dict.items():
            consts_str.extend(self.parse_const(args, const_name, const_type, level))
        return consts_str

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

    def parse_invented_bk_pred(self, line):
        """Parse string to invented predicates.
        """
        head, body = line.split(':-')
        arity = len(head.split(","))
        head_dtype_names = arity * ['group']
        dtypes = [DataType(dt) for dt in head_dtype_names]

        # pred_with_id = pred + f"_{i}"
        pred_with_id = head.split("(")[0]
        invented_pred = InventedPredicate(pred_with_id, int(arity), dtypes, args=None, pi_type=None)

        return invented_pred

    def count_arity_from_clauses(self, clause_cluster):
        OX = "O"
        arity = 1
        arity_list = []
        id = clause_cluster[0].split("inv_pred")[1].split("(")[0]
        while (OX + str(arity)) in clause_cluster[0]:
            arity_list.append(OX + str(arity))
            arity += 1
        return arity_list, id

    def load_invented_preds(self, clauses_str, target_clauses_str):
        inv_p_clauses = []
        inv_preds = []
        # generate clauses
        for clause_str in clauses_str:
            inv_pred = self.parse_invented_bk_pred(clause_str)
            if inv_pred not in self.invented_preds:
                self.invented_preds.append(inv_pred)
            inv_preds.append(inv_pred)
            tree = self.lp_clause.parse(clause_str)
            clause = ExpTree(self).transform(tree)
            # generate clauses
            inv_p_clauses.append(clause)

        self.all_invented_preds = self.invented_preds
        self.all_pi_clauses = inv_p_clauses

        target_clauses = []
        for target_clause_str in target_clauses_str:
            target_clause_str = target_clause_str.replace(" ", "")
            tree = self.lp_clause.parse(target_clause_str)
            clause = ExpTree(self).transform(tree)
            # generate clauses
            target_clauses.append(clause)
        self.all_clauses = target_clauses
        # unique predicate
        new_predicates = []
        p_names = []
        for pred in inv_preds:
            if "inv" in pred.name and pred.name not in p_names:
                p_names.append(pred.name)
                new_predicates.append(pred)

        for inv_pred in self.invented_preds:
            inv_pred.body = []
            for c in inv_p_clauses:
                if c.head.pred.name == inv_pred.name:
                    inv_pred.body.append(c.body)

        self.update_inv()

    def update_inv(self):
        self.invented_preds = self.all_invented_preds
        self.pi_clauses = self.all_pi_clauses
        self.generate_atoms()

    def load_lang(self, args, pi_type, level):
        self.preds = [self.parse_pred(line, pi_type) for line in bk.target_predicate]
        # preds += self.load_neural_preds()
        self.consts = self.load_consts(args, level)
        # pi_templates = self.load_invented_preds_template(str(self.base_path / 'neural_preds.txt'))

    def get_var_and_dtype(self, atom):
        """Get all variables in an input atom with its dtypes by enumerating variables in the input atom.

        Note:
            with the assumption with function free atoms.

        Args:
            atom (Atom): The atom.

        Returns:
            List of tuples (var, dtype)
        """
        var_dtype_list = []
        for i, arg in enumerate(atom.terms):
            if arg.is_var():
                dtype = atom.pred.dtypes[i]
                var_dtype_list.append((arg, dtype))
        return var_dtype_list

    def get_by_dtype(self, dtype):
        """Get constants that match given dtypes.

        Args:
            dtype (DataType): The data type.

        Returns:
            List of constants whose data type is the given data type.
        """
        return [c for c in self.consts if c.dtype == dtype]

    def get_by_dtype_name(self, dtype_name):
        """Get constants that match given dtype name.

        Args:
            dtype_name (str): The name of the data type to be used.

        Returns:
            List of constants whose datatype has the given name.
        """
        return [c for c in self.consts if c.dtype.name == dtype_name]

    def term_index(self, term):
        """Get the index of a term in the language.

        Args:
            term (Term): The term to be used.

        Returns:
            int: The index of the term.
        """
        terms = self.get_by_dtype(term.dtype)
        return terms.index(term)

    def get_const_by_name(self, const_name):
        """Get the constant by its name.

        Args:
            const_name (str): The name of the constant.

        Returns:
            Const: The matched constant with the given name.

        """
        const = [c for c in self.consts if const_name == c.name]
        assert len(const) == 1, 'Too many match in ' + const_name
        return const[0]

    def get_pred_by_name(self, pred_name):
        """Get the predicate by its name.

        Args:
            pred_name (str): The name of the predicate.

        Returns:
            Predicate: The matched preicate with the given name.
        """
        pred = [pred for pred in self.preds if pred.name == pred_name]
        assert len(pred) == 1, 'Too many or less match in ' + pred_name
        return pred[0]

    def get_invented_pred_by_name(self, invented_pred_name):
        """Get the predicate by its name.

        Args:
            invented_pred_name (str): The name of the predicate.

        Returns:
            InventedPredicat: The matched invented predicate with the given name.
        """
        invented_pred = [invented_pred for invented_pred in self.invented_preds if
                         invented_pred.name == invented_pred_name]
        if not len(invented_pred) == 1:
            raise ValueError('Too many or less match in ' + invented_pred_name)
        return invented_pred[0]

    def get_bk_invented_pred_by_name(self, invented_pred_name):
        """Get the predicate by its name.

        Args:
            invented_pred_name (str): The name of the predicate.

        Returns:
            InventedPredicat: The matched invented predicate with the given name.
        """
        invented_pred = [invented_pred for invented_pred in self.bk_inv_preds if
                         invented_pred.name == invented_pred_name]
        if not len(invented_pred) > 0:
            raise ValueError('Too less match in ' + invented_pred_name)
        return invented_pred[0]

    def inv_pred(self, args, arity, pi_dtypes, p_args, pi_type):
        """Get the predicate by its id.

        Args:
            pi_template (str): The name of the predicate template.

        Returns:
            InventedPredicat: The matched invented predicate with the given name.
        """
        prefix = "inv_pred"
        new_predicate_id = self.invented_preds_number
        if args is not None:
            args.p_inv_counter += 1
            self.invented_preds_number = args.p_inv_counter
        pred_with_id = prefix + str(new_predicate_id)

        new_predicate = InventedPredicate(pred_with_id, int(arity), pi_dtypes, p_args, pi_type=pi_type)
        # self.invented_preds.append(new_predicate)

        return new_predicate

    def load_inv_pred(self, id, arity, pi_dtypes, p_args, pi_type):
        """Get the predicate by its id.

        Args:
            pi_template (str): The name of the predicate template.

        Returns:
            InventedPredicat: The matched invented predicate with the given name.
        """
        prefix = "inv_pred"
        # new_predicate_id = self.invented_preds_number

        pred_with_id = prefix + str(id)

        new_predicate = InventedPredicate(pred_with_id, int(arity), pi_dtypes, p_args, pi_type=pi_type)
        # self.invented_preds.append(new_predicate)

        return new_predicate

    def update_bk(self, neural_pred=None, full_bk=True):

        # put everything into the bk
        if full_bk:
            if neural_pred is not None:
                self.preds = self.preds[:2] + neural_pred
            self.invented_preds = self.all_invented_preds
            self.preds += self.invented_preds
            self.pi_clauses = self.all_pi_clauses
        else:

            # only consider one category by the given nerual pred
            self.preds = self.preds[:2]
            self.preds.append(neural_pred)
            self.invented_preds = []
            self.pi_clauses = []
        self.generate_atoms()

        # self.mode_declaration = mode_declaration.get_mode_declarations(args, self)

        # PM = get_perception_module(args)
        # VM = get_valuation_module(args, lang)
        # PI_VM = PIValuationModule(lang=lang, device=args.device, dataset=args.dataset, dataset_type=args.dataset_type)
        # FC = facts_converter.FactsConverter(lang=lang, perception_module=PM, valuation_module=VM,
        #                                     pi_valuation_module=PI_VM, device=args.device)
        # # Neuro-Symbolic Forward Reasoner for clause generation
        # NSFR_cgen = get_nsfr_model(args, lang, clauses, atoms, pi_clauses, FC)
        # PI_cgen = pi_utils.get_pi_model(args, lang, clauses, atoms, pi_clauses, FC)
        #
        # mode_declarations = get_mode_declarations(args, lang)
        # clause_generator = ClauseGenerator(args, NSFR_cgen, PI_cgen, lang, mode_declarations,
        #                                    no_xil=args.no_xil)  # torch.device('cpu'))

        # pi_clause_generator = PIClauseGenerator(args, NSFR_cgen, PI_cgen, lang,
        #                                         no_xil=args.no_xil)  # torch.device('cpu'))


class DataType(object):
    """Data type in first-order logic.

    A class of data types in first-order logic.

    Args:
        name (str): The name of the data type.

    Attrs:
        name (str): The name of the data type.
    """

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        if type(other) == str:
            return self.name == other
        else:
            return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())


p_ = Predicate('.', 1, [DataType('spec')])
false = Atom(p_, [Const('__F__', dtype=DataType('spec'))])
true = Atom(p_, [Const('__T__', dtype=DataType('spec'))])
