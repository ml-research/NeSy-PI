# Created by jing at 01.06.23

from aitk.utils.fol.language import DataType


def get_mode_declarations(e, lang):
    basic_mode_declarations = get_mode_declarations_kandinsky(lang, e)
    pi_model_declarations = get_pi_mode_declarations(lang, e)
    return basic_mode_declarations + pi_model_declarations


def get_mode_declarations_kandinsky(lang, obj_num):
    p_image = ModeTerm('+', DataType('image'))
    m_object = ModeTerm('-', DataType('object'))
    p_object = ModeTerm('+', DataType('object'))

    m_group = ModeTerm('-', DataType('group'))
    p_group = ModeTerm('+', DataType('group'))

    s_color = ModeTerm('#', DataType('color'))
    s_shape = ModeTerm('#', DataType('shape'))
    s_rho = ModeTerm('#', DataType('rho'))
    s_phi = ModeTerm('#', DataType('phi'))
    s_slope = ModeTerm('#', DataType('slope'))
    s_group_shape = ModeTerm('#', DataType('group_shape'))
    s_number = ModeTerm('#', DataType('number'))

    modeb_list = []
    considered_pred_names = [p.name for p in lang.preds]
    if "in" in considered_pred_names:
        modeb_list.append(ModeDeclaration('body', obj_num, lang.get_pred_by_name('in'), [m_group, p_image]))
    if "color" in considered_pred_names:
        modeb_list.append(ModeDeclaration('body', obj_num, lang.get_pred_by_name('color'), [p_group, s_color]))
    if "shape" in considered_pred_names:
        modeb_list.append(ModeDeclaration('body', obj_num, lang.get_pred_by_name('shape'), [p_group, s_shape]))
    if "rho" in considered_pred_names:
        modeb_list.append(
            ModeDeclaration('body', obj_num, lang.get_pred_by_name('rho'), [p_group, p_group, s_rho], ordered=False))
    if "phi" in considered_pred_names:
        modeb_list.append(
            ModeDeclaration('body', obj_num, lang.get_pred_by_name('phi'), [p_group, p_group, s_phi], ordered=False))
    if "slope" in considered_pred_names:
        modeb_list.append(
            ModeDeclaration('body', obj_num, lang.get_pred_by_name('slope'), [p_group, s_slope], ordered=False))
    if "group_shape" in considered_pred_names:
        modeb_list.append(
            ModeDeclaration('body', obj_num, lang.get_pred_by_name('group_shape'), [p_group, s_group_shape],
                            ordered=False))
    if "shape_counter" in considered_pred_names:
        modeb_list.append(
            ModeDeclaration('body', obj_num, lang.get_pred_by_name('shape_counter'), [p_group, s_number],
                            ordered=False))
    if "color_counter" in considered_pred_names:
        modeb_list.append(
            ModeDeclaration('body', obj_num, lang.get_pred_by_name('color_counter'), [p_group, s_number],
                            ordered=False))
    return modeb_list


def get_pi_mode_declarations(lang, obj_num):
    p_object = ModeTerm('+', DataType('group'))

    pi_mode_declarations = []
    for pi_index, pi in enumerate(lang.invented_preds):
        pi_str = pi.name
        objects = [p_object] * pi.arity
        mode_declarations = ModeDeclaration('body', obj_num, lang.get_invented_pred_by_name(pi_str), objects,
                                            ordered=False)
        pi_mode_declarations.append(mode_declarations)
    for pi_index, pi in enumerate(lang.bk_inv_preds):
        pi_str = pi.name
        objects = [p_object] * pi.arity
        mode_declarations = ModeDeclaration('body', obj_num, lang.get_bk_invented_pred_by_name(pi_str), objects,
                                            ordered=False)
        pi_mode_declarations.append(mode_declarations)
    return pi_mode_declarations


class ModeDeclaration(object):
    """from https://www.cs.ox.ac.uk/activities/programinduction/Aleph/aleph.html
    p(ModeType, ModeType,...)

    Here are some examples of how they appear in a file:

    :- mode(1,mem(+number,+list)).
    :- mode(1,dec(+integer,-integer)).
    :- mode(1,mult(+integer,+integer,-integer)).
    :- mode(1,plus(+integer,+integer,-integer)).
    :- mode(1,(+integer)=(#integer)).
    :- mode(*,has_car(+train,-car)).
    Each ModeType is either (a) simple; or (b) structured.
    A simple ModeType is one of:
    (a) +T specifying that when a literal with predicate symbol p appears in a
    hypothesised clause, the corresponding argument should be an "input" variable of type T;
    (b) -T specifying that the argument is an "output" variable of type T; or
    (c) #T specifying that it should be a constant of type T.
    All the examples above have simple modetypes.
    A structured ModeType is of the form f(..) where f is a function symbol,
    each argument of which is either a simple or structured ModeType.
    Here is an example containing a structured ModeType:


    To make this more clear, here is an example for the mode declarations for
    the grandfather task from
     above::- modeh(1, grandfather(+human, +human)).:-
      modeb(*, parent(-human, +human)).:-
       modeb(*, male(+human)).
       The  first  mode  states  that  the  head  of  the  rule
        (and  therefore  the  target predicate) will be the atom grandfather.
         Its parameters have to be of the type human.
          The  +  annotation  says  that  the  rule  head  needs  two  variables.
            The second mode declaration states the parent atom and declares again
             that the parameters have to be of type human.
              Here,  the + at the second parameter tells, that the system is only allowed to
              introduce the atom parent in the clause if it already contains a variable of type human.
               That the first attribute introduces a new variable into the clause.
    The  modes  consist  of  a  recall n that  states  how  many  versions  of  the
    literal are allowed in a rule and an atom with place-markers that state the literal to-gether
    with annotations on input- and output-variables as well as constants (see[Mug95]).
    Args:
        recall (int): The recall number i.e. how many times the declaration can be instanciated
        pred (Predicate): The predicate.
        mode_terms (ModeTerm): Terms for mode declarations.
    """

    def __init__(self, mode_type, recall, pred, mode_terms, ordered=True):
        self.mode_type = mode_type  # head or body
        self.recall = recall
        self.pred = pred
        self.mode_terms = mode_terms
        self.ordered = ordered

    def __str__(self):
        s = 'mode_' + self.mode_type + '('
        if self.mode_terms is None:
            raise ValueError
        for mt in self.mode_terms:
            s += str(mt)
            s += ','
        s = s[0:-1]
        s += ')'
        return s

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())


class ModeTerm(object):
    """Terms for mode declarations. It has mode (+, -, #) and data types.
    """

    def __init__(self, mode, dtype):
        self.mode = mode
        assert mode in ['+', '-', '#'], "Invalid mode declaration."
        self.dtype = dtype

    def __str__(self):
        return self.mode + self.dtype.name

    def __repr__(self):
        return self.__str__()
