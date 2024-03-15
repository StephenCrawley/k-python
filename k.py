""" A simple k interpreter in python
    supports:
        int atom literals
        some basic array ops
        fenced expressions ()
        over(reduction) and scan(accumulate) operators
"""

import re
import operator
from enum import Enum
from functools import reduce
from itertools import accumulate

class NotYetImplementedError(Exception):
    pass

class LengthError(Exception):
    pass

class KType(Enum):
    OBJ_LIST = 0
    INT_ATOM = 1
    INT_LIST = 2
    OPERATOR = 3
    SYM_ATOM = 4

class K:
    """ A K object has:
            * a type code
            * a count
            * some data
            generic objects (KType.OBJ_LIST) point to other K objects
    """
    def __init__(self, ktype, ksize, kdata=None):
        self.ktype = ktype
        self.ksize = ksize
        self.kdata = kdata
        self.index = 0

    def __str__(self):
        if self.is_atom():
            return str(self.kdata)
        elif self.ksize == 0:
            return '()' if self.is_generic() else '!0'
        elif self.ksize == 1:
            string = str(self[0]) if self.is_int_list() else self[0].__str__()
            return ',' + string
        elif self.is_generic():
            string = ';'.join(x.__str__() for x in self.kdata)
            return '(' + string + ')'
        else: # int list
            return ' '.join(str(x) for x in self.kdata)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.ksize:
            self.index = 0
            raise StopIteration
        else:
            item = self.__getitem__(self.index)
            self.index += 1
            return item

    def __getitem__(self, index):
        if self.is_atom():
            return self
        elif isinstance(index, slice):
            data = self.kdata[index]
            return K(self.ktype, len(data), data)
        elif 0 <= index < self.ksize:
            item = self.kdata[index]
            if self.is_int_list():
                item = K.int_atom(item)
            return item
        else:
            raise IndexError("Index out of range")

    def is_int_atom(self):
        return self.ktype == KType.INT_ATOM

    def is_int_list(self):
        return self.ktype == KType.INT_LIST

    def is_atom(self):
        return self.is_int_atom() or self.ktype == KType.OPERATOR or self.is_sym_atom()

    def is_generic(self):
        return self.ktype == KType.OBJ_LIST

    def is_sym_atom(self):
        return self.ktype == KType.SYM_ATOM
    
    def is_operator(self, op):
        return self.ktype == KType.OPERATOR and self.kdata == op
    
    def expand(self):
        "if self is int list, return generic list of K int atoms"
        if not self.is_int_list():
            return self
        return K.obj_list(self.ksize, [K.int_atom(x) for x in self.kdata])
    
    def squeeze(self):
        "if self is an object list of int atoms, return an int list"
        if not self.is_generic():
            return self
        for k in self.kdata:
            if not k.is_int_atom():
                return self
        return K.int_list(self.ksize, [k.kdata for k in self.kdata])

    @staticmethod
    def obj_list(ksize, kdata):
        "create an generic K object list"
        return K(KType.OBJ_LIST, ksize, kdata)

    @staticmethod
    def int_atom(kdata):
        "create a K int atom"
        return K(KType.INT_ATOM, 1, kdata)

    @staticmethod
    def int_list(ksize, kdata):
        "create a K int list"
        return K(KType.INT_LIST, ksize, kdata)

    @staticmethod
    def operator(kdata):
        "create a K operator"
        return K(KType.OPERATOR, 1, kdata)

    @staticmethod
    def sym_atom(kdata):
        "create K symbol atom"
        return K(KType.SYM_ATOM, 1, kdata)

OPS = "+!,|*:"
ADVERBS = "\\/"

def tokenize(src):
    "return a list of tokens from source"
    tokens = re.split(r'([\d.]+|\W)', src)
    return [x for x in tokens if x and x != ' ']

def parse_num(tok):
    "convert a number token to a K number type"
    n = int(tok)
    x = K.int_atom(n)
    return x

def parse_adverb(tokens, x):
    "return (/;+) if next token is adverb"
    if len(tokens) == 0 or not tokens[0] in ADVERBS:
        return x
    adverb = K(KType.OPERATOR, 1, tokens.pop(0))
    return K.obj_list(2, [adverb, x])

def at_expr_end(tokens):
    return len(tokens) == 0 or tokens[0] in ');'

def parse_expressions(tokens):
    "parse single or ;-separated expressions"
    x = parse(tokens)
    if len(tokens) == 0 or tokens[0] != ';':
        return x
    x = cat(K(KType.OPERATOR, 1, '('), box(x))
    while len(tokens) != 0 and tokens[0] == ';':
        tokens.pop(0)
        x = cat(x, box(parse(tokens)))
    return x

def parse(tokens):
    "return a K parse tree from a list of tokens"

    if at_expr_end(tokens):
        raise SyntaxError(f"Unexpected token: {'EOL' if len(tokens)==0 else tokens[0]}")

    tok = tokens.pop(0)

    # parse (fenced expr)
    if tok == '(':
        x = parse_expressions(tokens)
        if len(tokens) == 0 or tokens.pop(0) != ')':
            raise SyntaxError("Expected ')'")
    # parse +x
    elif tok in OPS:
        x = K.operator(tok)
        x = parse_adverb(tokens, x)
        return K.obj_list(2, [x, parse(tokens)])
    # parse num
    elif tok.isdigit():
        x = parse_num(tok)
    elif tok.isalpha():
        x = K.sym_atom(tok)
    else:
        raise SyntaxError(f"Unexpected token: {tok}")

    if at_expr_end(tokens):
        return x

    # parse x+y
    tok = tokens.pop(0)
    if not tok in OPS:
        raise SyntaxError("Expected operator after number")
    y = K.operator(tok)
    y = parse_adverb(tokens, y)
    return K.obj_list(3, [y, x, parse(tokens)])

# K Operators

def nyi(*args):
    "raises NotYetImplementedError"
    raise NotYetImplementedError("not yet implemented")

def til(x):
    "!x (til x): Returns an enumeration [0,x)"
    if x.ktype != KType.INT_ATOM:
        raise ValueError("!x requires atomic int")
    return K.int_list(x.kdata, list(range(x.kdata)))

def box(x):
    ",x (box x): make x into a singleton list"
    new_type = KType.INT_LIST if KType.INT_ATOM == x.ktype else KType.OBJ_LIST
    new_data = x.kdata if  KType.INT_ATOM == x.ktype else x
    return K(new_type, 1, [new_data])

def rev(x):
    "|x (reverse x): reverses x"
    if x.is_int_atom():
        return x
    return K.int_list(x.ksize, list(reversed(x.kdata)))

def top(x):
    "*x (top x): first element of x"
    return x[0]

def idn(x):
    ":x (identity x): return x"
    return x

def binary_iter(op, x, y):
    "apply op(x,y)"
    if x.is_int_atom():
        if y.is_int_atom():
            return K.int_atom(op(x.kdata, y.kdata))
        else:
            return binary_iter(op, y, x)
    else:
        if y.is_int_atom():
            if x.is_generic():
                data = [binary_iter(op, xobj, y) for xobj in x]
                return K.obj_list(x.ksize, data)
            else:
                data = [op(xn, y.kdata) for xn in x.kdata]
                return K.int_list(x.ksize, data)
        else:
            if x.ksize != y.ksize:
                raise LengthError("x+y - operands must conform in length")
            elif x.is_generic() or y.is_generic():
                data = [binary_iter(op, xval, yval) for (xval,yval) in zip(x, y)]
                return K.obj_list(x.ksize, data)
            else:
                data = [op(xn, yn) for (xn,yn) in zip(x.kdata, y.kdata)]
                return K.int_list(x.ksize, data)

def add(x, y):
    "x+y (add x y): add 2 K objects"
    return binary_iter(operator.add, x, y)

def mul(x, y):
    "x*y (multiply x y): multiply 2 K objects"
    return binary_iter(operator.mul, x, y)

def cat(x, y):
    "x,y (concatenate x,y): joins two K objects"
    if x.is_atom():
        x = box(x)
    if y.ksize == 0:
        return x
    if y.is_atom():
        y = box(y)
    if x.ksize == 0:
        return y
    if x.is_int_list() and y.is_int_list():
        return K.int_list(x.ksize + y.ksize, x.kdata + y.kdata)
    else:
        x, y = x.expand(), y.expand()
        return K.obj_list(x.ksize + y.ksize, x.kdata + y.kdata)

def dex(x, y):
    "x:y (dex x y): return the right argument"
    return y

# ops tables
#             +    !    ,    |    *    :
unary_ops  = [nyi, til, box, rev, top, idn]
binary_ops = [add, nyi, cat, nyi, mul, dex]

def get_identity(op):
    "return identity element for certain operator"
    # identities: additive is 0, multiplicative is 1, cat is empty list
    if op.kdata in '+*':
        identity = K.int_atom(int(op.kdata == '*'))
    else: 
        identity = K.obj_list(0, [])
    return identity

ENV  = {}

def k_eval(tree):
    "evaluate the K parse tree"

    # numbers evaluate to themselves
    if tree.is_int_atom():
        return tree
    
    # get variable
    if tree.is_sym_atom():
        try:
            return ENV[tree.kdata]
        except:
            raise ValueError(f"'{tree.kdata}' not defined")

    # assign variable
    if tree.ksize == 3 and tree[0].is_operator(':') and tree[1].is_sym_atom():
        res = k_eval(tree[2])
        ENV[tree[1].kdata] = res
        return res

    # K has no operator precedence
    # all operators are right associative
    # so we evaluate the tree from right to left
    
    # else evaluate (+;..)
    head = tree[0]
    args = [k_eval(x) for x in tree.kdata[:0:-1]]
    args = list(reversed(args))
    if head.ktype == KType.OPERATOR and head.kdata == '(':
        return K.obj_list(len(args), args).squeeze()
    elif head.ksize == 1: # no adverb
        op_idx = OPS.find(head.kdata)
        func_list = unary_ops if len(args) == 1 else binary_ops
        return func_list[op_idx](*args)
    else: # adverb-modified operator
        adverb, op = head
        func = binary_ops[OPS.find(op.kdata)]
        # if unary +/x (as opposed to x+/y), append seed value
        if adverb.is_operator('/'):
            if len(args) == 1:
                args = [get_identity(op)] + args
            args = cat(box(args[0]), args[1])
            return reduce(lambda x,y: func(x,y), args)
        else:
            if len(args) == 2:
                arg0 = func(args[0], args[1][0])
                arg1 = cat(K.int_list(0, []), args[1])
                args = cat(box(arg0), arg1[1::])
            else:
                args = args[0]
            res = list(accumulate(args, lambda x,y: func(x,y)))
            return K.obj_list(len(res), res).squeeze()


def main():
    print('k is for "keys to the kingdom" ...')

    # interpreter loop
    while True:
        # read input
        src = input(' ').strip()

        # continue if no input
        if len(src) == 0:
            continue

        # exit is \
        if src[0] == '\\':
            exit(0)

        # else parse and evaluate
        try:
            tokens = tokenize(src)
            tree = parse(tokens)
            if len(tokens) != 0:
                raise SyntaxError(f"Unexpected token: {tokens.pop(0)}")
            res = k_eval(tree)
            print(res)
        except (ValueError, SyntaxError, NotYetImplementedError, LengthError) as e:
            print(f"'ERROR: {e}")

if __name__ == "__main__":
    main()


