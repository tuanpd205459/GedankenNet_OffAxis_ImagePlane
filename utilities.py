import torch
import numpy as np
import scipy.io
import torch.nn as nn
from functools import reduce
import operator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))
    return c
