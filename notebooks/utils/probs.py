import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm
import sys
#from .torch_imports import *
#from .torch_core import *
#from fastai.losses import FocalLossFlat as focal_loss

def RF_direct_prob(x, cl, w, p, m, s):
    """
    Gives probability of class conditional on RF-output
    x = RF output (between 0 and 1)
    cl = class (0 or 1)
    w = array of weights
    p = array of probs
    m = array of means for gaussian
    s = array of standard devs for gaussian
    """
    cl = 1 - cl
    c = 1 - 2**-10
    x2 = 0.5 + (x-0.5) * c
    y = np.log(x2/(1-x2))
    pc = p*cl + (1-p) * (1-cl)
    pjoint = np.sum(pc * w * norm.pdf(y, loc=m, scale=s))
    px = np.sum(w * norm.pdf(y, loc=m, scale=s))
    out = pjoint / px
    return out
def CNN_direct_prob(x, cl, w, p, m0, s0, m1, s1):
    """
    Gives probability of class conditional on CNN-output
    x = CNN output: vector of two real numbers
    cl = class (0 or 1)
    w = array of weights
    p = array of probs
    m0 = array of means for gaussian, output0
    s0 = array of standard devs for gaussian, output0
    m1 = array of means for gaussian, output1
    s1 = array of standard devs for gaussian, output1
    """
    pc = p*cl + (1-p)*(1-cl)
    pjoint = np.sum(w * pc * norm.pdf(x[0], loc=m0, scale=s0) * norm.pdf(x[1], loc=m1, scale=s1))
    px = np.sum(w * norm.pdf(x[0], loc=m0, scale=s0) * norm.pdf(x[1], loc=m1, scale=s1))
    prob = pjoint/px
    return prob
def CNN_direct_prob2(x, cl, w, p, m, s):
    """
    Gives probability of class conditional on CNN-output
    x = CNN output: vector of two real numbers
    cl = class (0 or 1)
    w = array of weights
    p = array of probs
    m0 = array of means for gaussian, output0
    s0 = array of standard devs for gaussian, output0
    m1 = array of means for gaussian, output1
    s1 = array of standard devs for gaussian, output1
    """
    
    pc = p*(1 - cl) + (1-p)*(cl)
    pjoint = np.sum(w * pc * norm.pdf(x[0], loc=m, scale=s) * norm.pdf(x[1], loc=m, scale=s))
    px = np.sum(w * norm.pdf(x[0], loc=m, scale=s) * norm.pdf(x[1], loc=m, scale=s))
    prob = pjoint/px
    return prob
#function for making a list of strings in df to list of floats
def get_substring_list(split):
    if len(split) <= 1:
        print('nooo')
    i = 0
    while len(split[i]) <= 1:
        i += 1
    s1 = split[i]
    i += 1
    while len(split[i]) <= 1:
        i += 1
    s2 = split[i]
    if s1[0] == '[':
        s1 = s1[1:]
    if s2[-1] == ']':
        s2 = s2[0:-1]
    s1 = np.float64(s1)
    s2 = np.float64(s2)
    return [s1, s2]