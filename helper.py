#!/usr/bin/env python
# -*- coding: utf-8 -*-

from constant import *
import numpy as np

sqrtPI = 1.7724538509055159

def Gaussian(v1, v2, sigma):
    '''
    Gaussian function of width \sigma for two values v1 and v2:
        G(v1, v2, sigma) = \frac{1}{\sqrt{\pi} \sigma} *
                           \exp[-(v1 - v2)^2 / \sigma^2]

    Args:
        v1, v2 : real or ndarray, two (set) quantities for Gaussian weights
        sigma  : real, Gaussian width
    Returns:
        gaussian : real or ndarray, depend on the type of v1 and v2. Gaussian weights
    Raises:
        None
    '''
    
    dv = v1 - v2
    dv_over_sigma = dv / sigma
    exponential = - np.square(dv_over_sigma)
    gaussian = np.exp(exponential)
    #gaussian /= (sqrtPI * sigma)
    return gaussian

def iGaussian(cutoff, sigma):
    '''
    Inverse Gaussian function return a value difference with specified width and cutoff.
         cutoff = \frac{1}{\sqrt{\pi} \sigma} *
                  \exp[-dv^2 / \sigma^2]

    Args: 
        cutoff : real, cutoff value for the contribution
        sigma  : real, Gaussian width
    Returns:
        dv     : difference between variable and the reference, with which the contribution
                 is less than the cutoff.
    '''
    sqrtPI = 1.7724538509055159    
    exponential = cutoff * sqrtPI * sigma
    dv_over_sigma_2 = -np.log(exponential)
    dv_over_sigma = np.sqrt(dv_over_sigma_2)
    dv = dv_over_sigma * sigma
    return dv



