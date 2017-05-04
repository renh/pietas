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
        v1     : real or ndarray, quantities for Gaussian weights
        v2     : real, the reference value
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

def getBandsRange(psi_0, psi_b, psi_f, param):
    """
    Get the bands range in which bands has non-negligible contributions at the Fermi level.
        The contributions is weighted by a Gaussian function with width 'sigma', and 
        non-negligible contribution means the Gaussian weights no less than 'cutoff'

    Args:
        psi_0, psi_b, psi_f : WaveFunction objects, contain all the necessary info about the 
                              wavefunction (pseudo and all-electron) at this (ispin, ik)
        param               : input configuration, augmented with some electronic structure 
                              info read from a static VASP calculation of the equilibrium geom.
    Returns:
        bands_contrib       : dictionary.
                              bands_contrib['bands_range'] : ndarray with two elements, correspond
                                                             to band_init, and band_final
                              bands_contrib['weights']     : ndarray with (band_final - band_init + 1)
                                                             elements, correspond to the Gaussian
                                                             weights for each band considered.
    """
    # parameters
    cutoff = param.get('cutoff')
    sigma = param.get('sigma')
    EF = param.get('Fermi energy')
    nbands = param.get('nbands')

    # eigen-engies
    E_0 = psi_0.getEig()
    E_b = psi_b.getEig()
    E_f = psi_f.getEig()

    # gaussian weights
    gw = np.zeros([nbands,4])
    gw[:,0] = Gaussian(E_0, EF, sigma)
    gw[:,1] = Gaussian(E_b, EF, sigma)
    gw[:,2] = Gaussian(E_f, EF, sigma)
    gw[:,3] = np.max(gw[:,:3], axis=1)

    # find band_init, band_final
    in_range = gw[:,3] > cutoff
    band_init = 0
    band_final = nbands - 1
    for ib in range(nbands):
        if in_range[band_init]:
            break
        band_init += 1
    for ib in range(nbands):
        if in_range[band_final]:
            break
        band_final -= 1
    bands_range = [band_init, band_final]

    # weights
    # note that the weights here is just for quantitative means, so only data 
    # calculated from the eigenenergies of the un-displaced data is reported.
    nbands_calc = band_final - band_init + 1
    weights = []
    for i in range(nbands_calc):
        weights.append(gw[i+band_init,0])

    bands_contrib = {}
    bands_contrib['bands_range'] = np.array(bands_range)
    bands_contrib['weights'] = np.array(weights)

    return bands_contrib
        
    




