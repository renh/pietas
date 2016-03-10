#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import scipy.fftpack

def calc_LDOS(psi, weights, index, param):
    """
    Calculate LDOS alike quantities from wavefunction psi and corresponding weight.
        \rho (r) = sum_{ib} < psi_{ib} |r >< r| psi_{ib} > * weight
    Args:
        psi     : ndarray (nb x nplw), Fourier coefficients for the nb bands considered in this calculation.
        weights : ndarray (nb x 1), contribution weights for each band
        param   : dictionary, input configuration and some DFT parameters.
            including:
            index  :  list of tuples (nplw * 3), index pointer to allocate the nplw coefficients into a 3D grid
            NGF    :  ndarray (3 x 1), FFT grid numbers
    """
    NGF = param.get('NGF')
    #index = param.get('index')
    nbands_calc = len(psi)


    rho = np.zeros(NGF)
    for ibm in range(nbands_calc):
        psi_i = np.zeros(NGF, dtype=np.complex128)
        for ind, C in zip(index, psi[ibm]):
            psi_i[ind] = C
        psi_i = scipy.fftpack.ifftn(psi_i)
        dump = np.conj(psi_i) * psi_i
        rho += (np.real(dump) * weights[ibm])

    return rho
    
       
        
        
    


def TersoffHamann(psi_fd, psi_0_orig, index, param):
    """
    Calculate LDOS and change in LDOS using the Tersoff-Hamann approximation.
    
    Args:
        psi_fd : dictionary. 
            psi_0_fd :  wavefunction for un-disturbed geometry
            dpsi_P   :  principal part of the change in wavefunction upon displacement
            dpsi_I   :  inelastic part of the change in wavefunction upon displacement
            weights_fd: Gaussian weights for the contribution of specific band at Fermi level
        param : input configuration
    Returns:
        None
    Raises:
        None
    Writes: (upon input configuration)
        rho0   : LDOS
        drhoP  : principal part of change in LDOS
        drhoI  : inelastic part of change in LDOS
        drhoT  : total change in LDOS
        itaP   : principal IETS efficiency 
        itaI   : inelastic IETS efficiency
        itaT   : total IETS efficiency
    """
    print('\nTersoff-Hamann calculation for LDOS...', end=' ')

    psi_0_fd = psi_fd['psi_0_fd']
    dpsi_P = psi_fd['dpsi_P']
    dpsi_I = psi_fd['dpsi_I']
    gw = psi_fd['weights_fd']

    q_out = param.get('output quantities')
    fmt_out = param.get('output formats')

    rho_0_orig = calc_LDOS(psi_0_orig, gw, index, param)
    rho_0_fd = calc_LDOS(psi_0_fd, gw, index, param)
    drho_P = calc_LDOS(dpsi_P, gw, index, param)
    drho_I = calc_LDOS(dpsi_I, gw, index, param)

    print ("FNISHED.")
    rho_this = {}
    rho_this['rho_0_orig'] = rho_0_orig
    rho_this['rho_0_fd'] = rho_0_fd
    rho_this['drho_P'] = drho_P
    rho_this['drho_I'] = drho_I
    return rho_this
