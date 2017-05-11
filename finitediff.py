#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import helper
from constant import PI
import normalcar as nm
import wavefunction as wf
import checker
import overlap

def inner_product(psi1, psi2):
    """
    Calculate the inner product between |psi1> and |psi2>.
    Args:
        psi1, psi2   :  ndarray (nplw x 1), (complex) Fourier coefficients of wavefunctions.
    Returns:
        innp         :  complex, the inner product
                        innp = < psi1 | psi2>
    Raises           :  
        AssertionError for different size of psi1 and psi2
    """
    assert( len(psi1) == len(psi2))
    innp = np.dot(
        np.conj(psi1), psi2
    )
    return innp
     


def finite_difference(psi_b, psi_f, pij_b, pij_f, qij, LMMax, bands_contrib, param):
    """
    Finite difference calculation for dpsi and psi_0.
    Args:
        psi_b, psi_f  : WaveFunction objects @(ispin,ik) correspond to the 
                                backward and forward displaced geometry.
        pij_b, pij_f  : 2D numpy array (NPROJ x NB), np.complex128
                                PAW projector coefficients @(ispin, ik)
        qij           : 3D numpy array (LMDIM x LMDIM x NIONS), float
                                Augmented charges
        LMMax         : 2D numpy array with (NIonTypes x 2), integer
                                No. of (l,m) channels and No. of ions per type
        bands_contrib        : dictionay for band contribution at the Fermi level.

        param                : input configuration and some electronic structure info.
    Returns:
        psi_fd               : dictionary, including psi_0_fd, dpsi_P, dpsi_I
    Raises:
        None
    """
    print("\nPerform finite-difference for wavefunctions...")
    EF = param.get('Fermi energy')
    scale = param.get('scale factor')
    sigma = param.get('sigma')
    print("  FD scale factor     : {:.2f}".format(scale))
    print("  Gaussian broadening : {:.3f}".format(sigma))
    
    bands_range = bands_contrib.get('bands_range')
    band_init, band_final = bands_range
    nbands_calc = band_final - band_init + 1

    E_b = psi_b.getEig()
    E_f = psi_f.getEig()
    E_0_fd = (E_b + E_f) / 2.0
    gw_fd = helper.Gaussian(E_0_fd, EF, sigma)


    # We first evaluate the overlap matrix between WFs will be used in the calculation
    #    The overlap operator:
    #       \hat{S} = 1 + \sum_{I,i,j} |beta_i> Q_{ij} <beta_j|
    #    thus the overlap matrix has two components:
    #       1) the overlap between pseudo WFs:  SPS_{mn} = < WPS_m | WPS_n >
    #       2) the augmentation charge:
    #               AC_{mn} = \sum_{I,i,j} (p_i)^* Q_{ij} p_j
    #    and the whole overlap in PAW or USPP formalism can be obtained by
    #           S = SPS + AC
    #
    # Two components for each matrix element: 1) inner product beween ps WFs; 
    #   2) augmented charge
    #     
    #   first calculate the inner product
    wps_b = psi_b.getWPS()
    wps_f = psi_f.getWPS()
    S_ps = np.dot(
            np.conj(wps_f[band_init : band_final+1]), 
            wps_b[band_init : band_final+1].T
            )
    print('\n  inner product calculated.')

    # then augmented charge
    AC = np.zeros([nbands_calc, nbands_calc], dtype=np.complex128)
    for m_band in range(nbands_calc):
        for n_band in range(nbands_calc):
            m_ind = m_band + band_init
            n_ind = n_band + band_init
            AC[m_band, n_band] = overlap.calc_aug_charge(
                    m_ind, n_ind, LMMax,
                    qij, pij_f, pij_b
                    )
            if n_band != m_band:
                AC[n_band, m_band] = np.conj(AC[m_band, n_band])
    print('\n  augmented charge calculated.')

    S = S_ps + AC

    fudge_factor = np.zeros(nbands_calc)

    print('\n  Overlap diagonal elements, small abs implies bands crossing')
    print('\n band\t    real\t\timag\t\t  phase/PI\t    abs'+'\n'+'-'*80)
    for i in range(nbands_calc):
        print('{:>4d}{:18.6E}{:18.6E}{:18.6f}{:15.6f}'.format(i+band_init+1,
            S[i,i].real, S[i,i].imag, np.angle(S[i,i])/np.pi, np.abs(S[i,i])
            ))
        if np.abs(S[i,i]) > 0.8: fudge_factor[i] = 1.0
    print('-'*80)

    # Need check the overlap < + | S | - >
    for m_band in range(nbands_calc):
        m_ind = m_band + band_init
        dump = wps_b[m_ind] / S[m_band, m_band]
        print(np.dot(np.conj(wps_f[m_ind]), dump.T))



    # calculate the principal part of change dpsi_P and psi_0_fd
    #                  1                          | Psi_b(m) >
    # | dpsi_P(m) > = --- [ | Psi_f(m) > - ------------------------ ]
    #                 2f                    < Psi_f(m) | Psi_b(m) >
    #
    #                    1                          | Psi_b(m) >
    # | psi_0_fd(m) > = --- [ | Psi_f(m) > + ------------------------ ]
    #                    2                    < Psi_f(m) | Psi_b(m) >
    print('  finite-difference for Psi_0 and principal part')
    nplw = psi_b.getNPlw()
    d_psi_P = np.zeros([nbands_calc, nplw], dtype=np.complex128)
    psi_0_fd = np.zeros([nbands_calc, nplw], dtype=np.complex128)

    for m_band in range(nbands_calc):
        m_ind = m_band + band_init
        d_psi_P[m_band] = wps_f[m_ind] - wps_b[m_ind] / S[m_band, m_band]
        psi_0_fd[m_band] = wps_f[m_ind] + wps_b[m_ind] / S[m_band, m_band]
        d_psi_P[m_band] *= fudge_factor[m_band]
        psi_0_fd[m_band] *= fudge_factor[m_band]
    d_psi_P /= (2.0 * scale)
    psi_0_fd /= 2.0


    #
    # calculate the inelastic part of change dpsi_I
    #
    # calculate the interband mixing coefficients
    # C_mn =\frac{1}{2*2f} [
    #        \frac{\ket{psi_m^b} \bra{psi_n^f}}{\ket{psi_m^b} \bra{psi_m^f}} -
    #        \frac{\ket{psi_m^f} \bra{psi_n^b}}{\ket{psi_n^f} \bra{psi_n^b}}
    #       ]
    #   This coefficient is equvilent to the inner product between wavefunctions
    #       \psi_m and d\psi_n, with m != n.
    #
    # the C matrix for for the inelastic part of delta_Psi
    #             1    1     S[n,m]*     S[m,n]
    #   C[m,n] = ---- --- [ --------- - ---------]
    #             2f   2     S[m,m]*     S[n,n]
    #
    print('\n  construct the coefficients matrix C[m,n]')
    C = np.zeros([nbands_calc, nbands_calc], dtype=np.complex128)
    for m in range(nbands_calc):
        for n in range(nbands_calc):
            C[m,n] = np.conj(S[n,m] / S[m,m]) - S[m,n] / S[n,n]
    C /= (4.0 * scale)
    print('\n  Interband mixing matrix calculated.')

    # real calculation for dPsi_I
    d_psi_I = np.zeros([nbands_calc, nplw], np.complex128)
    for ibn in range(nbands_calc):
        n = ibn + band_init
        for ibm in range(nbands_calc):
            m = ibm + band_init
            if ibm == ibn:
                d_psi_I[ibn] += psi_0_fd[ibn] * (E_f[n] - E_b[n]) * helper.Gaussian(
                    E_f[n], E_b[n], sigma) * fudge_factor[ibn]
            else:
                d_psi_I[ibn] += psi_0_fd[ibm] * C[ibm,ibn] * (E_0_fd[n] - E_0_fd[m]) * helper.Gaussian(
                    E_0_fd[n], E_0_fd[m], sigma)
    d_psi_I *= (-1.0j*PI)
    print("  Inelastic part dPsi_I calculated.")

    psi_fd = {}
    psi_fd['psi_0_fd'] = psi_0_fd
    psi_fd['d_psi_P'] = d_psi_P
    psi_fd['d_psi_I'] = d_psi_I
    psi_fd['weights_fd'] = gw_fd[band_init : band_final+1]
    # psi_fd['fudge_factor'] = fudge_factor

    return psi_fd
                
    
