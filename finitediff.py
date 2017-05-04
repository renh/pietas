#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import helper
from constant import PI

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
     


def finite_difference(psi_b, psi_f, bands_contrib, param):
    """
    Finite difference calculation for dpsi and psi_0.
    Args:
        psi_b, psi_f  : WaveFunction objects correspond to the backward and forward
                               displaced geometry.
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

    # In practice, we first construct the inner product matrix between backward and forward
    # displaced wavefunctions:
    #    A[m,n] = < Psi_b[m] | Psi_f[n] >
    # this will be used in the calculation of finite-differenced wavefunctions ( |Psi_0_fd> ),
    # and the inter-band mixing coefficients
    #A = np.zeros([nbands_calc, nbands_calc], dtype=np.complex128)

    #psi_b_calc = psi_b.getWAE(OccWeighted=False)[band_init : band_final+1]
    psi_b_calc = psi_b.getWPS()[band_init : band_final+1]
    nplw = len(psi_b_calc[0])
    assert(nplw == psi_b.getNPlw())
    print("\n  Wavefunction Psi^{-} read.")

    #psi_f_calc = psi_f.getWAE(OccWeighted=False)[band_init : band_final+1]
    psi_f_calc = psi_f.getWPS()[band_init : band_final+1]
    print("  Wavefunction Psi^{+} read.")
    
    A = np.dot(
        np.conj(psi_b_calc), psi_f_calc.T
    )

    # renormalize A that the wavefunction from which be projected will just
    # rotate with a phase factor with norm 1.
    A_norm = np.sqrt(np.conj(A) * A)
    A = A / A_norm
    assert(np.allclose(np.ones(A.shape), np.conj(A)*A))
    print("\n  Phase factor matrix < Psi^{-} | Psi^{+} > calculated.")



    # calculate the principal part of change dpsi_P and psi_0_fd
    #                  1                          | Psi_b(m) >
    # | dpsi_P(m) > = --- [ | Psi_f(m) > - ------------------------ ]
    #                 2f                    < Psi_f(m) | Psi_b(m) >
    #
    #                    1                          | Psi_b(m) >
    # | psi_0_fd(m) > = --- [ | Psi_f(m) > + ------------------------ ]
    #                    2                    < Psi_f(m) | Psi_b(m) >

    psi_b_calc_proj = []
    for ibm in range(nbands_calc):
        # first project Psi_b onto Psi_f, to eliminte the random phase difference
        dump = psi_b_calc[ibm] / np.conj(A[ibm,ibm])
        psi_b_calc_proj.append(dump) # / np.sqrt(inner_product(dump, dump)))
    psi_b_calc_proj = np.array(psi_b_calc_proj)
    dpsi_P = (psi_f_calc - psi_b_calc_proj) / (2.0 * scale)
    print("\n  Principal part dPsi_P calculated.")
    psi_0_fd = (psi_f_calc + psi_b_calc_proj) / 2.0
    print("  Finite-differenced equilibrium wavefunction calculated.")

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
    # using the inner product matrix, we have
    #             1    1     A[m,n]     A[n,m]*
    #   C[m,n] = ---- --- [ -------- - ---------]
    #             2f   2     A[m,m]     A[n,n]*
    #
    
    C = np.zeros([nbands_calc, nbands_calc], dtype=np.complex128)
    for ibm in range(nbands_calc):
        for ibn in range(nbands_calc):
            C[ibm, ibn] = A[ibm,ibn] / A[ibm,ibm] - np.conj(A[ibn,ibm] / A[ibn,ibn])
    C /= (4.0 * scale)
    print('\n  Interband mixing matrix calculated.')

    dpsi_I = np.zeros([nbands_calc, nplw], np.complex128)
    for ibn in range(nbands_calc):
        n = ibn + band_init
        for ibm in range(nbands_calc):
            m = ibm + band_init
            if ibm == ibn:
                dpsi_I[ibn] += psi_0_fd[ibn] * (E_f[n] - E_b[n]) * helper.Gaussian(
                    E_f[n], E_b[n], sigma
                )
            else:
                #dpsi_I[ibn] += psi_0_fd[ibm] * C[ibm,ibn] * (E_0_fd[n] - E_0_fd[m]) * helper.Gaussian(
                #    E_0_fd[n], E_0_fd[m], sigma
                dpsi_I[ibn] += psi_0_fd[ibm] * np.dot(np.conj(psi_0_fd[ibm]), dpsi_P[ibn]) * (E_0_fd[n] - E_0_fd[m]) * helper.Gaussian(
                    E_0_fd[n], E_0_fd[m], sigma
                )
                
    dpsi_I *= (-1.0j*PI)
    print("  Inelastic part dPsi_I calculated.")
    psi_fd = {}
    psi_fd['psi_0_fd'] = psi_0_fd
    psi_fd['dpsi_P'] = dpsi_P
    psi_fd['dpsi_I'] = dpsi_I
    psi_fd['weights_fd'] = gw_fd[band_init : band_final+1]
    return psi_fd
                

    

    


    
    return
    
