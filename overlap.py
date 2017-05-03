#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 Hao Ren <renh.cn@gmail.com>
#
# Distributed under terms of the LGPLv3 license.

"""
overlap calculation for PAW AE wavefunction verification.
"""
# Module import
from __future__ import print_function
import numpy as np
import wavefunction as wf
import normalcar as nm
#==============================================================================


def calc_aug_charge(bm, bn, LMMax, Qij, Pi, Pj):
    """
    Augmented charge for two bands from the same or different systems
    Args:
        bm, bn   : integer, band indicies for Psi_m and Psi_n
        LMMax    : integer 2D numpy array,
                     row  -- ion type
                     col1 -- No. of (l,m) channels
                     col2 -- No. of ions per type
        Qij      : float 4D numpy array [LMDIM, LMDIM, NIonsPerType, NType],
                   augmented charges for each pair of angular channels
        Pi, Pj   : complex 4D numpy array [NPRO, Nbands, Nkpts, Nspin]
                   projector coefficients
    Return:
        augchg   : complex
    """
    NTypes = len(LMMax)
    augchg = 0
    ion_ind = 0
    proj_ind = 0
    for itype in range(NTypes):
        NChannels, NIons = LMMax[itype]
        for iion in range(NIons):
            qij = Qij[:LMMax[itype,0], :LMMax[itype,0], ion_ind, 0]
            pi = Pi[proj_ind:proj_ind+LMMax[itype,0], bm, 0, 0]
            pj = Pj[proj_ind:proj_ind+LMMax[itype,0], bn, 0, 0]
            
            augchg += np.dot(np.conj(pi), np.dot(qij, pj))
            
            ion_ind += 1
            proj_ind += LMMax[itype, 0]
    return augchg



if __name__ == '__main__':
    wc = wf.WAVECAR('../test/WAVECAR')
    nmc = nm.NormalCAR(normalcar='../test/NormalCAR', 
            wavecar='../test/WAVECAR')
    
    # read pseudo wave function
    wps = wf.WaveFunction(wc, ispin=0, ik=0)
    psi_ps = wps.getWPS()
    print(psi_ps.shape)
    NBands = len(psi_ps)

    # read projector info
    LMDIM = nmc.getLMDim()
    Qij = nmc.getCQij()
    Pij = nmc.getProjCoeffs()
    LMMax = nmc.getLMMax()
    print(Qij.shape)
    print(Pij.shape)
    print(LMMax)

    # overlap between WPS
    S_PS = np.dot(
        np.conj(psi_ps), psi_ps.T 
            )
    print(S_PS.shape)

    
    # augmented charge
    S = np.zeros([NBands, NBands], dtype=np.complex128)
    for bm in range(NBands):
        for bn in range(bm, NBands):
            S[bm, bn] = S_PS[bm, bn] + \
                    calc_aug_charge(bm, bn, LMMax, Qij, Pij, Pij)
            if (bm != bn):
                S[bn, bm] = S[bm, bn]

    np.save('S.npy', S)


