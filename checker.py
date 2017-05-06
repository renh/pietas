#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from wavefunction import WAVECAR
import numpy as np
import overlap
import filecmp

def check_WAVECAR(wavecar1, wavecar2):
    """
    Check consistency between two VASP WAVECAR files.
    Note that I only check the file size, file header and each K-Header loop over spin and kpoints

    Args:
        wc1, wc2   : class WAVECAR, connected to a VASP WAVECAR file
    Returns:
        consistent : logical, true for consistent, false for not
    Raises:
        AssertionErrors if inconsistency detected.
    """
    # first check the file size
    sz1 = os.path.getsize(wavecar1)
    sz2 = os.path.getsize(wavecar2)
    assert(sz1 == sz2)

    # connect to WAVECAR files using the wavefunction.WAVECAR class
    wc1 = WAVECAR(wavecar1)
    wc2 = WAVECAR(wavecar2)

    # check WAVECAR header information
    assert(wc1.getRECL()    ==  wc2.getRECL()  )
    assert(wc1.getNSpin()   ==  wc2.getNSpin() )
    assert(wc1.getTag()     ==  wc2.getTag())
    assert(wc1.getPrec()    ==  wc2.getPrec())
    assert(wc1.getNKpts()   ==  wc2.getNKpts())
    assert(wc1.getNBands()  ==  wc2.getNBands())
    assert(
        abs(wc1.getENMax() - wc2.getENMax()) < 1E-6
    )
    assert(
        np.allclose(wc1.getRealLatt(), wc2.getRealLatt())
    )

    # check each k-headers, loop over spin and kpts
    for ispin in range(wc1.getNSpin()):
        for ik in range(wc1.getNKpts()):
            kh1 = wc1.readKHeader(ispin, ik)
            kh2 = wc2.readKHeader(ispin, ik)

            # number of plane-waves
            np1 = kh1[0]; np2 = kh2[0]
            assert(np1 == np2)

            # k-vector
            kv1 = kh1[1]; kv2 = kh2[1]
            assert(
                np.allclose(kv1, kv2)
            )

            # number of eigvalues
            ne1 = len(kh1[2]); ne2 = len(kh2[2])
            assert(ne1 == ne2)

            # number of Fermi weights
            nw1 = len(kh1[3]); nw2 = len(kh2[3])
            assert(nw1 == nw2)

    print("\nVASP WAVECAR files:'{}' and '{}'\n\tconsistency check passed.".format(
        wavecar1, wavecar2
    ))
        
    return True

def same_file(wavecar1, wavecar2, wavecar3):
    """
    Check whether three files are the same by comparing their md5sum.
    Args:
        wavecar1, wavecar2, wavecar3   : filenames in string
    Returns:
        same   : logical, true for same, false for not
    Raises:
        None
    """
    same = False
    if (filecmp.cmp(wavecar1, wavecar2, shallow=False)):
        print("Serious problem:: {} and {} are the same".format(wavecar1, wavecar2))
        same = True
    if (filecmp.cmp(wavecar1, wavecar3, shallow=False)):
        print("Serious problem:: {} and {} are the same".format(wavecar1, wavecar3))
        same = True
    if (filecmp.cmp(wavecar2, wavecar3, shallow=False)):
        print("Serious problem:: {} and {} are the same".format(wavecar2, wavecar3))
        same = True

    if same:
        print("It seems that you are using same files to do finite difference, exit")
        print("\tComment the 'same_file' checker if you know what you are doing")
        raise SystemExit

def check_orthonormal(psi, LMMax, Qij, Pij):
    print('check orthonomality')
    wps = psi.getWPS()
    nb = len(wps)
    S_ps = np.dot(np.conj(wps), wps.T)
    print(S_ps.shape)

    AC = np.zeros([nb,nb],dtype=np.complex128)
    for mband in range(nb):
        if (not mband%40): print('mband = ', mband)
        for nband in range(nb):
            AC[mband,nband] = overlap.calc_aug_charge(
                    mband, nband, LMMax, Qij, Pij, Pij
                    )
            if mband != nband:
                AC[nband, mband] = np.conj(AC[mband, nband])

    S = S_ps + AC
    I_matrix = np.eye(nb, dtype=np.complex128)
    if np.allclose(I_matrix, S):
        print('  orthonormality check passed.')
    else:
        diff = I_matrix - S
        ind = np.argmax(np.abs(diff))
        max_diff = np.max(np.abs(diff))
        m,n = ind // nb, ind - (ind // nb * nb)
        print('  largest discrepancy occurs S({}, {}) = {}'.format(
            m, n, S[m,n]
                ))
        if max_diff < 1.E-6:
            print('  orthonormality check passed.')
        else:
            raise ValueError('  orthonormality check failed.')







    return

        
