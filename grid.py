#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np

def length(vector):
    """
    Length of a real vector.
    Args:
        vector: array like with real elements.
    Returns:
        length (modulus) of the vector.
    Raise:
        None
    """
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector)
    return np.sqrt(
        np.sum(vector*vector)
    )

def real_grid(rgd, a):
    """
    Arrange real space grid with given grid density and lattice.
    Args:
        rgd :  real, real grid density in Angstrom
        a   :  real space lattice
    Returns:
        NGF :  ndarray (1x3), real space grids for rho, drho, ita, etc.
    Raises:
        None
    """
    #NGF = np.zeros(3, dtype=np.int)
    #for i in range(3):
    #    l = length(a[i])
    #    N = int(l / rgd)
    #    NGF[i] = N
    #return NGF
    lengths = np.sqrt(np.sum(a*a, axis=1))
    NGF = np.array(np.ceil(lengths/rgd), dtype=np.int)
    return NGF

def genIndex(GVEC, NGF):
    """
    Generate FFT index for a specific FFT grid NGF at a k point with G vectors GVEC.
    
    Args:
        GVEC  : ndarray (nplw x 1), G Vectors in the unit of reciprocal lattice vectors.
        NGF   : FFT grid for k-to-r fft
    Returns:
        index : index pointer to allocate the nplw Fourier coefficients in the wavefunction
                objects (nplw x 1) to a 3D grid (NGXF x NGYF x NGZF)
    Raises:
        None
    """
    index = []
    #print(GVEC.shape)
    #print(NGF)
    for G in GVEC:
        index.append(
            (G[0]%NGF[0], G[1]%NGF[1], G[2]%NGF[2])
        )
    return index
