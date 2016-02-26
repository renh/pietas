#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    NGF = np.zeros(3, dtype=np.int)
    for i in range(3):
        l = length(a[i])
        N = int(l / rgd) + 1
        NGF[i] = N
    return NGF
