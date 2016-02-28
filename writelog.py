#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

def write_bands_contrib(psi_0, psi_b, psi_f, bands_contrib):
    """
    Write out band information for calculation at this (ispin, ik)
    Args:
        psi_0, psi_b, psi_f: WaveFunction objects.
        bands_contrib: dictionary, see helper for details
    Returns:
        None
    Raises:
        None
    """
    bands_range = bands_contrib.get('bands_range')
    band_init, band_final = bands_range
    nbands_calc = band_final - band_init + 1
    gw = bands_contrib.get('weights')

    E_0 = psi_0.getEig()
    E_b = psi_b.getEig()
    E_f = psi_f.getEig()
    
    print("Bands taking into account for this k-point:")
    print("  nb        E0          Eb          Ef        weight")
    for ibm in range(nbands_calc):
        m = band_init + ibm
        print("{:4d}{:12.4f}{:12.4f}{:12.4f}{:14.5E}".format(
            m+1, E_0[m], E_b[m], E_f[m], gw[ibm]
        ))

    return
        
