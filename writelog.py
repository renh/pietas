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
    gw = bands_contrib.get('weights')

    E_0 = psi_0.getEig()
    E_b = psi_b.getEig()
    E_f = psi_f.getEig()
    
    print("\n {} Bands taking into account for this k-point:".format(
        band_final - band_init + 1
        ))
    print("  all contributions evaluated respect to the equilibrium Fermi level")
    print("\n  nb(from 1)    E0          Eb          Ef        weight")
    for ibm in range(band_init, band_final+1):
        print("{:8d}{:12.4f}{:12.4f}{:12.4f}{:14.5E}".format(
            ibm+1, E_0[ibm], E_b[ibm], E_f[ibm], gw[ibm-band_init]
        ))

    return
        
