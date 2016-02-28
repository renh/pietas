#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np

def TersoffHamann(psi_fd, param):
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
    print('\nGet into Tersoff-Hamann calculation for IETS...')

    psi_0_fd = psi_fd['psi_0_fd']
    dpsi_P = psi_fd['dpsi_P']
    dpsi_I = psi_fd['dpsi_I']
    gw = psi_fd['weights_fd']

    pass
