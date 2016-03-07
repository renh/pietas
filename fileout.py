#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np

def save_thdata(th_results, ispin, ik, param):
    """
    Save Tersorff-Hamann results for this (ispin, ik) to data formats specified in param.
    
    Args:
        th_results : dictionary, Tersorff-Hamann results (rho_0_fd, drho_P, drho_I)
        ispin      : int, spin index
        ik         : int, kpt index
        param      : dictionary, input configuration
    Returns:
        None
    Raises:
        IOError    : if IO errors encountered
    """
    q_out = param.get('output quantities')
    fmt_out = param.get('output formats')
    path = param.get('output path')

    spink = ".spin-{:1d}.k-{:02d}".format(ispin,ik)
    basename = path + "/{}" + spink

    # write numpy npy files
    if fmt_out.get('NPY'):
        for dname in ['rho_0_fd', 'drho_P', 'drho_I', 'rho_0_orig']:
            filename = basename.format(dname) + ".npy"
            np.save(filename, th_results.get(dname))
        


    # write XCrysden xsf files
    if fmt_out.get('XSF'):
        pass

    # write NetCDF files
    if fmt_out.get('NetCDF'):
        pass
