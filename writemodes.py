#!/usr/bin/env python
# -*- coding utf-8 -*-

from __future__ import print_function
from constant import *

def write_molden(fname, NormalModes):
    """
    Write the normal modes into molden format for visulization.
    Args:
        fname        :  file name to be saved
        NormalModes  :  dictionary, contains the system information
    Returns:
        None
    Raises:
        IOError
        ValueError if inappropriate dataset detected in the NormalModes dictionary
    """
    try:
        fh = open(fname, 'w')
    except:
        raise IOError("Failed to open file {}".format(fname))

    #
    # write molden header
    #
    fh.write('[Molden Format]\n')

    # write FREQ section
    fh.write('[FREQ]\n')

    dof = NormalModes.get('DOF')
    if not isinstance(dof, int):
        raise ValueError("Incorrect DOF value in NormalModes")

    freq = NormalModes.get('frequency')
    if len(freq) != dof:
        raise ValueError('Incorrect length for the freq array')

    for i in range(dof):
        fh.write('{:10.4f}\n'.format(freq[i]))

    #
    # write the equilibrium coordinates
    #
    fh.write('[FR-COORD]\n')
    IonList = NormalModes.get('IonList')
    X0 = NormalModes.get('X0')
    if len(IonList) != len(X0):
        raise ValueError("Inconsistent length for Ions and Coordinates")
    X0 /= AUTOA # Ang to Bohr
    NIons = len(IonList)
    for iion in range(NIons):
        fh.write("{:<3s}{:14.5f}{:14.5f}{:14.5f}\n".format(
            IonList[iion],
            X0[iion,0], X0[iion, 1], X0[iion, 2]
        ))

    #
    # write normal modes
    #
    fh.write('[FR-NORM-COORD]\n')
    modes = NormalModes.get('modes')
    if len(modes) != dof:
        raise ValueError('Inconsistent length for normal modes with DOF')

    for imode in range(dof):
        fh.write('vibration {}'.format(imode+1))
        modes[imode] /= AUTOA # Ang to Bohr
        for iion in range(NIons):
            fh.write('{:14.5f}{:14.5f}{:14.5f}\n'.format(
                *(modes[imode][iion])
            ))

    fh.close()
    return
    
    
