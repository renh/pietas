#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np

class OUTCAR:
    """
    OUTCAR class for VASP OUTCAR files.
    """
    def __init__(self, fname = 'OUTCAR'):
        try:
            self.__fh = open(fname, 'r')
            self.__fh.close()
            self.__fname = fname
        except:
            raise IOError("Can not open file {}".format(fname))

    def getFermiEnergy(self):
        """
        Parse OUTCAR for the Fermi energy.
        Args:
            None
        Returns:
            efermi:  real, Fermi energy
        """
        with open(self.__fname, 'r') as fh:
            for l in fh:
                l = l.strip()
                if l.startswith('E-fermi'):
                    break
        efermi = float(l.split()[2])
        return efermi

    def getKSampling(self):
        """
        Parse OUTCAR for K-sampling, including K-vectors and corresponding weights.
        Args:
            None
        Returns:
            kvec:  ndarray (nk x 3), k-vectors
            kweight : ndarray (nk), kpoint weights
        Raises:
            None
        """
        with open(self.__fname, 'r') as fh:
            for l in fh:
                if l.startswith(' Following reciprocal coordinates:'):
                    break
            l = fh.next()
            dump = []
            while True:
                l = fh.next()
                if len(l) < 10:
                    break
                dump.append([float(x) for x in l.split()])
        dump = np.array(dump)
        kvec = dump[:,:3]
        kweight = dump[:,3]
        return kvec, kweight
                
            
        
            
