#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Module imports
from __future__ import print_function
import numpy as np
from scipy.io import FortranFile as FF
from constant import *
import wavefunction as wf
#==============================================================================
class NormalCAR:
    """
    NormalCAR class for VASP PAW augmented charged and projector coefficients.

        the NormalCAR file can be written by VASP by seting the STM entry in
        INCAR (see stm.F for details), or by directly hack main.F to call 
            WRT_IETS()
        after SCF convergence.

    NormalCAR contents:
        rec 1:  LMDIM, NIONS, NRSPINNORS
        rec 2:  CQIJ[LMDIM, LMDIM, NIONS, NRSPINORS]
        rec 3:  NRPOD, NPRO, NTYP
        rec 4...3+NTYP:
                LMMAX, NITYP
        loop over spin:
            loop over kpts:
                loop over bands:
                    CPROJ[NPRO_TOT]
        end    
    """

    def __init__(self, normalcar='NormalCAR', wavecar='WAVECAR'):
        """
        Initialize the class with supplied file name (including path)
        Only the records untile projector coefficients are read for initialize
        projector coefficients will only be read when explicitly invoked.

        Args:
            fnm (optional): NormalCAR filename (including path)
        Returns:
            None
        Raise:
            IOError if encounter problems reading NormalCAR
            ValueError if inconsistent record/info read

        Note:
            All 'private' members or attributes (starts with single underline) 
            have names following the nice VASP style... with each has a
            corresponding a callable getXXX() method, where XXX might be easily
            understood...
        """

        self._normalcar = normalcar
        self._wavecar = wavecar
        try:
            self._fh = FF(self._normalcar, 'r')
        except:
            raise IOError("Failed to open NormalCAR file: {}".format(self._fnm))

        # rec 1 
        self._LMDIM, self._NIONS, self._NRSPINORS = FF.read_ints(self._fh)
        
        # rec 2
        self._CQIJ = FF.read_reals(self._fh)
        dump = self._LMDIM * self._LMDIM *  self._NIONS * self._NRSPINORS
        if len(self._CQIJ) != dump:
            raise ValueError("Inconsistent dimension for CQIJ:\n"
                    "\t# floats intend to read: {}\n\t# floats in this record".format(
                        dump, len(self._CQIJ))
                    )
        self._CQIJ = self._CQIJ.reshape(
                [self._LMDIM, self._LMDIM, self._NIONS, self._NRSPINORS],
                order='F'
            )
        del dump

        # rec 3
        self._NPROD, self._NPRO, self._NTYP = FF.read_ints(self._fh)

        # rec 4 ... 3 + NTypes
        self._LMMAX = []
        for i in range(self._NTYP):
            self._LMMAX.append(FF.read_ints(self._fh))
        self._LMMAX = np.array(self._LMMAX)
        self._NPROJ = np.sum(self._LMMAX[:,0] * self._LMMAX[:,1])
        
        # read wavefunction stuff, number of kpts and bands
        _WC = wf.WAVECAR(self._wavecar)
        self._NK = _WC.getNKpts()
        self._NB = _WC.getNBands()


    def getLMDim(self):
        """max number of angular momentums (l,m)"""
        return self._LMDIM

    def getNIons(self):
        "Number of Ions"
        return self._NIONS

    def getNSpin(self):
        "Number of spinors"
        return self._NRSPINORS

    def getCQij(self):
        "Augmented charge matrix"
        return self._CQIJ

    def getNProD(self):
        "first dimension of projected wave array"
        return self._NPROD

    def getNPro(self):
        "local number of elements of projected wave array"
        return self._NPRO

    def getNTypes(self):
        "Number of ion types"
        return self._NTYP

    def getLMMax(self):
        "max angular channels for each ion type"
        return self._LMMAX

    def getNProj(self):
        "total number of projectors"
        return self._NPROJ

    def getProjCoeffs(self):
        _CPROJ = np.zeros([self._NPROJ, self._NB, self._NK, self._NRSPINORS],
                dtype=np.complex128)
        for ispin in range(self._NRSPINORS):
            for ik in range(self._NK):
                for ib in range(self._NB):
                    dump = FF.read_reals(self._fh).reshape(-1,2)
                    if len(dump) != self._NPROJ:
                        raise ValueError(
                                "Insurfficient dimension for projector coeffs "
                                "for Spin: {}, Kpt: {}, Band: {}\n"
                                "\t # elements read {}, # elements required {}".format(
                                    ispin+1, ik+1, ib+1, self._NPROJ, len(dump)
                                ))
                    _CPROJ[:,ib,ik,ispin] = dump[:,0] + dump[:,1] * 1.0j
        return _CPROJ







if __name__ == '__main__':
    norm = NormalCAR(normalcar = '../mode0/NormalCAR', 
            wavecar = '../mode0/WAVECAR')
    print(norm.getLMDim())
    print(norm.getNIons())
    print(norm.getNSpin())
    print(norm.getCQij().shape)
    print('NRPOD = ', norm.getNProD())
    print('NRO = ', norm.getNPro())
    print('number of ion types = ', norm.getNTypes())
    print('\nLMMAX : \n', norm.getLMMax())
    print('number of projectors = ', norm.getNProj())
    print('\nreading projector coeffs...')
    dump = norm.getProjCoeffs()
    print('Done!')

