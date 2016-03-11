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
        self.__IonsPerType = self.getIonsPerType()
        self.__NIons = self.getNIons()
        self.__NTypes = self.getNTypes()
        self.__IonMasses = self.getIonMasses()
        self.__IBRION = int(self.getParameter('IBRION'))

    def getFermiEnergy(self):
        """
        Parse OUTCAR for the Fermi energy.
        Args:
            None
        Returns:
            efermi:  real, Fermi energy
        """
        with open(self.__fname, 'r') as fh:
            while True:
                l = fh.readline().strip()
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
            while True:
                l = fh.readline()
                if l.startswith(' Following reciprocal coordinates:'):
                    break
            l = fh.readline()
            dump = []
            while True:
                l = fh.readline()
                if len(l) < 10:
                    break
                dump.append([float(x) for x in l.split()])
        dump = np.array(dump)
        kvec = dump[:,:3]
        kweight = dump[:,3]
        return kvec, kweight

    def getParameter(self, param):
        """
        Parse OUTCAR, get paramter value.
        Args:
            param  : string, parameter name.
        Returns:
            value  : string, parameter value in string.
        Raises:
            None.
        """
        pattern = "   {}".format(param)
        with open(self.__fname, 'r') as fh:
            while True:
                l = fh.readline()
                if l.startswith(pattern):
                    break
            value = l.split()[2]
        return value

    def getDOF(self):
        """
        Parse OUTCAR, get degrees of freedom.
        Args:
            None
        Returns:
            dof : int, degrees of freedom
        Raises:
            None
        """
        with open(self.__fname, 'r') as fh:
            while True:
                l = fh.readline()
                if l.startswith('   Degrees of freedom'):
                    break
            dof = int(l.split()[-1])
        return dof

    def getX0(self):
        """
        Parse OUTCAR, get the equilibrium coordinates.
        Args:
            None
        Returns:
            coord: ndarray (Nions x 3), xyz coordinates 
        Raises:
            None
        """
        coord = []
        with open(self.__fname, 'r') as fh:
            while True:
                l = fh.readline()
                if l.startswith(' position of ions in cartesian'):
                    break
            for i in range(self.__NIons):
                l = fh.readline()
                coord.append([float(x) for x in l.split()])
        return np.array(coord)

    def getIonsPerType(self):
        with open(self.__fname, 'r') as fh:
            while True:
                l = fh.readline()
                if l.startswith('   ions per type'):
                    break
            IonsPerType = [int(x) for x in l.split()[4:]]
        return IonsPerType

    def getNIons(self):
        return sum(self.__IonsPerType)

    def getNTypes(self):
        return len(self.__IonsPerType)
    
    def getIonTypes(self):
        with open(self.__fname, 'r') as fh:
            types = []
            while True:
                l = fh.readline()
                if l.startswith(' POTCAR'):
                    types.append(l.split()[2])
                    if len(types) == self.__NTypes:
                        break
        return types

    def getLattice(self):
        with open(self.__fname, 'r') as fh:
            while True:
                l = fh.readline()
                if l.startswith('      direct lattice vectors'):
                    break
            a = []
            for i in range(3):
                l = fh.readline()
                a.append([float(x) for x in l.split()[:3]])
        return np.array(a)

    def getNormalMode(self, imode):
        """
        Parse OUTCAR with IBRION = 5, get normal mode #imode, in cartesian coordinates.
        Args:
            imode  : the number of mode to be parsed, starts from 1
        Returns:
            Mode   : dictionary, with
                mode  : ndarray (NIons x 3), cartesian displacements for each ion along mode #imode
                Omega : Angular frequency in THz, (2 x Pi x \nu)
        Raises:
            ValueError if IBRION != 5
        """
        IBRION = self.__IBRION
        if IBRION != 5:
            raise ValueError("I need a VASP calculation with IBRION=5! exit..")
        
        pattern = '{:4d} f  ='.format(imode)
        with open(self.__fname, 'r') as fh:
            while True:
                l = fh.readline()
                if l.startswith(pattern):
                    break
            dump = l.split()
            Omega = dump[3:]  # 4x2 datasets, \nu(THz), \omega(THz), wavenumber(cm-1), and energy(meV)
            for i in range(4):
                Omega[2*i] = float(Omega[2*i])

            l = fh.readline()
            mode = []
            for ii in range(self.__NIons):
                l = fh.readline()
                mode.append([float(x) for x in l.split()[3:]])

        mode = np.array(mode)
        # now mode contains the imode-th eigenvector of the Hessian matrix
        # need to divide sqrt(mass) for each ion to get cartesian displacements
        sqrt_m = np.sqrt(self.__IonMasses)
        for ia in range(self.__NIons):
            mode[ia] /= sqrt_m[ia]


        Mode = {}
        Mode['Omega'] = Omega
        Mode['mode'] = mode
        return Mode

    def getIonMasses(self):
        """
        Parse OUTCAR, get ion masses.
        Args:
            None
        Returns:
            masses  : ndarray (NIons x 1), ion masses
        Raises:
            None
        """
        M = []
        with open(self.__fname, 'r') as fh:
            while True:
                l = fh.readline()
                if l.startswith("   POMASS"):
                    this_m = float(l.split()[2][:-1])
                    M.append(this_m)
                    if len(M) == self.__NTypes:
                        break
        masses = []
        for it in range(self.__NTypes):
            masses.extend([M[it]]*self.__IonsPerType[it])
        return np.array(masses)


if __name__ == '__main__':
    vo = OUTCAR('freq/OUTCAR')
    a = vo.getLattice()
    print(a)
    coord = vo.getX0()
    #print(coord)

    m = vo.getNormalMode(3)
    omega = m.get('Omega')
    mode = m.get('mode')
    print(omega)
    #print(mode)

    IonsPerType = vo.getIonsPerType()
    IonTypes = vo.getIonTypes()
    print(zip(IonTypes, IonsPerType))
    
