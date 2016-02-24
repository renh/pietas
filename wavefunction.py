#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Module imports
from __future__ import print_function
import numpy as np
from constant import *
#==============================================================================
class WaveFunction:
    """Wavefunction class for VASP WAVECAR.
    Only read the WAVECAR headers:
        rec 1: rec length, nspin, complex precision
        rec 2: nkpts, nbands, E cutoff, and lattice consts
        rec 3 to the end would not read until explicit readWF is invoked.
    """

    def __init__(self, fnm = "WAVECAR"):
        self.fname = fnm

        try:
            self.fh = open(self.fname, 'rb')
        except:
            raise IOError("Failed open wavefunction file: {}".format(self.fname))

        self.recl, self.nspin, self.tag = self.read_tag()
        self.prec = self.setWFPrec()
        self.nk, self.nb, self.Ec, self.a = self.read_rec2()
        self.Omega = self.calOmega()
        self.b = self.calRecLatt()
        self.ngmax = self.calNG()

        # will not initialize overlap matrix
        # explicitly call obj.calOverlap() for the whole matrix 
        # self.S = self.calcOverlap()

    def read_tag(self):
        'Read WAVECAR header for recl, nspin, and tag'
        self.fh.seek(0)
        rrecl, rispin, rtag = np.fromfile(self.fh,dtype=np.float, count=3)
        return int(rrecl), int(rispin), int(rtag)

    def setWFPrec(self):
        '''Set wavefunction coefficients precision:
            TAG = 45200: single precision complex, np.complex64, or complex(qs)
            TAG = 45210: double precision complex, np.complex128, or complex(q)
        '''
        if self.tag == 45200:
            return np.complex64
        elif self.tag == 45210:
            return np.complex128
        elif self.tag == 53300:
            raise ValueError("VASP5 WAVECAR format, not implemented yet")
        elif self.tag == 53310:
            raise ValueError("VASP5 WAVECAR format with double precision "
                          +"coefficients, not implemented yet")
        else:
            raise ValueError("Invalid TAG values: {}".format(self.tag))

    def read_rec2(self):
        '''Read rec #2, length 12, with real(q) or double precision:
            #1 - #3: nkpts, nbands, E cutoff
            #4 - #12: lattice vector (real space)
        '''
        self.fh.seek(self.recl)
        rec = np.fromfile(self.fh, dtype=float, count=12)
        nk, nb, Ec = int(rec[0]), int(rec[1]), rec[2]
        a = np.reshape(rec[3:],[3,3])
        return nk, nb, Ec, a

    def calOmega(self):
        '''Calculate the supercell volume
        Omega = a0 \dot (a1 \cross a2)
        '''
        return np.dot(self.a[0], np.cross(self.a[1], self.a[2]))

    def calRecLatt(self):
        '''Calculate the reciprocal lattice vectors.
        a_i \dot b_j = 2 \pi \delta_{ij}
        '''
        b = np.zeros([3,3])
        b[0] = np.cross(self.a[1], self.a[2])
        b[1] = np.cross(self.a[2], self.a[0])
        b[2] = np.cross(self.a[0], self.a[1])
        b *= (2.0 * np.pi / self.Omega)
        return b

    def calNG(self):
        import collections
        ng = np.zeros([3,3],dtype=np.int)
        ngmax = []
        lb = np.zeros(3)
        for i in range(3):
            lb[i] = np.linalg.norm(self.b[i])
        #Ec_sqrt = np.sqrt(self.Ec*Const)
        Ec_sqrt = np.sqrt(self.Ec / RYTOEV) / AUTOA
        ind = collections.deque([0,1,2])
        for i in range(3):
            ind.rotate(i)
            b0 = self.b[ind[0]]; lb0 = lb[ind[0]]
            b1 = self.b[ind[1]]; lb1 = lb[ind[1]]
            b2 = self.b[ind[2]]; lb2 = lb[ind[2]]
            phi = abs(np.arccos(np.dot(b0,b1) / (lb0 * lb1)))
            sin_phi = abs(np.sin(phi))
            v = np.cross(b0, b1)
            lv = np.linalg.norm(v)
            sin_gamma = abs(np.dot(b2, v) / (lv * lb2))
            ind.rotate(-i)

            sines = collections.deque([sin_phi, sin_phi, sin_gamma])
            sines.rotate(-i)

            for j in range(3):
                ng[i,j] = int(Ec_sqrt / (lb[j] * sines[j]) + 1)
        return np.max(ng,axis=0)

    def locateREC(self, ispin = 0, ik = 0):
        '''Locate record position in WAVECAR for spin #ispin, and kpt #ik
           raise ValueError for invalid ispin or ik values
           return value: record number (irec)
        '''
        if ispin >= self.nspin:
            raise ValueError("ISPIN in WAVECAR: {};\t ispin to be read: # {}".format(
                self.nspin, ispin+1
            ))
        if ik >= self.nk:
            raise ValueError("#kpts in WAVECAR: {}; \t ik to be read: # {}".format(
                self.nk, ik+1
            ))
        irec = 2
        irec += ispin * self.nk * (self.nb + 1)
        irec += ik * (self.nb + 1)
        #print("\nRecord for spin #{} and kpt #{} located: IREC = {}".format(
        #    ispin, ik, irec+1
        #))
        return irec

    def readKHeader(self, ispin = 0, ik = 0):
        '''Read the header information for a specified kpt with specified spin
           return NPLW, kvec, eigenvalues and Fermi occupation numbers
        '''
        irec = self.locateREC(ispin, ik)
        self.fh.seek(self.recl * irec)
        dump = np.fromfile(self.fh, dtype=float, count = 4+3*self.nb)
        nplw = int(dump[0])
        kvec = np.array(dump[1:4])
        tmp = np.reshape(dump[4:], [-1,3])
        eig = tmp[:,0]
        FermiW = tmp[:,2]
        return nplw, kvec, eig, FermiW

    
    def readWF(self, ispin = 0, ik = 0):
        '''Read wave function info for specified spin (def 0) and k-point (def 0)
           dumped info including: 
               k vector    
               G vectors
               # of planewaves (NPLW)
               Eigenvalues (in complex)
               Fermi occupation number
               Fourier coefficients
        '''
        nplw, kvec, eig, FermiW = self.readKHeader(ispin, ik)
        GVEC = self.prepareGVEC(kvec)

        if len(GVEC) != nplw:
            raise ValueError("nplw = {}, len(GVEC) = {}".format(
                nplw, len(GVEC)
            ))

        # no matter what prec used in WAVCAR, we use double precision (complex128)
        # to operate the wavefunctions.
        bands = np.zeros([self.nb, nplw], dtype=np.complex128)
        for ib in range(self.nb):
            bands[ib] = self.readBandCoeff(ispin, ik, ib)

        print("Wavefunctions with spin {} and kpt {} read".format(
            ispin, ik
        ))
        return GVEC, bands

    def readBandCoeff(self, ispin=0, ik=0, ib=0):
        '''Read band coefficients for index (ispin, ik, ib)
           return a (nplw x 1) complex ndarray (complex64, single precision)
        '''
        irec = self.locateREC(ispin, ik)
        self.fh.seek(self.recl*irec)
        dump = np.fromfile(self.fh, dtype=float, count=1)
        nplw = int(dump)

        irec += 1
        irec += ib
        self.fh.seek(self.recl*irec)
        coeff = np.fromfile(self.fh, dtype=self.prec, count=nplw)

        # expand coefficients into double precision complex and re-normalize if required
        coeff = np.array(coeff, dtype = np.complex128)

        #if normalize:
        #    dump = np.dot(np.conj(coeff), coeff)
        #    coeff /= np.sqrt(dump)
        
        return coeff



    def prepareGVEC(self, ik):
        '''Prepare the set of G vectors that satisfying
           \hbar^2 |kvec + Gvec|^2 <= 2 * m * Ec
           Note that each k point corresponds to a unique set of G vectors
        '''
        # constant for 2*m_e/\hbar in (eV*Ang)^{-1}
        if not isinstance(ik, int):
            print('prepareGVEC need an integer as argument (the n-th kpt, start from 0)')
        dump = self.readKHeader(0, ik)
        kvec = dump[1]
        print("Preparing GVEC for kvec:", kvec)
        const = 1.0 / RYTOEV / AUTOA2
        Ec = self.Ec * const
        GVEC = []
        for k in range(2*self.ngmax[2]+1):
            igz = k
            if igz > self.ngmax[2]:
                igz = k - 2 * self.ngmax[2] - 1
            for j in range(2*self.ngmax[1]+1):
                igy = j
                if igy > self.ngmax[1]:
                    igy = j - 2 * self.ngmax[1] - 1
                for i in range(2*self.ngmax[0]+1):
                    igx = i
                    if igx > self.ngmax[0]:
                        igx = i - 2 * self.ngmax[0] - 1
                    G = np.array([igx, igy, igz])
                    dump = (G + kvec).reshape(1,-1)
                    #print(dump.shape)
                    sumkg = np.dot(dump, self.b)
                    #print(sumkg.shape)
                    #print(sumkg)
                    #raise SystemExit
                    etot = np.sum(sumkg * sumkg)
                    if etot < Ec:
                        GVEC.append([igx,igy,igz])
        
                        
                    
        return np.array(GVEC)

    #def calcOverlap(self):
    #    S = np.zeros([self.nb, self.nb, self.nk, self.nspin], dtype=np.complex128)
    #    for ispin in range(self.nspin):
    #        for ik in range(self.nk):
    #            S[:,:,ik,ispin] = self.calcOverlapKSp(ispin, ik)
    #    return S

    #def calcOverlapKSp(self, ispin = 0, ik = 0):
    #    SKSp = np.zeros([self.nb, self.nb], dtype=np.complex128)
    #    for ibm in range(self.nb):
    #        psi_m = self.readBandCoeff(ispin, ik, ibm)
    #        for ibn in range(ibm, self.nb):
    #            psi_n = self.readBandCoeff(ispin, ik, ibn)
    #            SSKp[ibm,ibn] = np.dot(np.conj(psi_m), psi_n)
    #            if ibm != ibn:
    #                SKSp[ibn,ibm] = np.conj(SKSp[ibm,ibn])
    #    return SKSp
        
        



if __name__ == '__main__':
    Psi = WaveFunction('mode0/WAVECAR')
    print("\nHeader info: recl {}, nspin {}, and tag {}".format(
        Psi.recl, Psi.nspin, Psi.tag))
    print("Prec: {}".format(Psi.prec))
    print("nk: {}\tnbands: {}\tEc: {}".format(
        Psi.nk, Psi.nb, Psi.Ec
    ))
    print("\nLattice vectors:")
    for i in range(3):
        print("{} {} {}".format(Psi.a[i,0], Psi.a[i,1], Psi.a[i,2]))

    print("\nSupercell volume: {}".format(Psi.Omega))
    print("\nLattice vectors:")
    for i in range(3):
        print("{:12.8f} {:12.8f} {:12.8f}".format(
            Psi.b[i,0], Psi.b[i,1], Psi.b[i,2]
        ))
    print("max No. of G vectors: {} {} {}".format(
        Psi.ngmax[0], Psi.ngmax[1], Psi.ngmax[2]
    ))


    #GVEC = Psi.prepareGVEC(0)
    #np.save("GVEC.k{:02d}.npy".format(0), GVEC)
    #np.save("bands.s{:1d}.k{:02d}.npy".format(0,0), bands)

    # check nomalization
    #for i in range(Psi.nb):
    #    dump = np.dot(np.conj(bands[i]), bands[i])
    #    print("{:6d}{:12.5f}".format(i,dump))

#    dd = np.zeros([Psi.nb, Psi.nb], dtype=np.complex128)
#    for i in range(Psi.nb):
#        for j in range(i,Psi.nb):
#            dd[i,j] = np.dot(np.conj(bands[i]), bands[j])
#    print(dd)


    #
    # check orthognormality of the pseudo wavefunctions
    #
    # first read all pseudo wavefunctions into memory
    psi = []
    for ibm in range(Psi.nb):
        psi.append(Psi.readBandCoeff(0,0,ibm))
    psi = np.array(psi)

    S = np.zeros([Psi.nb, Psi.nb], dtype=np.complex128)
    print(S.shape)
    for ibm in range(Psi.nb):
        for ibn in range(ibm, Psi.nb):
            S[ibm, ibn] = np.dot(np.conj(psi[ibm]), psi[ibn])
            
    for ibm in range(Psi.nb):
        for ibn in range(ibm):
            S[ibm,ibn] = np.conj(S[ibn,ibm])

    print('Overlap between pseudo wavefunctions calculated.')
    np.save('S.npy', S)

    
    
            


    print("\n\n")
