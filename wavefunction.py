#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Module imports
from __future__ import print_function
import numpy as np
from constant import *
import scipy.linalg
#==============================================================================
class WAVECAR:
    """
    WAVECAR class for VASP pseudo wavefunctions
    At initiate Only read the WAVECAR headers:
        rec 1: rec length, nspin, complex precision
        rec 2: nkpts, nbands, E cutoff, and lattice consts
        rec 3 to the end would not be read until explicit reading methods invoked.
    """

    def __init__(self, fnm = "WAVECAR"):
        """
        Initialize the class with supplied WAVECAR file name.
        
        Args:
            fnm (optional): WAVECAR filename
        Returns:
            None
        Raises:
            IOError if can not open WAVECAR file.
            ValueError if inconsistent record/message read.
        """
        self.__fname = fnm

        try:
            self.__fh = open(self.__fname, 'rb')
        except:
            raise IOError("Failed open wavefunction file: {}".format(self.__fname))

        self.__recl, self.__nspin, self.__tag = self.read_tag()

        self.__prec = self.setWFPrec()

        self.__nk, self.__nb, self.__Ec, self.__a = self.read_rec2()
        self.__Omega = self.getOmega()
        self.__b = self.getRecipLatt()
        self.__ngmax = self.getNGMax()

        # will not initialize overlap matrix
        # explicitly call obj.calOverlap() for the whole matrix 
        # self.S = self.calcOverlap()

    def getRECL(self):      return self.__recl
    def getNSpin(self):     return self.__nspin
    def getTag(self):       return self.__tag
    def getPrec(self):      return self.__prec
    def getNKpts(self):     return self.__nk
    def getNBands(self):    return self.__nb
    def getENMax(self):     return self.__Ec
    def getRealLatt(self):  return self.__a
    def getNGMax(self):     return self.__ngmax
    
    
        
    def read_tag(self):
        'Read WAVECAR header for recl, nspin, and tag'
        self.__fh.seek(0)
        rrecl, rispin, rtag = np.fromfile(self.__fh,dtype=np.float, count=3)
        return int(rrecl), int(rispin), int(rtag)

    

    def setWFPrec(self):
        '''
        Set wavefunction coefficients precision:
            TAG = 45200: single precision complex, np.complex64, or complex(qs)
            TAG = 45210: double precision complex, np.complex128, or complex(q)
        '''
        if self.__tag == 45200:
            return np.complex64
        elif self.__tag == 45210:
            return np.complex128
        elif self.__tag == 53300:
            raise ValueError("VASP5 WAVECAR format, not implemented yet")
        elif self.__tag == 53310:
            raise ValueError("VASP5 WAVECAR format with double precision "
                          +"coefficients, not implemented yet")
        else:
            raise ValueError("Invalid TAG values: {}".format(self.tag))

    def read_rec2(self):
        '''Read rec #2, length 12, with real(q) or double precision:
            #1 - #3: nkpts, nbands, E cutoff
            #4 - #12: lattice vector (real space)
        '''
        self.__fh.seek(self.__recl)
        rec = np.fromfile(self.__fh, dtype=float, count=12)
        nk, nb, Ec = int(rec[0]), int(rec[1]), rec[2]
        a = np.reshape(rec[3:],[3,3])
        return nk, nb, Ec, a

    def getOmega(self):
        '''
        Calculate the supercell volume
            Omega = a0 \dot (a1 \cross a2)
        '''
        return np.dot(self.__a[0], np.cross(self.__a[1], self.__a[2]))

    def getRecipLatt(self):
        '''Calculate the reciprocal lattice vectors.
        a_i \dot b_j = 2 \pi \delta_{ij}
        '''
        b = np.zeros([3,3])
        b[0] = np.cross(self.__a[1], self.__a[2])
        b[1] = np.cross(self.__a[2], self.__a[0])
        b[2] = np.cross(self.__a[0], self.__a[1])
        b *= (2.0 * PI / self.__Omega)
        return b

    def getNGMax(self):
        import collections
        ng = np.zeros([3,3],dtype=np.int)
        ngmax = []
        lb = np.zeros(3)
        for i in range(3):
            lb[i] = np.linalg.norm(self.__b[i])
        #Ec_sqrt = np.sqrt(self.Ec*Const)
        Ec_sqrt = np.sqrt(self.__Ec / RYTOEV) / AUTOA
        ind = collections.deque([0,1,2])
        for i in range(3):
            ind.rotate(i)
            b0 = self.__b[ind[0]]; lb0 = lb[ind[0]]
            b1 = self.__b[ind[1]]; lb1 = lb[ind[1]]
            b2 = self.__b[ind[2]]; lb2 = lb[ind[2]]
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
        '''
        Locate record position in WAVECAR for spin #ispin, and kpt #ik
        Args:
            ispin: spin index
            ik   : kpt index
        Returns:
            record position
        Raises:
            raise ValueError for invalid ispin or ik values
        '''
        if ispin >= self.__nspin:
            raise ValueError("ISPIN in WAVECAR: {};\t ispin to be read: # {}".format(
                self.__nspin, ispin+1
            ))
        if ik >= self.__nk:
            raise ValueError("#kpts in WAVECAR: {}; \t ik to be read: # {}".format(
                self.__nk, ik+1
            ))
        irec = 2
        irec += ispin * self.__nk * (self.__nb + 1)
        irec += ik * (self.__nb + 1)
        #print("\nRecord for spin #{} and kpt #{} located: IREC = {}".format(
        #    ispin, ik, irec+1
        #))
        return irec

    def readKHeader(self, ispin = 0, ik = 0):
        '''
        Read the header information for a specified kpt and specified spin
        Args:
            ispin : spin index
            ik    : kpt index
        Returns:
            nplw  : number of plane-waves
            kvec  : kpt vector of this kpt
            eig   : eigen-energies of the NBands bands
            FermiW: Fermi occupation weights of the NBands bands
        Raises:
            None
        '''
        irec = self.locateREC(ispin, ik)
        self.__fh.seek(self.__recl * irec)
        dump = np.fromfile(self.__fh, dtype=float, count = 4+3*self.__nb)
        nplw = int(dump[0])
        kvec = np.array(dump[1:4])
        tmp = np.reshape(dump[4:], [-1,3])
        eig = tmp[:,0]
        FermiW = tmp[:,2]
        return nplw, kvec, eig, FermiW

    
    def readBandsCoeffK(self, ispin = 0, ik = 0):
        '''
        Read wave function info for specified spin (def 0) and k-point (def 0)
        Args: 
            ispin  : int, spin index
            ik     : int, kpt index
        Returns:
            bands  : ndarray (nb x nplw), Frouier coefficients for the nb pseudo
                     wavefunctions Psi_{ib,ik,ispin} with ib = 1 ... nb
        Raises:
            None
        '''
        nplw, kvec, eig, FermiW = self.readKHeader(ispin, ik)
        GVEC = self.prepareGVEC(ik)

        if len(GVEC) != nplw:
            raise ValueError("nplw = {}, len(GVEC) = {}".format(
                nplw, len(GVEC)
            ))

        # no matter what prec used in WAVCAR, we use double precision (complex128)
        # to operate the wavefunctions.
        bands = np.zeros([self.__nb, nplw], dtype=np.complex128)
        for ib in range(self.__nb):
            bands[ib] = self.readBandCoeff(ispin, ik, ib)

        print("Wavefunctions with spin {} and kpt {} read".format(
            ispin, ik
        ))
        return bands

    def readBandCoeff(self, ispin=0, ik=0, ib=0, normalize=True):
        '''
        Read band coefficients for index (ispin, ik, ib)
        Args:
            ispin : spin index
            ik    : kpt index
            ib    : band index
        Returns:
            coeff : ndarray, Fourier coefficients for pseudo wavefunction Psi_{ib,ik,ispin}
        Raises:
            None

        '''
        irec = self.locateREC(ispin, ik)
        self.__fh.seek(self.__recl*irec)
        dump = np.fromfile(self.__fh, dtype=float, count=1)
        nplw = int(dump)

        irec += 1
        irec += ib
        self.__fh.seek(self.__recl*irec)
        coeff = np.fromfile(self.__fh, dtype=self.__prec, count=nplw)

        # expand coefficients into double precision complex and re-normalize if required
        coeff = np.array(coeff, dtype = np.complex128)

        if normalize:
            dump = np.dot(np.conj(coeff), coeff)
            coeff /= np.sqrt(dump)
        
        return coeff



    def prepareGVEC(self, ik):
        '''
        Prepare the set of G vectors that satisfying
           \hbar^2 |kvec + Gvec|^2 <= 2 * m * Ec
        Args:
            ik : int, kpt index
        Returns:
            GVEC : ndarray, G vectors with dimension (nplw*3)
        Raises:
            None
        '''
        # constant for 2*m_e/\hbar in (eV*Ang)^{-1}
        if not isinstance(ik, int):
            print('prepareGVEC need an integer as argument (the n-th kpt, start from 0)')
        dump = self.readKHeader(0, ik)
        kvec = dump[1]
        print("Preparing GVEC for kvec:", kvec)
        const = 1.0 / RYTOEV / AUTOA2
        Ec = self.__Ec * const
        GVEC = []
        for k in range(2*self.__ngmax[2]+1):
            igz = k
            if igz > self.__ngmax[2]:
                igz = k - 2 * self.__ngmax[2] - 1
            for j in range(2*self.__ngmax[1]+1):
                igy = j
                if igy > self.__ngmax[1]:
                    igy = j - 2 * self.__ngmax[1] - 1
                for i in range(2*self.__ngmax[0]+1):
                    igx = i
                    if igx > self.__ngmax[0]:
                        igx = i - 2 * self.__ngmax[0] - 1
                    G = np.array([igx, igy, igz])
                    dump = (G + kvec).reshape(1,-1)
                    #print(dump.shape)
                    sumkg = np.dot(dump, self.__b)
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

class WaveFunction:
    """
    class WaveFunction.
    describe the wavefunction (pseudo and all-electron) for specified spin and kpt indices.
    usage: WaveFunction(WAVECAR, ispin, ik)
    """
    def __init__(self, WAVECAR, ispin = 0, ik = 0):
        """
        Initializer for class WaveFunction.
        Args:
            WAVECAR  : class WAVECAR, VASP WAVECAR reader
            ispin    : spin index
            ik       : kpt index
        Returns:
            None
        Raises:
            None
        """
        self.__WC = WAVECAR
        self.__nb = self.__WC.getNBands()
        self.__ispin = ispin
        self.__ik = ik
        self.__nplw, self.__kvec, self.__eig, self.__FermiW = self.readKHeader()
        self.__wps = self.readWPS()

        
    def getWPS(self): return self.__wps
    def getNPlw(self): return self.__nplw
    def getKVec(self): return self.__kvec
    def getEig(self): return self.__eig
    def getFermiW(self): return self.__FermiW

    def readKHeader(self):
        kheader = self.__WC.readKHeader(self.__ispin, self.__ik)
        nplw, kvec, eig, FermiW = kheader
        return nplw, np.array(kvec), np.array(eig), np.array(FermiW)
    
    def readWPS(self):
        WC = self.__WC
        nb = self.__nb
        nplw = self.__nplw
        wps = np.zeros([nb, nplw], dtype=np.complex128)
        for ib in range(nb):
            wps[ib] = WC.readBandCoeff(self.__ispin, self.__ik, ib)
        return wps

    def getOverlap(self):
        nb = self.__nb
        wps = self.__wps
        S = np.dot(
            np.conj(wps), wps.T
        )
        return S

    def getWMatrix(self, tol):
        """
        Construct the occupation number diagonal matrix.
            Small occupation numbers will be replaced by a small cutoff value tol, to bypass
            possible numerical problems due to linear dependence.
            see Szabo, "Modern Quantum Chemistry", Sec. 3.4.5 for details.
        """
        W = np.array([max(tol, x) for x in self.__FermiW])
        return np.diag(W)

    def getOrthMatrix(self, OccWeighted, tol):
        """
        Construct the unitary transform matrix to orthogonalize the pseudo wavefunction.
            use Lowding's symmetric or occupation weighted orthogonalization scheme.
            Lowding's:
                   C = S^{-1/2}
            Occupation weighted:
                   C = W(WSW)^{-1/2}
        Args:
            OccWeighted   : logical (optional), occupation weighted?
            tol           : float (optional), small value for occupation cutoff
        Returns:
            C    : ndarray (nb x nb), the unitary transform matrix.
        """
        S = self.getOverlap()
        if OccWeighted:
            W = self.getWMatrix(tol)
            wsw = np.dot(W, np.dot(S, W)) # (WSW)
            wsw = scipy.linalg.inv(wsw)   # (WSW)^{-1}
            wsw = scipy.linalg.sqrtm(wsw) # (WSW)^{-1/2}
            C = np.dot(W,wsw)
        else:
            C = scipy.linalg.inv(S)       # (S)^{-1}
            C = scipy.linalg.sqrtm(s)     # (S)^{-1/2}

        return C


    def getWAE(self, OccWeighted = True, tol = 1.0E-3, check = True):
        """
        Orthgonalize the PS wavefunction, get the AE wavefunction.
        Args:
            OccWeighted : logical
            tol         : float
            check       : logical, check < WAE | WAE > == I
        Returns:
            w           : ndarray (nb x nplw), all-electron wavefunction
        """
        C = self.getOrthMatrix(OccWeighted = OccWeighted, tol = tol)
        w = self.getWPS()
        w = np.dot(w.T, C)
        w = w.T
        #print('checking orthgonality...')

        I = np.eye(len(w), dtype=np.complex128)
        WW = np.dot(
            np.conj(w), w.T
        )
        orth_success = np.allclose(WW, I)
        if orth_success:
            #print("Wavefunction orthogonalized.")
            return w
        else:
            raise ValueError('Orthognalization failed.')
        

    
            
            
        
        
        
        
     
        



if __name__ == '__main__':
    Psi = WAVECAR('mode0/WAVECAR')
    print("\nHeader info: recl {}, nspin {}, and tag {}".format(
        Psi.getRECL(), Psi.getNSpin(), Psi.getTag()))
    print("Prec: {}".format(Psi.getPrec))
    print("nk: {}\tnbands: {}\tEc: {}".format(
        Psi.getNKpts(), Psi.getNBands(), Psi.getENMax()
    ))
    print("\nLattice vectors:")
    a = Psi.getRealLatt()
    for i in range(3):
        print("{} {} {}".format(a[i,0], a[i,1], a[i,2]))

    print("\nSupercell volume: {}".format(Psi.getOmega()))
    print("\nLattice vectors:")
    b = Psi.getRecipLatt()
    for i in range(3):
        print("{:12.8f} {:12.8f} {:12.8f}".format(
            b[i,0], b[i,1], b[i,2]
        ))
    ngmax = Psi.getNGMax()
    print("max No. of G vectors: {} {} {}".format(
        ngmax[0], ngmax[1], ngmax[2]
    ))

    W = WaveFunction(Psi,0,0)
    S = W.getOverlap()
    print(S.shape)
    wae = W.getWAE()
    print(wae.shape)
            

    print("\n\n")
