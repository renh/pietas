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
        self._fname = fnm

        try:
            self._fh = open(self._fname, 'rb')
        except:
            raise IOError("Failed open wavefunction file: {}".format(self._fname))

        self._recl, self._nspin, self._tag = self.read_tag()

        self._prec = self.setWFPrec()

        self._nk, self._nb, self._Ec, self._a = self.read_rec2()
        self._Omega = self.getOmega()
        self._b = self.getRecipLatt()
        self._ngmax = self.getNGMax()

        # will not initialize overlap matrix
        # explicitly call obj.calOverlap() for the whole matrix 
        # self.S = self.calcOverlap()

    def getRECL(self):      return self._recl
    def getNSpin(self):     return self._nspin
    def getTag(self):       return self._tag
    def getPrec(self):      return self._prec
    def getNKpts(self):     return self._nk
    def getNBands(self):    return self._nb
    def getENMax(self):     return self._Ec
    def getRealLatt(self):  return self._a
    def getNGMax(self):     return self._ngmax
    
    
        
    def read_tag(self):
        'Read WAVECAR header for recl, nspin, and tag'
        self._fh.seek(0)
        rrecl, rispin, rtag = np.fromfile(self._fh,dtype=np.float, count=3)
        return int(rrecl), int(rispin), int(rtag)

    

    def setWFPrec(self):
        '''
        Set wavefunction coefficients precision:
            TAG = 45200: single precision complex, np.complex64, or complex(qs)
            TAG = 45210: double precision complex, np.complex128, or complex(q)
        '''
        if self._tag == 45200:
            return np.complex64
        elif self._tag == 45210:
            return np.complex128
        elif self._tag == 53300:
            raise ValueError("VASP5 WAVECAR format, not implemented yet")
        elif self._tag == 53310:
            raise ValueError("VASP5 WAVECAR format with double precision "
                          +"coefficients, not implemented yet")
        else:
            raise ValueError("Invalid TAG values: {}".format(self._tag))

    def read_rec2(self):
        '''Read rec #2, length 12, with real(q) or double precision:
            #1 - #3: nkpts, nbands, E cutoff
            #4 - #12: lattice vector (real space)
        '''
        self._fh.seek(self._recl)
        rec = np.fromfile(self._fh, dtype=float, count=12)
        nk, nb, Ec = int(rec[0]), int(rec[1]), rec[2]
        a = np.reshape(rec[3:],[3,3])
        return nk, nb, Ec, a

    def getOmega(self):
        '''
        Calculate the supercell volume
            Omega = a0 \dot (a1 \cross a2)
        '''
        return np.dot(self._a[0], np.cross(self._a[1], self._a[2]))

    def getRecipLatt(self):
        '''Calculate the reciprocal lattice vectors.
        a_i \dot b_j = 2 \pi \delta_{ij}
        '''
        b = np.zeros([3,3])
        b[0] = np.cross(self._a[1], self._a[2])
        b[1] = np.cross(self._a[2], self._a[0])
        b[2] = np.cross(self._a[0], self._a[1])
        b *= (2.0 * PI / self._Omega)
        return b

    def getNGMax(self):
        import collections
        ng = np.zeros([3,3],dtype=np.int)
        ngmax = []
        lb = np.zeros(3)
        for i in range(3):
            lb[i] = np.linalg.norm(self._b[i])
        #Ec_sqrt = np.sqrt(self.Ec*Const)
        Ec_sqrt = np.sqrt(self._Ec / RYTOEV) / AUTOA
        ind = collections.deque([0,1,2])
        for i in range(3):
            ind.rotate(i)
            b0 = self._b[ind[0]]; lb0 = lb[ind[0]]
            b1 = self._b[ind[1]]; lb1 = lb[ind[1]]
            b2 = self._b[ind[2]]; lb2 = lb[ind[2]]
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
        if ispin >= self._nspin:
            raise ValueError("ISPIN in WAVECAR: {};\t ispin to be read: # {}".format(
                self._nspin, ispin+1
            ))
        if ik >= self._nk:
            raise ValueError("#kpts in WAVECAR: {}; \t ik to be read: # {}".format(
                self._nk, ik+1
            ))
        irec = 2
        irec += ispin * self._nk * (self._nb + 1)
        irec += ik * (self._nb + 1)
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
        self._fh.seek(self._recl * irec)
        dump = np.fromfile(self._fh, dtype=float, count = 4+3*self._nb)
        nplw = int(dump[0])
        kvec = np.array(dump[1:4])
        tmp = np.reshape(dump[4:], [-1,3])
        eig = tmp[:,0]
        FermiW = tmp[:,2]
        return nplw, kvec, eig, FermiW

    
    def readBandsCoeffK(self, ispin = 0, ik = 0):
        '''
        Read wave function info for specified spin and k-point 
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
        bands = np.zeros([nb, nplw], dtype=np.complex128)
        for ib in range(nb):
            bands[ib] = self.readBandCoeff(ispin, ik, ib)

        print("Wavefunctions with spin {} and kpt {} read".format(
            ispin, ik
        ))
        return bands

    def readBandCoeff(self, ispin=0, ik=0, ib=0, normalize=False):
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
        self._fh.seek(self._recl*irec)
        dump = np.fromfile(self._fh, dtype=float, count=1)
        nplw = int(dump)

        irec += 1
        irec += ib
        self._fh.seek(self._recl*irec)
        coeff = np.fromfile(self._fh, dtype=self._prec, count=nplw)

        # expand coefficients into double precision complex and re-normalize if required
        coeff = np.array(coeff, dtype = np.complex128)

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
        Ec = self._Ec * const
        GVEC = []
        for k in range(2*self._ngmax[2]+1):
            igz = k
            if igz > self._ngmax[2]:
                igz = k - 2 * self._ngmax[2] - 1
            for j in range(2*self._ngmax[1]+1):
                igy = j
                if igy > self._ngmax[1]:
                    igy = j - 2 * self._ngmax[1] - 1
                for i in range(2*self._ngmax[0]+1):
                    igx = i
                    if igx > self._ngmax[0]:
                        igx = i - 2 * self._ngmax[0] - 1
                    G = np.array([igx, igy, igz])
                    dump = (G + kvec).reshape(1,-1)
                    #print(dump.shape)
                    sumkg = np.dot(dump, self._b)
                    #print(sumkg.shape)
                    #print(sumkg)
                    #raise SystemExit
                    etot = np.sum(sumkg * sumkg)
                    if etot < Ec:
                        GVEC.append([igx,igy,igz])
                            
        return np.array(GVEC)

    def gvectors(self, ikpt=0, gamma=False):
        '''
        Generate the G-vectors that satisfies the following relation
            (G + k)**2 / 2 < ENCUT
        '''
        assert 0 <= ikpt  <= self._nk - 1,  'Invalid kpoint index!'

        # kvec = self._kvecs[ikpt-1]
        dump = self.readKHeader(0, ikpt)
        nplw = dump[0]
        kvec = dump[1]
        ngrid = 2 * self._ngmax + 1
        # fx, fy, fz = [fftfreq(n) * n for n in self._ngrid]
        # fftfreq in scipy.fftpack is a little different with VASP frequencies
        fx = [ii if ii < ngrid[0] / 2 + 1 else ii - ngrid[0]
                for ii in range(ngrid[0])]
        fy = [jj if jj < ngrid[1] / 2 + 1 else jj - ngrid[1]
                for jj in range(ngrid[1])]
        fz = [kk if kk < ngrid[2] / 2 + 1 else kk - ngrid[2]
                for kk in range(ngrid[2])]
        if gamma:
            # parallel gamma version of VASP WAVECAR exclude some planewave
            # components, -DwNGZHalf
            kgrid = np.array([(fx[ii], fy[jj], fz[kk])
                              for kk in range(ngrid[2])
                              for jj in range(ngrid[1])
                              for ii in range(ngrid[0])
                              if (
                                  (fz[kk] > 0) or
                                  (fz[kk] == 0 and fy[jj] > 0) or
                                  (fz[kk] == 0 and fy[jj] == 0 and fx[ii] >= 0)
                              )], dtype=float)
        else:
            kgrid = np.array([(fx[ii], fy[jj], fz[kk])
                              for kk in range(ngrid[2])
                              for jj in range(ngrid[1])
                              for ii in range(ngrid[0])], dtype=float)

        # Kinetic_Energy = (G + k)**2 / 2
        # HSQDTM    =  hbar**2/(2*ELECTRON MASS)
        KENERGY = HSQDTM * np.linalg.norm(
                    np.dot(kgrid + kvec[np.newaxis,:] , self._b), axis=1
                )**2
        # find Gvectors where (G + k)**2 / 2 < ENCUT
        Gvec = kgrid[np.where(KENERGY < self._Ec)[0]]

        assert Gvec.shape[0] == nplw, 'No. of planewaves not consistent! %d %d %d' % \
                (Gvec.shape[0], nplw, np.prod(ngrid))
        return np.asarray(Gvec, dtype=int)

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
        self._WC = WAVECAR
        self._nb = self._WC.getNBands()
        self._ispin = ispin
        self._ik = ik
        self._nplw, self._kvec, self._eig, self._FermiW = self.readKHeader()
        self._wps = self.readWPS()
        

        
    def getWPS(self): return self._wps
    def getNPlw(self): return self._nplw
    def getKVec(self): return self._kvec
    def getEig(self): return self._eig
    def getFermiW(self): return self._FermiW

    def readKHeader(self):
        kheader = self._WC.readKHeader(self._ispin, self._ik)
        nplw, kvec, eig, FermiW = kheader
        return nplw, np.array(kvec), np.array(eig), np.array(FermiW)
    
    def readWPS(self):
        WC = self._WC
        nb = self._nb
        nplw = self._nplw
        wps = np.zeros([nb, nplw], dtype=np.complex128)
        for ib in range(nb):
            wps[ib] = WC.readBandCoeff(self._ispin, self._ik, ib)
        return wps

#    def getOverlap(self):
#        nb = nb
#        wps = wps
#        S = np.dot(
#            np.conj(wps), wps.T
#        )
#        return S
#
#    def getWMatrix(self, tol):
#        """
#        Construct the occupation number diagonal matrix.
#            Small occupation numbers will be replaced by a small cutoff value tol, to bypass
#            possible numerical problems due to linear dependence.
#            see Szabo, "Modern Quantum Chemistry", Sec. 3.4.5 for details.
#        """
#        W = np.array([max(tol, x) for x in FermiW])
#        return np.diag(W)
#
#    def getOrthMatrix(self, OccWeighted, tol):
#        """
#        ***
#            Incorrect for AE purpose, but might be usefule for later transformations.
#        ***
#        Construct the unitary transform matrix to orthogonalize the pseudo wavefunction.
#            use Lowding's symmetric or occupation weighted orthogonalization scheme.
#            Lowding's:
#                   C = S^{-1/2}
#            Occupation weighted:
#                   C = W(WSW)^{-1/2}
#        Args:
#            OccWeighted   : logical (optional), occupation weighted?
#            tol           : float (optional), small value for occupation cutoff
#        Returns:
#            C    : ndarray (nb x nb), the unitary transform matrix.
#        """
#        S = self.getOverlap()
#        if OccWeighted:
#            W = self.getWMatrix(tol)
#            wsw = np.dot(W, np.dot(S, W)) # (WSW)
#            wsw = scipy.linalg.inv(wsw)   # (WSW)^{-1}
#            wsw = scipy.linalg.sqrtm(wsw) # (WSW)^{-1/2}
#            C = np.dot(W,wsw)
#        else:
#            C = scipy.linalg.inv(S)       # (S)^{-1}
#            C = scipy.linalg.sqrtm(C)     # (S)^{-1/2}
#
#        return C
#
#
#    def getWAE(self, OccWeighted = True, tol = 1.0E-3, check = True):
#        """
#        Orthogonalize the WPS using the overlap matrix.
#        The overlap matrix can be constructed by the projector coeffs written by
#            VASP
#
#        Args:
#            OccWeighted : logical
#            tol         : float
#            check       : logical, check < WAE | WAE > == I
#        Returns:
#            w           : ndarray (nb x nplw), all-electron wavefunction
#        """
#        C = self.getOrthMatrix(OccWeighted = OccWeighted, tol = tol)
#        w = self.getWPS()
#        w = np.dot(w.T, C)
#        w = w.T
#        #print('checking orthgonality...')
#
#        I = np.eye(len(w), dtype=np.complex128)
#        WW = np.dot(
#            np.conj(w), w.T
#        )
#        orth_success = np.allclose(WW, I)
#        if orth_success:
#            #print("Wavefunction orthogonalized.")
#            return w
#        else:
#            raise ValueError('Orthognalization failed.')
        




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
