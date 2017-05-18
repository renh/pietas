#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import yaml
import argparse
import version
import logging
import sys
import os
import wavefunction as wf
import normalcar
import checker
import grid
import outcar
import helper
import writelog
import finitediff
import th
import fileout
import numpy as np


# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input',
                    help='Input file for configuration', default=None)

parser.add_argument('-v', '--version',
                    help='Version information', action='store_true')

parser.add_argument('-o', '--output',
                    help='Oputput file', default='run.log')

parser.add_argument('-l', '--log',
                    help='Log verbosity level', type=int,
                    default=20)

args = parser.parse_args()
# =================================================

logging.basicConfig(filename=args.output, level=args.log)

if args.version:
    version.print_version_info()
    raise SystemExit

# load input configuration
if not args.input:
    print(" No configuration provided. Using '{} -i' or '-h' for more information.".format(
            sys.argv[0]))
    raise SystemExit
else:
    try:
        with open(args.input, 'r') as fh:
            param = yaml.load(fh)
    except:
        raise IOError('can not load configuration file')
# =================================================    

#print(param)
param['cutoff'] = float(param.get('cutoff'))
#
# connect to WAVECAR files for the equilibrium and backward/forward displaced systems
#

# get file name from input configuration
WAVECAR_files = param.get('wavecar')
wc0 = WAVECAR_files.get('equilibrium')
wcb = WAVECAR_files.get('backward')
wcf = WAVECAR_files.get('forward')

# check three WAVECARs are not the same
checker.same_file(wc0, wcb, wcf)

# check consistency for three WAVECAR files
#    NOTE that I can not guarantee the WAVECARs are fully complete and consistent
#    but most of the consistency problems might be detected...

checker.check_WAVECAR(wc0, wcb)
checker.check_WAVECAR(wc0, wcf)

# connect to WAVECAR files
wc0 = wf.WAVECAR(wc0)
wcb = wf.WAVECAR(wcb)
wcf = wf.WAVECAR(wcf)
# now wc0, wcb and wcf are WAVECAR classes conneted to corresponding WAVECAR files.

# get general info for the pseudo wavefunctions
# nspin, nk, nb
param['nspin'] = wc0.getNSpin()
param['nkpts'] = wc0.getNKpts()
param['nbands'] = wc0.getNBands()

#
# arrange real-space grid by using real-space lattice and real grid density
#
rgd = param.get('real grid density')
a = wc0.getRealLatt()
NGF = grid.real_grid(rgd, a)
print("\nReal space grid (for FFT): NGXF = {:d}, NGYF = {:d}, NGZF = {:d}".format(
    *NGF
))
param['NGF'] = NGF

# parse OUTCAR for necessary parameters
OUTCAR_file = param.get('outcar').get('equilibrium')
#print(OUTCAR_file)
vaspOUTCAR = outcar.OUTCAR(OUTCAR_file)
EFermi = vaspOUTCAR.getFermiEnergy()
print("\nFermi energy for equilibrium configuration: {:10.4f}".format(EFermi))
param['Fermi energy'] = EFermi
kvec, kweight = vaspOUTCAR.getKSampling()
param['k vectors'] = kvec
param['k weights'] = kweight

print("\nK-sampling info:")
print("          coordinates              weight")
for i in range(len(kvec)):
    print("{:10.6f}{:10.6f}{:10.6f}{:12.6f}".format(
        kvec[i,0], kvec[i,1], kvec[i,2], kweight[i]
    ))

# =================================================    
# real calculation starts here
# all calculation will be conducted k-point wise,
# I will prepare two set of variables to track the current and total quantities along with the loops.
# loops over spinors and kpoints...
# summation over kpts will be weighted by kweights
# =================================================    
#
method = param.get('approximation')

opath = param.get('output path')
if not os.path.exists(opath):
    os.mkdir(opath)
else:
    print("\n" + "*" * (54 + len(opath)) + 
          "\n* Warning: files in directory {} might be overwritten! *".format(
              opath) + 
          "\n" + "*" * (54 + len(opath)) + '\n')

# Read PAW projector info
# read projector coeffs before (spin, kpt) loop because the NormalCAR file
# is not written in the direct access mode. It's better read stuff once for all.
#nmc0 = normalcar.NormalCAR(normalcar=param.get('normalcar').get('equilibrium'), 
#        wavecar=param.get('wavecar').get('equilibrium'))
nmc_b = normalcar.NormalCAR(normalcar=param.get('normalcar').get('backward'), 
        wavecar=param.get('wavecar').get('backward'))
nmc_f = normalcar.NormalCAR(normalcar=param.get('normalcar').get('forward'), 
        wavecar=param.get('wavecar').get('forward'))

# (l,m) channel dimensions
LMDim_b = nmc_b.getLMDim()
LMDim_f = nmc_f.getLMDim()
assert(LMDim_b == LMDim_f)


# aug charges
Qij_b = nmc_b.getCQij()
Qij_f = nmc_f.getCQij()
assert(np.allclose(Qij_b, Qij_f))

# projector dimensions
LMMax_b = nmc_b.getLMMax()
LMMax_f = nmc_f.getLMMax()
assert(np.allclose(LMMax_b, LMMax_f))

# projector coeffs
Pij_b = nmc_b.getProjCoeffs()
Pij_f = nmc_f.getProjCoeffs()


# loop over spinors
for ispin in range(param.get('nspin')):
    rho_0_tot = np.zeros(NGF)
    drho_P_tot = np.zeros(NGF)
    drho_I_tot = np.zeros(NGF)

    #loop over kpts
    for ik in range(param.get('nkpts')):
        print("\n" + "="*50)
        print("\n Calculation for spin = {}, kpt = {}".format(ispin,ik))
        print("\n"+"="*50+"\n")
        
        # first get the WaveFunction objects @(isipn,ik)
        psi_0 = wf.WaveFunction(wc0, ispin, ik)
        psi_b = wf.WaveFunction(wcb, ispin, ik)
        psi_f = wf.WaveFunction(wcf, ispin, ik)
        kvec = psi_0.getKVec()
        nplw = psi_0.getNPlw()
        print(" k-vector: {:10.3f}{:10.3f}{:10.3f}".format(*kvec))


        GVEC = wc0.prepareGVEC(ik)
        if (len(GVEC) != nplw):
            raise ValueError("!!! Incorrect dimension for GVEC = {}, nplw = {}".format(
                len(GVEC), nplw
            ))
        print(" {} G vectors prepared.".format(nplw))

        index = grid.genIndex(GVEC, param.get('NGF'))
        
        #
        # find the bands with non-negligible contributions to the (change of) Fermi level LDOS
        #    this contribution is Gaussian weighted:
        #        G(Ei, EF, sigma) >= cutoff
        bands_contrib = helper.getBandsRange(psi_0, psi_b, psi_f, param)
        writelog.write_bands_contrib(psi_0, psi_b, psi_f, bands_contrib)
        
        #
        # check orthonormality of wavefunctions @(ispin, ik)
        #
        if param.get('check').get('orthonormality'):
            print('\n  checking orthonormality of WFs from backward displaced sys...')
            checker.check_orthonormal(psi_b, LMMax_b, Qij_b[:,:,:,ispin], 
                    Pij_b[:,:,ik,ispin])

            print('\n  checking orthonormality of WFs from forward displaced sys...')
            checker.check_orthonormal(psi_f, LMMax_f, Qij_f[:,:,:,ispin], 
                    Pij_f[:,:,ik,ispin])
        else:
            print('\n  Warning: WF orthonomality not checked. Proceeding...')

        #
        # Evaluate the 'overlap' matrix < + | - >
        #   only matrix elements between WFs considered in the calculation 
        #   will be calculated
        #
        
        # finite difference for dpsi (psi^+ - psi^-) and psi0_fd (psi^+ + psi^-)
        print("\nFinite difference calculation for undisturbed and the change of wavefunctions")

        # ugly function here, encapsulate PAW related data into classes later.
        psi_fd = finitediff.finite_difference(
                psi_b, psi_f, 
                Pij_b[:,:,ik,ispin], Pij_f[:,:,ik,ispin], 
                Qij_b[:,:,:,ispin], LMMax_b, 
                bands_contrib, param
                )
        print('finite difference finished')
        
        # #
        # # test TH calc/output for IETS
        # #
        # #eig = (psi_b.getEig() + psi_f.getEig()) / 2.0
        # band_init, band_final = bands_contrib.get('bands_range')
        # gw_fd = helper.Gaussian(eig[band_init:band_final+1], EFermi, param.get('sigma'))
        # rho_0 = th.calc_LDOS(psi_0_fd, gw_fd, fudge_factor, index,param)
        # drho_P = th.calc_LDOS(dpsi_P, gw_fd, fudge_factor, index, param)
        # np.save('{}/rho_0-k{}.npy'.format(opath,ik),rho_0)
        # np.save('{}/drho_P-k{}.npy'.format(opath,ik), drho_P)

        # rho_0_tot += rho_0 * kweight[ik]
        # drho_P_tot += drho_P * kweight[ik]
        # #
        # #
        # # end test here
        # #
        # continue

        # IETS calculation
        if method.upper().startswith('T'):
            th_results = th.TersoffHamann(psi_fd, index, param)
            #fileout.save_thdata(th_results, ispin, ik, param)
            np.save('{}/rho_0.k{}.spin{}.npy'.format(opath, ik, ispin),
                    th_results.get('rho_0'))
            np.save('{}/drho.P.k{}.spin{}.npy'.format(opath, ik, ispin),
                    th_results.get('drho_P'))
            np.save('{}/drho.I.k{}.spin{}.npy'.format(opath, ik, ispin),
                    th_results.get('drho_I'))
            
            # sum over kpt
            rho_0_tot += th_results.get('rho_0') * kweight[ik]
            drho_P_tot += th_results.get('drho_P') * kweight[ik]
            drho_I_tot += th_results.get('drho_I') * kweight[ik]
        elif method.upper().startswith('B'):
            print('Get into Bardeen method for IETS...')
            print('  Not implemented yet, exit...')
            raise SystemExit


    np.save('{}/rho_0.tot.spin-{:1d}.npy'.format(opath, ispin), rho_0_tot)
    np.save('{}/drho.P.tot.spin-{:1d}.npy'.format(opath, ispin), drho_P_tot)
    np.save('{}/drho.I.tot.spin-{:1d}.npy'.format(opath, ispin), drho_I_tot)




#








        
