#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import yaml
import argparse
import version
import logging
import sys
import wavefunction as wf
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
if os.path.exists(opath):
    os.mkdir(opath)
else:
    print("\n*******************************************************")
    print( "* Warning: files in directory {} might be overwritten! *")
    print("*******************************************************\n")
for ispin in range(param.get('nspin')):
    rho_0_fd_tot = np.zeros(NGF)
    drho_P_tot = np.zeros(NGF)
    drho_I_tot = np.zeros(NGF)
    rho_0_orig_tot = np.zeros(NGF)

    for ik in range(param.get('nkpts')):
        print("\n" + "="*50)
        print("\n Calculation for spin = {}, kpt = {}".format(ispin,ik))
        print("\n"+"="*50+"\n")
        # first get the WaveFunction objects @(isipn,ik)
        psi_0 = wf.WaveFunction(wc0, ispin, ik, normalize=True)
        psi_b = wf.WaveFunction(wcb, ispin, ik, normalize=True)
        psi_f = wf.WaveFunction(wcf, ispin, ik, normalize=True)
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
        
        # find the bands with non-negligible contributions to the (change of) Fermi level LDOS
        #    this contribution is Gaussian weighted:
        #        G(Ei, EF, sigma) >= cutoff
        bands_contrib = helper.getBandsRange(psi_0, psi_b, psi_f, param)
        writelog.write_bands_contrib(psi_0, psi_b, psi_f, bands_contrib)

        # finite difference for dpsi (psi^+ - psi^-) and psi0_fd (psi^+ + psi^-)
        print("\nFinite difference calculation for undisturbed and the change of wavefunctions")
        psi_fd = finitediff.finite_difference(psi_b, psi_f, bands_contrib, param)

        # get required psi_0 for LDOS calculation
        band_init, band_final = bands_contrib.get('bands_range')
        psi_0_calc = psi_0.getWAE()[band_init : band_final+1]

        # IETS calculation
        if method.startswith('T'):
            th_results = th.TersoffHamann(psi_fd, psi_0_calc, index, param)
            fileout.save_thdata(th_results, ispin, ik, param)
            rho_0_orig_tot += th_results.get('rho_0_orig') * kweight[ik]
            rho_0_fd_tot += th_results.get('rho_0_fd') * kweight[ik]
            drho_P_tot += th_results.get('drho_P') * kweight[ik]
            drho_I_tot += th_results.get('drho_I') * kweight[ik]
        elif method.startswith('B'):
            print('Get into Bardeen method for IETS...')
            print('  Not implemented yet, exit...')
            #raise SystemExit


    np.save('{}/rho_0_orig.tot.spin-{:1d}.npy'.format(opath, ispin), rho_0_orig_tot)
    np.save('{}/rho_0_fd.tot.spin-{:1d}.npy'.format(opath, ispin),rho_0_fd_tot)
    np.save('{}/drho.P.tot.spin-{:1d}.npy'.format(opath, ispin), drho_P_tot)
    np.save('{}/drho.I.tot.spin-{:1d}.npy'.format(opath, ispin), drho_I_tot)




#








        
