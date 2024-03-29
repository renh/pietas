from __future__ import print_function
import os
import datetime
import shutil
import time
import argparse
import numpy as np
import outcar
from constant import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input',
                    help = 'Input (VASP IBRION=5) OUTCAR',
                    default = 'OUTCAR')
parser.add_argument('-f', '--scale',
                    help = 'Scale factor to displace configuration along vib mode',
                    default = None, type=float)

parser.add_argument('-m', '--mode',
                    help = 'Mode number (from 1) for which to generate new configurations',
                    type = int, default = None)

parser.add_argument('-o', '--outpath',
                    help = 'Output path for generated displaced POSCAR files',
                    default = None,
        )
args = parser.parse_args()

VASP_OUTCAR = args.input
try:
    fh_outcar = open(VASP_OUTCAR, 'r')
    fh_outcar.close()
except:
    raise IOError("Failed open file '{}', exit..".format(VASP_OUTCAR))

# scale factor related
if args.scale is None:
    print('\nNo scale factor provided. I will use f = 0.5 to generate '
            'the distored structures along vib modes.')
    # time.sleep(4)
    scale_factor = 0.5
else:
    scale_factor = args.scale
    print("\nUsing scale factor f = {} for displacement".format(scale_factor))

# output path
if args.outpath is None:
    outpath = './poscar_disp_f_{:g}'.format(scale_factor)
    print('\nOutput path not specified, I will use {} to store POSCAR files.'.format(
        outpath))
else:
    outpath = args.outpath
    print('\nAll displaced POSCAR files will be found in {}'.format(outpath))
    
# first check OUTCAR is generated from a IBRION=5 calculation
out = outcar.OUTCAR(VASP_OUTCAR)
ibrion = int(out.getParameter('IBRION'))
if ibrion != 5:
    print('\nI am now only support IBRION=5 calculation from VASP')
    raise SystemExit
print('\nConfirmed OUTCAR is generated by a IBRION = 5 calculation')


# get degrees of freedom
dof = out.getDOF()
print('\nThere are in total {} vibrational modes in the calculation.'.format(
    dof)
)

# get equilibrium configuration
IonsPerType = out.getIonsPerType()
IonTypes = out.getIonTypes()
ions = list(zip(IonTypes, IonsPerType))
NAtoms = out.getNIons()
Masses = out.getIonMasses()
#print(ions)

X0 = out.getX0()
latt = out.getLattice()

# construct mode-list for which will be displaced
if args.mode is None:
    modes = [i+1 for i in range(dof)]
else:
    modes = [args.mode]

# make the new directory for generated displaced configurations
# if exists, back up with a tailing time string
if os.path.exists(outpath):
    time_str = datetime.datetime.fromtimestamp(
        os.path.getmtime(outpath)
    ).strftime("%Y%m%d-%H%M%S")
    new_path = "{}.{}".format(outpath, time_str)
    print('\nDirectory {} already exists, back up to {}'.format(
        outpath, new_path
    ))
    shutil.move(outpath, new_path)
os.mkdir(outpath)

# record the scale factor for each normal mode in the fixed_norm generation
fname = '{}/scale.dat'.format(outpath)
scale_fh = open(fname, 'w')
scale_fh.write('{:g}'.format(scale_factor))

# construct POSCAR header
first_line = "{} " * len(ions) + "\n"
first_line = first_line.format(*[ion[0] for ion in ions])
header = first_line + " 1.00000\n"
for i in range(3):
    line = "{:24.16f}{:24.16f}{:24.16f}\n".format(*latt[i])
    header += line

sixth_line = " {}" * len(ions) + "\n"
sixth_line = sixth_line.format(*[ion[1] for ion in ions])
header += sixth_line
header += 'Cartesian\n'


# write un-displaced POSCAR
m0_fname = '{}/POSCAR.mode0'.format(outpath)
m0_fh = open(m0_fname, 'w')
m0_fh.write(header)
for i in range(NAtoms):
    m0_fh.write('{:24.16f}{:24.16f}{:24.16f}\n'.format(*(X0[i])))
m0_fh.close()


# displacing coordinates
print("\n\nGenerating displaced geometry:\n")

# modes is a list containing the required vib modes need to be displaced
# mode is an integer corresponds the mode number in VASP OUTCAR, starts from 1
AMTOAU = 1834.
for mode in modes:
    print("mode #{:3d} :  ".format(mode), end='')
    m = out.getNormalMode(mode)
    omega = m.get('Omega')[6] # vib frequency in meV
    # convert omega to atomic unit in energy (Hartree)
    omega_hartree = (omega / 1.0E3) / RYTOEV / 2.0
    l = m.get('mode')
    # print(l)
    
    mode_factor = np.sqrt(2.8 * omega_hartree * AMTOAU * Masses)

    dX = l * AUTOA * scale_factor / mode_factor.reshape([-1,1]).repeat(3, axis=1)
    # print(dX)

    RMSD = np.sqrt(np.sum(dX * dX) / NAtoms) 
    print("  RMSD = {:.6f} Ang, max displacement = {:.6f} Ang".format(
        RMSD, np.max(dX)))

    # create POSCAR files for positive and negative displaced coord
    #  and write headers to them
    poscar_p = '{}/POSCAR.back.{:03d}'.format(outpath, mode)
    poscar_p_fh = open(poscar_p, 'w')
    poscar_p_fh.writelines(header)
    poscar_m = '{}/POSCAR.forw.{:03d}'.format(outpath, mode)
    poscar_m_fh = open(poscar_m, 'w')
    poscar_m_fh.writelines(header)

    # write displaced coord to POSCARs
    fmt = "{:24.16f}"*3 + "\n"
    newX = X0 + dX
    # print("writing to {} and ".format(poscar_p), end='')
    for i in range(NAtoms):
        poscar_p_fh.write(fmt.format(*newX[i]))
    poscar_p_fh.close()

    newX = X0 - dX
    for i in range(NAtoms):
        poscar_m_fh.write(fmt.format(*newX[i]))
    poscar_m_fh.close()
    # print("{}...DONE!\n".format(poscar_m))






