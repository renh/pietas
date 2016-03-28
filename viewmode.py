#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import outcar
import argparse
import writemodes

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input',
                    help = 'VASP frequency calculation OUTCAR with IBRION=5',
                    default = 'OUTCAR'
)


args = parser.parse_args()

VASP_OUTCAR = args.input
try:
    fh_outcar = open(VASP_OUTCAR, 'r')
    fh_outcar.close()
except:
    raise IOError('Failed to open file {}, exit...'.format(VASP_OUTCAR))

# check whether the OUTCAR is generated from freq calculation
# only support IBRION=5 currently
vo = outcar.OUTCAR(VASP_OUTCAR)
IBRION = int(vo.getParameter('IBRION'))
if IBRION  == 5:
    print("Confirmed {} is a VASP IBRION=5 calculation".format(VASP_OUTCAR))
else:
    raise ValueError("I need a VASP IBRION=5 calculation to view the vib modes.\n"
                     "IBRION = {} in the current file.\nexit...".format(IBRION)
    )

# first extract system information: Ions, coordinates and normal modes
IonList = vo.getIonList()
#print(IonList)
X0 = vo.getX0()
NIons = len(IonList)
dof = vo.getDOF()
modes = []
freqs =[]
print("Parsing the {} normal modes...".format(dof), end=' ')
for i in range(dof):
    imode = i + 1
    mode = vo.getNormalMode(imode)
    freqs.append(mode.get('Omega')[-2])
    modes.append(mode.get('mode'))
print('DONE!')

# package into a dictionary
NormalModes = {}
NormalModes['IonList'] = IonList
NormalModes['X0'] = X0
NormalModes['DOF'] = dof
NormalModes['frequency'] = freqs
NormalModes['modes'] = modes

writemodes.write_molden('vib.mol', NormalModes)


