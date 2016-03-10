#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Module imports
from __future__ import print_function
#==============================================================================
#  Some important Parameters, to convert to a.u., 
#  taken from VASP.4.6/constant.inc
#  - AUTOA  = 1. a.u. in Angstroem
#  - RYTOEV = 1 Ry in Ev
#  - EVTOJ  = 1 eV in Joule
#  - AMTOKG = 1 atomic mass unit ("proton mass") in kg
#  - BOLKEV = Boltzmanns constant in eV/K
#  - BOLK   = Boltzmanns constant in Joule/K

AUTOA = 0.529177249
RYTOEV = 13.605826
EVTOJ = 1.60217733E-19
AMTOKG = 1.6605402E-27
BOLKEV = 8.6173857E-5
BOLK = BOLKEV * EVTOJ
EVTOKCAL = 23.06

# FELECT = (the electronic charge)/(4*pi*the permittivity of free space)
#         in atomic units that is just e^2
# EDEPS = electron charge divided by the permittivity of free space
#         in atomic units that is just 4 pi e^2
# HSQDTM = (plancks CONSTANT/(2*PI))**2/(2*ELECTRON MASS)
#

PI = 3.141592653589793238
TPI = 2.0 * PI
CITPI = 1.0j * TPI
FELECT = 2.0 * AUTOA * RYTOEV
EDEPS = 4.0 * PI * 2.0 * RYTOEV * AUTOA
HSQDTM = RYTOEV * AUTOA * AUTOA

AUTOA2 = AUTOA * AUTOA
AUTOA3 = AUTOA2 * AUTOA
AUTOA4 = AUTOA2 * AUTOA2
AUTOA5 = AUTOA2 * AUTOA3

# Add for reduced mass calculation
ALPHA = 1.0/137.00
C0 = 299792458.0
V0 = ALPHA*C0
NU0 = V0/AUTOA*1E10
NU0_THz = NU0 / 1E12
AMTOAU = 1822.88839

if __name__ == '__main__':
    print(NU0_THz)
