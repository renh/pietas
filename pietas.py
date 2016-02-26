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


#=================================================
# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input',
                    help = 'Input file for configuration', default = None
)

parser.add_argument('-v', '--version',
                    help = 'Version information', action='store_true'
)

parser.add_argument('-o', '--output',
                    help = 'Oputput file', default = 'run.log'
)

parser.add_argument('-l', '--log',
                    help = 'Log verbosity level', type = int,
                    default = 20
)

args = parser.parse_args()
#=================================================

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
#=================================================    

print(param)

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


#
# arrange real-space grid by using real-space lattice and real grid density
#
rgd = param.get('real grid density')
a = wc0.getRealLatt()
NGR = grid.real_grid(rgd, a)
print("\nReal space grid (for FFT): NGXF = {:d}, NGYF = {:d}, NGZF = {:d}".format(
    *NGR
))


# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input',
                    help = 'Input file for configuration', default = None
)

parser.add_argument('-v', '--version',
                    help = 'Version information', action='store_true'
)

parser.add_argument('-o', '--output',
                    help = 'Oputput file', default = 'run.log'
)

parser.add_argument('-l', '--log',
                    help = 'Log verbosity level', type = int,
                    default = 20
)

args = parser.parse_args()
#=================================================

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
#=================================================    

print(param)

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


#
# arrange real-space grid by using real-space lattice and real grid density
#
rgd = param.get('real grid density')
a = wc0.getRealLatt()
NGR = grid.real_grid(rgd, a)
print("\nReal space grid (for FFT): NGXF = {:d}, NGYF = {:d}, NGZF = {:d}".format(
    *NGR
))


# parse OUTCAR for necessary parameters

#=================================================    
# real calculation starts here
# all calculation will be conducted k-point wise,
# I will prepare two set of variables to track the current and total quantities along with the loops.
# loops over spinors and kpoints...
#=================================================    
#

#








        
