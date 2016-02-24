from __future__ import print_function
import yaml
import argparse
import version
import logging
import sys

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

if not args.input:
    logging.critical(
        " No configuration provided. Using '{} -i' or '-h' for more information.".format(
            sys.argv[0]
    ))
    raise SystemExit
else:
    try:
        with open(args.input, 'r') as fh:
            param = yaml.load(fh)
    except:
        raise IOError('can not load configuration file')

#=================================================    





print(param)
        
