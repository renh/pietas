#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import logging


version = '0.2'

def print_version_info():
    """
    Print version information of the current release.

    Args:
        None
    Returns:
        None
    Raises:
        None
    Remark:
        Version number stored in the module level global 'version'
    """
    print('pietas: Version {}'.format(version))


if __name__ == '__main__':
    print_version_info()
