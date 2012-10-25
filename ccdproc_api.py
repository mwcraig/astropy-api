#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from astropy.nddata import NDdata
'''
The ccdproc package provides tools for the reduction and 
analysis of optical data captured with a CCD.   

The CCDPROC task should be able to properly deal with the
propogation of errors and propogate bad pixle frames
through each of the tasks.

Issues to deal with:
-Input object types
-Object oriented vs. functional

'''

# ============
# Base Objects
# ============
'''
CCDFrame is an object that inherits from NDData class and specifically
describes an object created from the readout of a CCD.  

The CCDFrame should have the ability to be initialized from a 
NDData object or a FITS object.

'''

# Functional Requirements
# ----------------------
'''
The following algorithms are required for performing basic CCD Reductions:
-basic arithmatic
--additional
--subtraction
--multiplication
--division
--power
-statistical tools 
--iterative measurements
-fitting tools 
--1D
--2D
-Array transformation/re-sampling
-Array combining
-updating meta data

The following functions are required for performing basic CCD correction:
-creation of bad pixel map
-creation of variance frame
-overscan subtraction
-bias subtraction 
-gain correction
-xtalk correction
-dark frames correction
-flat field correction
-illumination correction
-fringe correction
-scattered light correction
-cosmic ray cleaning
-distortion correction 

More advanced features would include:
-astrometric corrections

'''


