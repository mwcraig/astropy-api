#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np

from astropy.nddata import NDData
from astropy.io import fits
from astropy import units as u

import ccdproc
'''
The ccdproc package provides tools for the reduction and
analysis of optical data captured with a CCD.   The package
is built around the CCDData class, which has built into
it all of the functions to process data.  The CCDData object
contains all the information to describe the 2-D readout
from a single amplifier/detector.

The CCDData class inherits from the NDData class as its base object
and the object on which all actions will be performed.  By
inheriting from the CCD data class, basic array manipulation
and error handling are already built into the object.

The CCDData task should be able to properly deal with the
propogation of errors and propogate bad pixel frames
through each of the tasks.  It should also update the meta
data, units, and WCS information as the data are processed
through each step.

The following functions are required for performing basic CCD correction:
-creation of variance frame
-overscan subtraction
-bias subtraction
-trimming the data
-gain correction
-xtalk correction
-dark frames correction
-flat field correction
-illumination correction
-fringe correction
-scattered light correction
-cosmic ray cleaning
-distortion correction

In addition to the CCDData and CCDList class, the ccdproc does
require some additional features in order to properly
reduce CCD data. The following features are required
for basic processing of CCD data:
-fitting data
-combining data
-re-sampling data
-transforming data

All actions of ccdproc should be logged and recorded.

Multi-Extension FITS files can be handled by treating
each extension as a CCDData object and

'''

# ============
# Base Objects
# ============
'''
CCDData is an object that inherits from NDData class and specifically
describes an object created from the single readout of a CCD.

Users should be able to create a CCDData object from scratch, from
an existing NDData object, or a single extension from a FITS file.

In the case of the CCDData, the parameter 'uncertainty' will
be mapped to variance as that will be more explicitly describing
the information that will be kept for the processing of the

'''
data = 100 + 10 * np.random.random((110, 100))
ccddata = ccdproc.CCDData(data=data)
ccddata = ccdproc.CCDData(NDData.NDData(data))
ccddata = ccdproc.CCDData(fits.ImageHDU(data))

#Setting basic properties of the object
# ----------------------
ccddata.variance = data**0.5
ccddata.mask = np.ones(110, 100)
ccddata.flags = np.zeros(110, 100)
ccddata.wcs = None
ccddata.meta = {}

# making the metadata a Header instance gets us a case-insensitive dictionary...
assert isinstance(ccddata.meta, fits.Header)
ccddata.units = u.adu  # is this valid?

#The ccddata class should have a functional form to create a CCDData
#object directory from a fits file
ccddata = ccdproc.CCDData.fromFITS('img.fits')

'''
Keyword is an object that represents a key, value pair for use in passing
data between functions in ``ccdproc``. The value is an astropy.units.Quantity,
with the unit specified explicitly when the Keyword instance is created.
The key is case-insensitive, and synonyms can be supplied that will be used
to look for the value in CCDData.meta.
'''
key = ccdproc.Keyword('exposure', unit=u.sec, synonyms=['exptime'])
header = fits.Header()
header['exposure'] = 15.0
# value matched  by keyword name exposure
value = key(header)
assert value == 15 * u.sec
del header['exposure']
header['exptime'] = 15.0
# value matched by synonym exptime
value = key(header)
assert value == 15 * u.sec
# inconsistent values in the header raise an error:
header['exposure'] = 25.0
value = key(header)  # raises ValueError

# Functional Requirements
# ----------------------
# A number of these different fucntions are convenient functions that
# just outline the process that is needed.   The important thing is that
# the process is being logged and that a clear process is being handled
# by each step to make building a pipeline easy.   Then again, it might
# not be easy to handle all possible steps which are needed, and the more
# important steps will be things which aren't already handled by NDData.

#All functions should propogate throught to the variance frame and
#bad pixel mask

#convenience function based on a given value for
#the readnoise and gain.   Units should be specified
#for these values though.
#Question: Do we have an object that is just a scalar
#and a unit?  Or a scalar, unit and an error?   ie, This
#could actually be handled by the gain and readnoise being
#specified as an NDData object
ccddata = ccdproc.createvariance(ccddata, gain=1.0, readnoise=5.0)

#Overscan subtract the data
#Should be able to provide the meta data for
#the keyworkd or provide a section to define and
#possible an axis to specify the oritation of the
#Question:  Best way to specify the section?  Should it be given
#Error Checks: That the section is within the image
ccddata = ccdproc.subtract_overscan(ccddata, section='[:,100:110]',
                                    function='polynomial', order=3)

#trim the images--the section gives the  part of the image to keep
#That the trim section is within the image
ccddata = ccdproc.trim_image(ccddata, section='[0:100,0:100]')

#subtract the master bias. Although this is a convenience function as
#subtracting the two arrays will do the same thing. This should be able
#to handle logging of subtracting it off (logging should be added to NDData
#and then this is really just a convenience function
#Error checks: the masterbias and image are the same shape
masterbias = NDData.NDData(np.zeros(100, 100))
ccddata = ccdproc.subtract_bias(ccddata, masterbias)

#correct for dark frames
#Error checks: the masterbias and image are the same shape
#Options: Exposure time of the data image and the master dark image can be
#         specified as either an astropy.units.Quantity or as a ccdata.Keyword;
#         in the second case the exposure time will be extracted from the
#         metadata for each image.
masterdark = ccdproc.CCDData(np.zeros(100, 100))
masterdark.meta['exptime'] = 30.0
ccddata.meta['EXPOSURE'] = 15.0

exposure_time_key = ccdproc.Keyword('exposure',
                                    unit=u.sec,
                                    synonyms=['exptime'])

# explicitly specify exposure times
ccddata = ccdproc.subtract_dark(ccddata, masterdark,
                                data_exposure=15 * u.sec,
                                dark_exposure=30 * u.sec,
                                scale=True
                                )

# get exposure times from metadata
ccddata = ccdproc.subtract_dark(ccddata, masterdark,
                                exposure_time=exposure_time_key,
                                scale=True)

#correct for gain--once again gain should have a unit and even an error
#associated with it.

# gain can be specified as a Quantity...
ccddata = ccdproc.gain_correct(ccddata, gain=1.0 * u.ph / u.adu)
# ...or the gain can be specified as a ccdproc.Keyword:
gain_key = ccdproc.Keyword('gain', unit=u.ph / u.adu)
ccddata = ccdproc.gain_correct(ccddata, gain=gain_key)

#Also the gain may be non-linear
ccddata = ccdproc.gain_correct(ccddata, gain=np.array([1.0, 0.5e-3]))
#although then this step should be apply before any other corrections
#if it is non-linear, but that is more up to the person processing their
#own data.

#crosstalk corrections--also potential a convenience function, but basically
#multiples the xtalkimage by the coeffient and then subtracts it.  It is kept
#general because this will be dependent on the CCD and the set up of the CCD.
#Not applicable for a single CCD situation
#Error checks: the xtalkimage and image are the same shape
xtalkimage = NDData.NDData(np.zeros(100, 100))
ccddata = ccdproc.xtalk_correct(ccddata, xtalkimage, coef=1e-3)

#flat field correction--this can either be a dome flat, sky flat, or an
#illumination corrected image.  This step should normalize by the value of the
#flatfield after dividing by it.
#Error checks: the  flatimage and image are the same shape
#Error checks: check for divive by zero
#Features: If the flat is less than minvalue, minvalue is used
flatimage = NDData.NDData(np.ones(100, 100))
ccddata = ccdproc.flat_correct(ccddata, flatimage, minvalue=1)

#fringe correction or any correction that requires subtracting
#off a potentially scaled image
#Error checks: the  flatimage and image are the same shape
fringeimage = NDData.NDData(np.ones(100, 100))
ccddata = ccdproc.fringe_correct(ccddata, fringeimage, scale=1,
                                 operation='multiple')

#cosmic ray cleaning step--this should have options for different
#ways to do it with their associated steps.  We also might want to
#implement this as a slightly different step.  The cosmic ray cleaning
#step should update the mask and flags. So the user could have options
#to replace the cosmic rays, only flag the cosmic rays, or flag and
#mask the cosmic rays, or all of the above.
ccddata = ccdproc.cosmicray_laplace(ccddata, method='laplace')
ccddata = ccdproc.cosmicray_median(ccddata, method='laplace')

#Apply distortion corrections
#Either update the WCS or transform the frame
ccddata = ccdproc.distortion_correct(ccddata, distortion)


# =======
# Logging
# =======

# By logging we mean simply keeping track of what has been to each image in
# its as opposed to logging in the sense of the python logging module. Logging
# at that level is expected to be done by pipelines using the functions in
# ccdproc.

# for the purposes of illustration this document describes how logging would
# be handled for subtract_bias; handling for other functions would be similar.

# OPTION: One entry is added to the metadata for each processing step and the
# key added is the __name__ of the processing step.

# Subtracting bias like this:

ccddata = ccdproc.subtract_bias(ccddata, masterbias)

# adds a keyword to the metadata:

assert ccddata.meta['subtract_bias']  # name is the __name__ of the
                                      # processing step

# this allows fairly easy checking of whether the processing step is being
# repeated.

# OPTION: One entry is added to the metadata for each processing step and the
# key added is more human-friendly.

# Subtracting bias like this:

ccddata = ccdproc.subtract_bias(ccddata, masterbias)

# adds a keyword to the metadata:

assert ccddata.meta['bias_subtracted']  # name reads more naturally than previous
                                        # option

# OPTION: Each of the processing steps allows the user to specify a keyword
# that is added to the metadata. An optional value can also be given.

#  SUB-OPTION: Allow the Keyword class to have string values? Then could use
#  those as an argument for add_keywords

# add keyword named SUBBIAS but don't set a value for it:

ccddata = ccdproc.subtract_bias(ccddata, masterbias, add_keyword='SUBBIAS')

# add keyword CALSTAT with value 'B':

ccddata = ccdproc.subtract_bias(ccddata, masterbias, 
                                add_keyword='CALSTAT',
                                value='B')

# ================
# Helper Functions
# ================

#fit a 1-D function with iterative rejections and the ability to
#select different functions to fit.
#other options are reject parameters, number of iteractions
#and/or convergernce limit
coef = ccdproc.iterfit(x, y, function='polynomial', order=3)

#fit a 2-D function with iterative rejections and the ability to
#select different functions to fit.
#other options are reject parameters, number of iteractions
#and/or convergernce limit
coef = ccdproc.iterfit(data, function='polynomial', order=3)

#in addition to these operations, basic addition, subtraction
# multiplication, and division should work for CCDDATA objects
ccddata = ccddata + ccddata
ccddata2 = ccddata * 2


#combine a set of NDData objects
alldata = ccdproc.combine([ccddata, ccddata2], method='average',
                          reject=None)

#re-sample the data to different binnings (either larger or smaller)
ccddata = ccdproc.rebin(ccddata, binning=(2, 2))

#tranform the data--ie shift, rotate, etc
#question--add convenience functions for image shifting and rotation?
#should udpate WCS although that would actually be the prefered method
ccddata = ccdproc.transform(ccddata, transform, conserve_flux=True)
