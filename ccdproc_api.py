#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np

from astropy.nddata import NDData
from astropy.io import fits
from astropy import units as u
from astropy.stats.funcs import sigma_clip

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
# initializing without a unit raises an error
ccddata = ccdproc.CCDData(data=data) # ValueError
ccddata = ccdproc.CCDData(data=data, unit=u.adu)
ccddata = ccdproc.CCDData(NDData.NDData(data), unit=u.photon)
ccddata = ccdproc.CCDData(fits.ImageHDU(data), unit=u.adu)

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
ccddata = ccdproc.CCDData.fits_ccddata_reader('img.fits', image_unit=u.adu)

# omitting a unit causes an error for now -- in the future an attempt should
# be made to extract the unit from the FITS header.
ccddata = ccdproc.CCDData.fits_ccddata_reader('img.fits')  # raises ValueError

# If a file has multiple extensions the desired hdu should be specified. It
# defaults to zero.
ccddata = ccdproc.CCDData.fits_ccddata_reader('multi_extension.fits',
                                              image_unit=u.adu,
                                              hdu=2)

# any additional keywords are passed through to fits.open, with the exception
# of those related scaling (do_not_scale_image_data and scale_back)
ccddata = ccdproc.CCDData.fits_ccddata_reader('multi_extension.fits',
                                              image_unit=u.adu,
                                              memmap=False)
# The call below raises a TypeErrror
ccddata = ccdproc.CCDData.fits_ccddata_reader('multi_extension.fits',
                                              image_unit=u.adu,
                                              do_not_scale_image_data=True)


# This function should then be registered with astropy.io.registry, and the 
# FITS format auto-identified using the fits.connect.is_fits so
# the standard way for reading in a fits image will be
# the following: 
ccddata = ccdproc.CCDData.read('img.fits', image_unit=u.adu)


# CCDData raises an error if no unit is provided; eventually an attempt to
# extract the unit from the FITS header should be made.

# Writing is handled in a similar fashion; the image ccddata is written to
# the file img2.fits with:
ccdproc.CCDData.fits_ccddata_writer(ccddata, 'img2.fits')

# all additional keywords are passed on to the underlying FITS writer, e.g.
ccdproc.CCDData.fits_ccddata_writer(ccddata, 'img2.fits', clobber=True)

# The writer is registered with unified io so that in practice a user will do
ccddata.write('img2.fits')

# NOTE: for now any flag, mask and/or unit for ccddata is discarded when
# writing. If you want all or some of that information preserved you must
# create the FITS files manually.

# To be completely explicit about this:
ccddata.mask = np.ones(110, 100)
ccddata.flags = np.zeros(110, 100)
ccddata.write('img2.fits')

ccddata2 = ccdproc.CCDData.read('img2.fits', image_unit=u.adu)
assert ccddata2.mask is None  # even though we set ccddata.mask before saving
assert ccddata2.flag is None  # even though we set ccddata.flag before saving

'''
Keyword is an object that represents a key, value pair for use in passing
data between functions in ``ccdproc``. The value is an astropy.units.Quantity,
with the unit specified explicitly when the Keyword instance is created.
The key is case-insensitive.
'''
key = ccdproc.Keyword('exposure', unit=u.sec)
header = fits.Header()
header['exposure'] = 15.0
# value matched  by keyword name exposure
value = key.value_from(header)
assert value == 15 * u.sec

# the value of a Keyword can also be set directly:
key.value = 20 * u.sec

# String values are accommodated by setting the unit to the python type str:

string_key = ccdproc.Keyword('filter', unit=str)
string_key.value = 'V'

'''
Combiner is an object to facilitate image combination. Its methods perform
the steps that are bundled into the IRAF task combine

It is discussed in detail below, in the section "Image combination"
'''
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

ccddata.unit = u.adu
# The call below should raise an error because gain and readnoise are provided
# without units.
ccddata = ccdproc.create_variance(ccddata, gain=1.0, readnoise=5.0)

# The electron unit is provided by ccddata
gain = 1.0 ccddata.electron / u.adu
readnoise = 5.0 * u.electron
# This succeeds because the units are consistent
ccddata = ccdproc.create_variance(ccddata, gain-gain, readnoise=readnoise)

# with the gain units below there is a mismatch between the units of the
# image after scaling by the gain and the units of the readnoise...
gain = 1.0 u.photon / u.adu

# ...so this should fail with an error.
ccddata = ccdproc.create_variance(ccddata, gain-gain, readnoise=readnoise)


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

assert 'subtract_bias' in ccddata.meta  # name is the __name__ of the
                                        # processing step

# this allows fairly easy checking of whether the processing step is being
# repeated.

# OPTION: One entry is added to the metadata for each processing step and the
# key added is more human-friendly.

# Subtracting bias like this:

ccddata = ccdproc.subtract_bias(ccddata, masterbias)

# adds a keyword to the metadata:

assert 'bias_subtracted' in ccddata.meta  # name reads more naturally than
                                          # previous option

# OPTION: Each of the processing steps allows the user to specify a keyword
# that is added to the metadata. The keyword can either be a string or a
# ccdproc.Keyword instance

# add keyword as string:
ccddata = ccdproc.subtract_bias(ccddata, masterbias, add_keyword='SUBBIAS')

# add keyword/value using a ccdproc.Keyword object:
key = ccdproc.Keyword('calstat', unit=str)
key.value = 'B'
ccddata = ccdproc.subtract_bias(ccddata, masterbias,
                                add_keyword=key)

# =================
# Image combination
# =================

# The ``combine`` task from IRAF performs several functions:
# 1. Selection of images to be combined by image type with optional grouping
#    into subsets.
# 2. Offsetting of images based on either user-specified shifts or on WCS
#    information in the image metadata.
# 3. Rejection of pixels from inclusion in the combination based on masking,
#    threshold rejection prior to any image scaling or zero offsets, and
#    automatic rejection through a variety of algorithms (minmax, sigmaclip,
#    ccdclip, etc) that allow for scaling, zero offset and in some cases
#    weighting of the images being combined.
# 4. Scaling and/or zero offset of images before combining based on metadata
#    (e.g. image exposure) or image statistics (e.g image median, mode or
#    average determined by either an IRAF-selected subset of points or a
#    region of the image supplied by the user).
# 5. Combination of remaining pixels by either median or average.
# 6. Logging of the choices made by IRAF in carrying out the operation (e.g.
#    recording what zero offset was used for each image).

# As much as is practical, the ccdproc API separates these functions, discussed
# in detail below.

# 1. Image selection: this will not be provided by ccdproc (or at least not
#    considered part of image combination). We assume that the user will have
#    selected a set of images prior to beginning combination.

# 2. Position offsets: In ccdproc this is not part of combine. Instead,
#    offsets and other transforms are handled by ccdproc.transform, described
#    below under "Helper Function"

# The combination process begins with the creation of a Combiner object,
# initialized with the images to be combined

combine = ccdproc.Combiner(ccddata1, ccddata2, ccddata3)

# 3. 
#   Masking: ccdpro.CCDData objects are already masked arrays, allowing
#   automatic exclusion of masked pixels from all operations.
#
#   Threshold rejection of all pixels with data value over 30000 or under -100:

combine.threshold_reject(max=30000, min=-100)

#   automatic rejection by min/max, sigmaclip, ccdclip, etc. provided through
#   one method, with different helper functions

# min/max
combine.clip(method=ccdproc.minmax)

# sigmaclip (relies on astropy.stats.funcs)
combine.clip(method=sigma_clip,
             sigma_high=3.0, sigma_low=2.0,
             centerfunc=np.mean,
             exclude_extrema=True)  # min/max pixels can be excluded in the 
                                    # sigma_clip. IRAF's clip excludes them

# ccdclip
combine.clip(method=ccdproc.ccdclip,
             sigma_high=3.0, sigma_low=2.0,
             gain=1.0, read_noise=5.0,
             centerfunc=np.mean,
             exclude_extrema=True)

# 4. Image scaling/zero offset with scaling determined by the mode of each
#    image and the offset by the median. This method calculates what are
#    effectively weights for each image

combine.calc_weights(scale_by=np.mode, offset_by=np.median)

# 5. The actual combination -- a couple of ways images can be combined

# median; the excluded pixels based on the individual image masks, threshold
# rejection, clipping, etc, are wrapped into a single mask for each image
combined_image = combine(method=np.ma.median)

# average; in this case image weights can also be specified
combined_image = combine(method=np.ma.mean, weights=[0.5, 1, 2])

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
