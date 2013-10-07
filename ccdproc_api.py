#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from astropy.nddata import NDdata
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
data=100+10*np.random.random((110,100))
ccddata=CCDData(data=data)
ccddata=CCDData(NDData.NDData)
ccddata=CCDData(pyfits.ImageHDU)

#Setting basic properties of the object
# ----------------------
ccddata.variance=data**0.5
ccddata.mask=np.ones(110,100)
ccddata.flags=np.zeros(110,100)
ccddata.wcs=None  
ccddata.meta={}
ccddata.units=u.adu  #is this valid?  

#The ccddata class should have a functional form to create a CCDData
#object directory from a fits file
ccddata=fromFITS('img.fits')

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
ccddata=createvariance(ccddata, gain=1.0, readnoise=5.0)

#Overscan subtract the data
#Should be able to provide the meta data for
#the keyworkd or provide a section to define and
#possible an axis to specify the oritation of the 
#Question:  Best way to specify the section?  Should it be given 
#Error Checks: That the section is within the image
ccddata=subtract_overscan(ccddata, section='[:,100:110]', function='polynomial', order=3)

#trim the images--the section gives the  part of the image to keep
#That the trim section is within the image
ccddata=trim_image(ccddata, section='[0:100,0:100]')

#subtract the master bias.  Although this is a convenience function as subtracting
#the two arrays will do the same thing.   This should be able to handle logging of
#of subtracting it off (logging should be added to NDData and then this is really
#just a convenience function
#Error checks: the masterbias and image are the same shape
masterbias=NDData.NDData(np.zeros(100,100))
ccddata=subtract_bias(ccddata, masterbias)

#correct for dark frames
#Error checks: the masterbias and image are the same shape
masterdark=NDData.NDData(np.zeros(100,100))
ccddata=subtract_dark(ccddata,darkframe)

#correct for gain--once again gain should have a unit and even an error associated with it.  
ccddata=gain_correct(ccddata, gain=1.0)
#Also the gain may be non-linear
ccddata=gain_correct(ccddata, gain=np.array([1.0,0.5e-3])
#although then this step should be apply before any other corrections if it is non-linear
#but that is more up to the person processing their own data.

#crosstalk corrections--also potential a convenience function, but basically multiples the
#xtalkimage by the coeffient and then subtracts it.  It is kept general because this will
#be dependent on the CCD and the set up of the CCD.   Not applicable for a single CCD
#situation
#Error checks: the xtalkimage and image are the same shape
xtalkimage=NDData.NDData(np.zeros(100,100))
ccddata=xtalk_correct(ccddata, xtalkimage, coef=1e-3)

#flat field correction--this can either be a dome flat, sky flat, or an 
#illumination corrected image.  This step should normalize by the value of the
#flatfield after dividing by it.
#Error checks: the  flatimage and image are the same shape
#Error checks: check for divive by zero
#Features: If the flat is less than minvalue, minvalue is used
flatimage=NDData.NDData(np.ones(100,100))
ccddata=flat_correct(ccddata, flatimage, minvalue=1)

#fringe correction or any correction that requires subtracting
#off a potentially scaled image
#Error checks: the  flatimage and image are the same shape
fringemage=NDData.NDData(np.ones(100,100))
ccddata=fringe_correct(ccddata, fringeimage, scale=1, operation='multiple')

#cosmic ray cleaning step--this should have options for different
#ways to do it with their associated steps.  We also might want to 
#implement this as a slightly different step.  The cosmic ray cleaning
#step should update the mask and flags. So the user could have options
#to replace the cosmic rays, only flag the cosmic rays, or flag and 
#mask the cosmic rays, or all of the above.
ccddata=cosmicray_laplace(ccddata, method='laplace', args=*kwargs)
ccddata=cosmicray_median(ccddata, method='laplace', args=*kwargs)

#Apply distortion corrections
#Either update the WCS or transform the frame
ccddata=distortion_correct(ccddata, distortion)


# ================
# Helper Functions 
# ================

#fit a 1-D function with iterative rejections and the ability to 
#select different functions to fit.  
#other options are reject parameters, number of iteractions 
#and/or convergernce limit
coef=iterfit(x, y, function='polynomial', order=3)

#fit a 2-D function with iterative rejections and the ability to 
#select different functions to fit.  
#other options are reject parameters, number of iteractions 
#and/or convergernce limit
coef=iterfit(data, function='polynomial', order=3)

#in addition to these operations, basic addition, subtraction
# multiplication, and division should work for CCDDATA objects
ccddata= ccdata + ccddata
ccddata= ccddata * 2


#combine a set of NDData objects
alldata=combine([ccddata, ccddata2], method='average', reject=None, **kwargs)

#re-sample the data to different binnings (either larger or smaller)
ccddata=rebin(ccddata, binning=(2,2))

#tranform the data--ie shift, rotate, etc
#question--add convenience functions for image shifting and rotation?
#should udpate WCS although that would actually be the prefered method
ccddata=transform(ccddata, transform, conserve_flux=True)
