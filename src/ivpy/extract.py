import pandas as pd
import numpy as np
from PIL import Image

from skimage.io import imread
from skimage import color
from scipy.stats import entropy
from skimage.feature import greycomatrix, greycoprops
from sklearn.neighbors import KernelDensity

from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import Model
from keras.applications import InceptionV3

from .data import _typecheck,_colfilter
from .plottools import _progressBar

#------------------------------------------------------------------------------

def extract(feature=None,pathcol=None,aggregate=True):
    _typecheck(**locals())
    pathcol,xcol,ycol,facetcol = _colfilter(pathcol)

    if feature==None:
        raise ValueError("Must supply 'feature'")
    elif feature=='brightness':
        return _brightness(pathcol,aggregate)
    elif feature=='saturation':
        return _saturation(pathcol,aggregate)
    elif feature=='hue':
        return _hue(pathcol,aggregate)
    elif feature=='entropy':
        return _entropy_brightness(pathcol)
    elif feature=='std':
        return _std_brightness(pathcol)
    elif feature=='contrast':
        return _glcm(pathcol,prop='contrast')
    elif feature=='dissimilarity':
        return _glcm(pathcol,prop='dissimilarity')
    elif feature=='homogeneity':
        return _glcm(pathcol,prop='homogeneity')
    elif feature=='ASM':
        return _glcm(pathcol,prop='ASM')
    elif feature=='energy':
        return _glcm(pathcol,prop='energy')
    elif feature=='correlation':
        return _glcm(pathcol,prop='correlation')
    elif feature=='neural':
        return _neural(pathcol,tags=False)
    elif feature=='tags':
        return _neural(pathcol,tags=True)

#------------------------------------------------------------------------------

def _iterextract(pathcol,outstructure,breaks,pct,func,**kwargs):
    counter=-1
    for i in pathcol.index:
        counter+=1
        if counter in breaks:
            print pct[breaks.index(counter)],
        imgpath = pathcol.loc[i]
        outstructure.loc[i] = func(imgpath,**kwargs)
    return outstructure

#------------------------------------------------------------------------------

def _imgfilter(imgpath):
    """Returns HSV array. Might add auto-resizing to this as well"""

    img = imread(imgpath)
    if len(img.shape)==3:
        return color.rgb2hsv(img)
    elif len(img.shape)==2:
        return color.rgb2hsv(color.gray2rgb(img))

def _brightness(pathcol,aggregate):
    """Returns either average brightness or 10-bin distribution"""

    if isinstance(pathcol,basestring):
        if aggregate==True:
            return _hsv_mean(pathcol,axis=2)
        elif aggregate==False:
            return _hsv_10bin(pathcol,axis=2)

    elif isinstance(pathcol,pd.Series):
        breaks,pct = _progressBar(pathcol)
        if aggregate==True:
            featcol = pd.Series(index=pathcol.index)
            return _iterextract(pathcol,featcol,breaks,pct,_hsv_mean,axis=2)

        elif aggregate==False:
            featdf = pd.DataFrame(index=pathcol.index,columns=range(10))
            return _iterextract(pathcol,featdf,breaks,pct,_hsv_10bin,axis=2)

def _saturation(pathcol,aggregate):
    """Returns either average saturation or 10-bin distribution"""

    if isinstance(pathcol,basestring):
        if aggregate==True:
            return _hsv_mean(pathcol,axis=1)
        elif aggregate==False:
            return _hsv_10bin(pathcol,axis=1)

    elif isinstance(pathcol,pd.Series):
        breaks,pct = _progressBar(pathcol)
        if aggregate==True:
            featcol = pd.Series(index=pathcol.index)
            return _iterextract(pathcol,featcol,breaks,pct,_hsv_mean,axis=1)

        elif aggregate==False:
            featdf = pd.DataFrame(index=pathcol.index,columns=range(10))
            return _iterextract(pathcol,featdf,breaks,pct,_hsv_10bin,axis=1)

def _hsv_mean(imgpath,axis):
    img = _imgfilter(imgpath)
    return np.mean(img[:,:,axis])

def _hsv_10bin(imgpath,axis):
    img = _imgfilter(imgpath)
    return np.histogram(img[:,:,axis],bins=10)[0]

def _hue(pathcol,aggregate):
    """Returns either huepeak or 8-bin perceptual hue distribution"""

    if isinstance(pathcol,basestring):
        if aggregate==True:
            return _huepeak(pathcol)
        elif aggregate==False:
            return _hue_8bin(pathcol)

    elif isinstance(pathcol,pd.Series):
        breaks,pct = _progressBar(pathcol)
        if aggregate==True:
            featcol = pd.Series(index=pathcol.index)
            return _iterextract(pathcol,featcol,breaks,pct,_huepeak)

        elif aggregate==False:
            featdf = pd.DataFrame(index=pathcol.index,
                                  columns=["red","orange","yellow","green",
                                           "cyan","blue","purple","magenta",
                                           "highred"])

            featdf = _iterextract(pathcol,featdf,breaks,pct,_hue_8bin)
            featdf['red'] = featdf.red + featdf.highred
            del featdf['highred']

            return featdf

def _huepeak(imgpath):
    img = _imgfilter(imgpath)
    imghue = img[:,:,0]
    imghue = imghue.flatten()

    # Silverman's rule of thumb Gaussian KDE bandwidth selection
    n = len(imghue)
    thetahat = np.std(imghue)
    h = 1.06 * thetahat * n**(-1/float(5)) # float() or python uses int division
    if h==0: # some are zero bc std of hue is zero
        h = 1E-6 # h != 0 so kde will work

    X = imghue[:,np.newaxis]
    kde = KernelDensity(kernel='gaussian',bandwidth=h).fit(X)

    Xeval = np.linspace(0,1,360)[:,np.newaxis] # huepeak as degrees: 0-360
    logDensity = kde.score_samples(Xeval)
    return np.argmax(logDensity)

def _hue_8bin(imgpath):
    # nonuniform width hue bins
    huebreaks = [0.0,
                 0.05555555555555555,
                 0.1388888888888889,
                 0.19444444444444445,
                 0.4444444444444444,
                 0.5555555555555556,
                 0.7222222222222222,
                 0.7916666666666666,
                 0.9166666666666666,
                 1.0]

    img = _imgfilter(imgpath)
    return np.histogram(img[:,:,0],bins=huebreaks)[0]

#------------------------------------------------------------------------------

def _entropy_brightness(pathcol):
    """Returns brightness entropy"""

    if isinstance(pathcol,basestring):
        return _entropy(pathcol,axis=2)

    elif isinstance(pathcol,pd.Series):
        featcol = pd.Series(index=pathcol.index)
        breaks,pct = _progressBar(pathcol)
        return _iterextract(pathcol,featcol,breaks,pct,_entropy,axis=2)

def _entropy(imgpath,axis=None):
    img = _imgfilter(imgpath)
    return entropy(np.histogram(img[:,:,axis],bins=10)[0])

def _std_brightness(pathcol):
    """Returns standard deviation of brightness"""

    if isinstance(pathcol,basestring):
        return _std(pathcol,axis=2)

    elif isinstance(pathcol,pd.Series):
        featcol = pd.Series(index=pathcol.index)
        breaks,pct = _progressBar(pathcol)
        return _iterextract(pathcol,featcol,breaks,pct,_std,axis=2)

def _std(imgpath,axis=None):
    img = _imgfilter(imgpath)
    return np.std(img[:,:,axis])

#------------------------------------------------------------------------------

def _glcm(pathcol,prop):
    """Returns gray-level co-occurrence matrix property"""
    return None

#------------------------------------------------------------------------------

def _neural(pathcol,tags):
    """Returns Inception v3 tags or penultimate vector"""
    return None
