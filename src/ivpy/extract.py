from skimage.io import imread
from skimage import color
import numpy as np
from scipy.stats import entropy
#from skimage.feature import greycomatrix, greycoprops
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import Model
from keras.applications import InceptionV3
import pandas as pd

from .data import _typecheck,_colfilter

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
        return _entropy(pathcol)
    elif feature=='std':
        return _std(pathcol)
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

def _progressBar(pathcol):
    n = len(pathcol)
    breaks = [int(n * item) for item in np.arange(.1,1,.1)]
    pct = [str(int(item*100))+"%" for item in np.arange(.1,1,.1)]

    return breaks,pct

#------------------------------------------------------------------------------

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
            counter=-1
            for i in pathcol.index:
                counter+=1
                if counter in breaks:
                    print pct[breaks.index(counter)],
                imgpath = pathcol.loc[i]
                featcol.loc[i] = _hsv_mean(imgpath,axis=2)
            return featcol
        elif aggregate==False:
            featdf = pd.DataFrame(index=pathcol.index,columns=range(10))
            counter=-1
            for i in pathcol.index:
                counter+=1
                if counter in breaks:
                    print pct[breaks.index(counter)],
                imgpath = pathcol.loc[i]
                featdf.loc[i] = _hsv_10bin(imgpath,axis=2)
            return featdf

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
            counter=-1
            for i in pathcol.index:
                counter+=1
                if counter in breaks:
                    print pct[breaks.index(counter)],
                imgpath = pathcol.loc[i]
                featcol.loc[i] = _hsv_mean(imgpath,axis=1)
            return featcol
        elif aggregate==False:
            featdf = pd.DataFrame(index=pathcol.index,columns=range(10))
            counter=-1
            for i in pathcol.index:
                counter+=1
                if counter in breaks:
                    print pct[breaks.index(counter)],
                imgpath = pathcol.loc[i]
                featdf.loc[i] = _hsv_10bin(imgpath,axis=1)
            return featdf

def _hsv_mean(imgpath,axis=None):
    img = imread(imgpath)
    img = color.rgb2hsv(img)
    return np.mean(img[:,:,axis])

def _hsv_10bin(imgpath,axis=None):
    img = imread(imgpath)
    img = color.rgb2hsv(img)
    return np.histogram(img[:,:,axis],bins=10)[0]

def _hue(pathcol,aggregate):
    """Returns either huepeak or 8-bin perceptual hue distribution"""
    return None

#------------------------------------------------------------------------------

def _entropy(pathcol):
    """Returns brightness entropy"""
    return None

def _std(pathcol):
    """Returns standard deviation of brightness"""
    return None

#------------------------------------------------------------------------------

def _glcm(pathcol,prop):
    """Returns gray-level co-occurrence matrix property"""
    return None

#------------------------------------------------------------------------------

def _neural(pathcol,tags):
    """Returns Inception v3 tags or penultimate vector"""
    return None
