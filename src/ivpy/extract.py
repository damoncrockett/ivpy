from skimage.io import imread
from skimage import color
import numpy as np
from scipy.stats import entropy
from skimage.feature import greycomatrix, greycoprops
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import Model
from keras.applications import InceptionV3

from .data import _typecheck,_colfilter

#------------------------------------------------------------------------------

def extract(pathcol=None,feature=None,aggregate=True):
    _typecheck(**locals())
    if pathcol==None:
        raise ValueError("Must supply 'pathcol'")
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

def _brightness(pathcol,aggregate):
    """Returns either average brightness or 10-bin distribution"""

    n = len(pathcol)
    breaks = [int(n * item) for item in np.arange(.1,1,.1)]
    pct = [str(int(item*100))+"%" for item in np.arange(.1,1,.1)]

    featcol = []
    for i in pathcol.index:
        if i in breaks:
            idx = breaks.index(i)
            print pct[idx],
        img = imread(pathcol.loc[i])
        img = color.rgb2hsv(img)
        featcol.append(np.mean(img[:,:,2]))

    return featcol

def _brightness_mean(img):
    return None

def _saturation(pathcol,aggregate):
    """Returns either average saturation or 10-bin distribution"""
    return None

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
