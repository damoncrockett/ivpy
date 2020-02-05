import pandas as pd
import numpy as np
from PIL import Image
from six import string_types

from skimage.io import imread
from skimage import color
from skimage import img_as_ubyte
from skimage.transform import resize
from scipy.stats import entropy
from scipy.stats import percentileofscore as pct
from skimage.feature import greycomatrix, greycoprops
from sklearn.neighbors import KernelDensity

from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import Model
from keras.applications import ResNet50

from .data import _typecheck,_pathfilter
from .plottools import _progressBar

#------------------------------------------------------------------------------

def extract(feature,pathcol=None,aggregate=True,scale=True,verbose=False):
    _typecheck(**locals())
    pathcol = _pathfilter(pathcol)

    if feature=='brightness':
        return _brightness(pathcol,aggregate,scale,verbose)
    elif feature=='saturation':
        return _saturation(pathcol,aggregate,scale,verbose)
    elif feature=='hue':
        return _hue(pathcol,aggregate,scale,verbose)
    elif feature=='entropy':
        return _entropy_brightness(pathcol,scale,verbose)
    elif feature=='std':
        return _std_brightness(pathcol,scale,verbose)
    elif feature=='contrast':
        return _glcm(pathcol,scale,verbose,prop='contrast')
    elif feature=='dissimilarity':
        return _glcm(pathcol,scale,verbose,prop='dissimilarity')
    elif feature=='homogeneity':
        return _glcm(pathcol,scale,verbose,prop='homogeneity')
    elif feature=='ASM':
        return _glcm(pathcol,scale,verbose,prop='ASM')
    elif feature=='energy':
        return _glcm(pathcol,scale,verbose,prop='energy')
    elif feature=='correlation':
        return _glcm(pathcol,scale,verbose,prop='correlation')
    elif feature=='neural':
        return _neural(pathcol,verbose)
    elif feature=='dmax':
        return _dmax(pathcol,scale,verbose)

#------------------------------------------------------------------------------

def _iterextract(pathcol,outstructure,breaks,pct,func,verbose=False,**kwargs):
    n = len(pathcol)
    counter=-1
    for i in pathcol.index:
        counter+=1
        imgpath = pathcol.loc[i]

        if verbose==False:
            if counter in breaks:
                pctstring = pct[breaks.index(counter)]
                print(pctstring,end=" ")
        elif verbose==True:
            print(str(counter),'of',str(n),imgpath)
        try:
            outstructure.loc[i] = func(imgpath,**kwargs)
        except Exception as e:
            print(e)
            outstructure.loc[i] = None
    return outstructure

#------------------------------------------------------------------------------

def _imgprocess(imgpath,scale):
    """Returns (possibly scaled) HSV array"""
    img = imread(imgpath)
    if scale==True:
        img = _scale(img)
    if len(img.shape)==3:
        return color.rgb2hsv(img)
    elif len(img.shape)==2:
        return color.rgb2hsv(color.gray2rgb(img))

def _scale(img):
    """Scales images to 256px max side for feature extraction. This
       function is distinct from resize() in data.py and does not save any
       images to file."""

    h,w = img.shape[0],img.shape[1] # note weird order
    if any([h>256,w>256]):
        if h>w:
            ratio = 256 / float(h)
            newh = 256
            neww = int( w * ratio )
        elif w>h:
            ratio = 256 / float(w)
            neww = 256
            newh = int( h * ratio )
        elif w==h:
            newh = 256
            neww = 256
        return resize(img,(newh,neww))
    else:
        return img

def _featscale(ser):
    init_min = min(ser[ser.notnull()])
    ser_adj = ser.map(lambda x:x-init_min)
    adj_max = max(ser[ser.notnull()]) - init_min

    try:
        ser_adj = ser_adj.map(lambda x:x/adj_max)
    except:
        raise ValueError("Min and max are the same, causing division by zero")

    return ser_adj

def _pct(ser):
    ser_notnull = ser[ser.notnull()]
    ser = ser.map(lambda x: pct(ser_notnull,x)/100) # pct returns 0-100

    return ser

def norm(arr, normtype='featscale'):
    _typecheck(**locals())

    if isinstance(arr,pd.DataFrame):
        if normtype=='featscale':
            return arr.apply(_featscale, axis=0)
        elif normtype=='pct':
            return arr.apply(_pct, axis=0)

    elif isinstance(arr,pd.Series):
        if normtype=='featscale':
            return _featscale(arr)
        elif normtype=='pct':
            return _pct(arr)

    elif not isinstance(arr,(pd.DataFrame,pd.Series)):
        raise TypeError("""Data must be either a pandas DataFrame or Series""")

#------------------------------------------------------------------------------

def _brightness(pathcol,aggregate,scale,verbose):
    """Returns either average brightness or 10-bin distribution"""

    if isinstance(pathcol,string_types):
        if aggregate==True:
            return _hsv_mean(pathcol,scale,axis=2)
        elif aggregate==False:
            return _hsv_10bin(pathcol,scale,axis=2)

    elif isinstance(pathcol,pd.Series):
        breaks,pct = _progressBar(pathcol)
        if aggregate==True:
            featcol = pd.Series(index=pathcol.index)
            return _iterextract(pathcol,featcol,breaks,pct,_hsv_mean,verbose,
                                scale=scale,axis=2)

        elif aggregate==False:
            featdf = pd.DataFrame(index=pathcol.index,columns=range(10))
            return _iterextract(pathcol,featdf,breaks,pct,_hsv_10bin,verbose,
                                scale=scale,axis=2)

def _saturation(pathcol,aggregate,scale,verbose):
    """Returns either average saturation or 10-bin distribution"""

    if isinstance(pathcol,string_types):
        if aggregate==True:
            return _hsv_mean(pathcol,scale,axis=1)
        elif aggregate==False:
            return _hsv_10bin(pathcol,scale,axis=1)

    elif isinstance(pathcol,pd.Series):
        breaks,pct = _progressBar(pathcol)
        if aggregate==True:
            featcol = pd.Series(index=pathcol.index)
            return _iterextract(pathcol,featcol,breaks,pct,_hsv_mean,verbose,
                                scale=scale,axis=1)

        elif aggregate==False:
            featdf = pd.DataFrame(index=pathcol.index,columns=range(10))
            return _iterextract(pathcol,featdf,breaks,pct,_hsv_10bin,verbose,
                                scale=scale,axis=1)

def _hsv_mean(imgpath,scale,axis):
    img = _imgprocess(imgpath,scale)
    return np.mean(img[:,:,axis])

def _hsv_10bin(imgpath,scale,axis):
    img = _imgprocess(imgpath,scale)
    binedges = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] # fixed bin edges
    return np.histogram(img[:,:,axis],bins=binedges)[0]

def _hue(pathcol,aggregate,scale,verbose):
    """Returns either huepeak or 8-bin perceptual hue distribution"""

    if isinstance(pathcol,string_types):
        if aggregate==True:
            return _huepeak(pathcol,scale)
        elif aggregate==False:
            return _hue_8bin(pathcol,scale)

    elif isinstance(pathcol,pd.Series):
        breaks,pct = _progressBar(pathcol)
        if aggregate==True:
            featcol = pd.Series(index=pathcol.index)
            return _iterextract(pathcol,featcol,breaks,pct,_huepeak,verbose,
                                scale=scale)

        elif aggregate==False:
            featdf = pd.DataFrame(index=pathcol.index,
                                  columns=["red","orange","yellow","green",
                                           "cyan","blue","purple","magenta",
                                           "highred"])

            featdf = _iterextract(pathcol,featdf,breaks,pct,_hue_8bin,verbose,
                                  scale=scale)
            featdf['red'] = featdf.red + featdf.highred
            del featdf['highred']

            return featdf

def _huepeak(imgpath,scale):
    img = _imgprocess(imgpath,scale)
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

def _hue_8bin(imgpath,scale):
    # nonuniform width hue bins
    binedges = [0.0,
                 0.05555555555555555,
                 0.1388888888888889,
                 0.19444444444444445,
                 0.4444444444444444,
                 0.5555555555555556,
                 0.7222222222222222,
                 0.7916666666666666,
                 0.9166666666666666,
                 1.0]

    img = _imgprocess(imgpath,scale)
    return np.histogram(img[:,:,0],bins=binedges)[0]

#------------------------------------------------------------------------------

def _entropy_brightness(pathcol,scale,verbose):
    """Returns brightness entropy"""

    if isinstance(pathcol,string_types):
        return _entropy(pathcol,scale,axis=2)

    elif isinstance(pathcol,pd.Series):
        featcol = pd.Series(index=pathcol.index)
        breaks,pct = _progressBar(pathcol)
        return _iterextract(pathcol,featcol,breaks,pct,_entropy,verbose,
                            scale=scale,axis=2)

def _entropy(imgpath,scale,axis=None):
    img = _imgprocess(imgpath,scale)
    return entropy(np.histogram(img[:,:,axis],bins=10)[0])

def _std_brightness(pathcol,scale,verbose):
    """Returns standard deviation of brightness"""

    if isinstance(pathcol,string_types):
        return _std(pathcol,scale,axis=2)

    elif isinstance(pathcol,pd.Series):
        featcol = pd.Series(index=pathcol.index)
        breaks,pct = _progressBar(pathcol)
        return _iterextract(pathcol,featcol,breaks,pct,_std,verbose,
                            scale=scale,axis=2)

def _std(imgpath,scale,axis=None):
    img = _imgprocess(imgpath,scale)
    return np.std(img[:,:,axis])

#------------------------------------------------------------------------------

def _glcm(pathcol,scale,verbose,prop):
    """Returns gray-level co-occurrence matrix property"""

    if isinstance(pathcol,string_types):
       return _greycoprops(pathcol,scale,prop)

    elif isinstance(pathcol,pd.Series):
       featcol = pd.Series(index=pathcol.index)
       breaks,pct = _progressBar(pathcol)
       return _iterextract(pathcol,featcol,breaks,pct,_greycoprops,verbose,
                           scale=scale,prop=prop)

def _greycoprops(imgpath,scale,prop):
    """Note that _imgprocess is not used; here we need gray integer img"""
    img = imread(imgpath)
    if scale==True:
        img = _scale(img)
    imgray = color.rgb2gray(img)
    imgray = img_as_ubyte(imgray)
    glcmat = greycomatrix(imgray,[1],[0],levels=256,symmetric=True,normed=True)
    return greycoprops(glcmat, prop)[0][0]

#------------------------------------------------------------------------------

def _neural(pathcol,verbose):
    """Returns ResNet50 penultimate vector"""

    preprocess = imagenet_utils.preprocess_input
    base_model = ResNet50(weights='imagenet')
    penlayer = base_model.layers[-2].name # unpredictable
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer(penlayer).output)

    if isinstance(pathcol,string_types):
        return _featvector(pathcol,preprocess,model)

    elif isinstance(pathcol,pd.Series):
        breaks,pct = _progressBar(pathcol)
        featdf = pd.DataFrame(index=pathcol.index,columns=range(2048))
        featdf = _iterextract(pathcol,featdf,breaks,pct,_featvector,verbose,
                              preprocess=preprocess,
                              model=model)
        return featdf

def _featvector(imgpath,preprocess,model):
    inputShape = (224,224)
    image = load_img(imgpath,target_size=inputShape)
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    image = preprocess(image)
    return model.predict(image)[0]

#------------------------------------------------------------------------------

def _dmax(pathcol,scale,verbose):
    """Returns brightness at 'dmax'; used for measuring photo fading"""

    if isinstance(pathcol,string_types):
        return _dmax_convolution(pathcol,scale)

    elif isinstance(pathcol,pd.Series):
        breaks,pct = _progressBar(pathcol)
        featdf = pd.DataFrame(index=pathcol.index,columns=['dmax','dmin','contrast'])

        return _iterextract(pathcol,featdf,breaks,pct,_dmax_convolution,verbose,
                            scale=scale)

def _dmax_convolution(imgpath,scale):
    img = color.rgb2gray(imread(imgpath))
    if scale==True:
        img = _scale(img)

    nrows,ncols = img.shape[0],img.shape[1]

    valdict = {}

    for i in range(nrows):
        for j in range(ncols):
            pixel = (i,j)
            neighborhood = neighbors(pixel)
            neighborhood = [item for item in neighborhood if all([item[0]>-1,
                                                                  item[1]>-1,
                                                                  item[0]<nrows,
                                                                  item[1]<ncols])]
            vals = [img[item[0]][item[1]] for item in neighborhood]
            valdict[pixel] = np.mean(vals)

    minpx = min(valdict,key=valdict.get)
    maxpx = max(valdict,key=valdict.get)
    dmin = valdict[maxpx] # bc silver density inverts brightness
    dmax = valdict[minpx] # bc silver density inverts brightness

    return dmax,dmin,dmin-dmax # can modify to return other stuff if necessary

def neighbors(pixel):
    i = pixel[0]
    j = pixel[1]

    neighborhood = [
        (i,j),
        (i-1,j),
        (i-2,j),
        (i-3,j),
        (i+1,j),
        (i+2,j),
        (i+3,j),
        (i,j-1),
        (i,j-2),
        (i,j-3),
        (i,j+1),
        (i,j+2),
        (i,j+3),
        (i-1,j-1),
        (i-2,j-1),
        (i-3,j-1),
        (i+1,j-1),
        (i+2,j-1),
        (i+3,j-1),
        (i-1,j+1),
        (i-2,j+1),
        (i-3,j+1),
        (i+1,j+1),
        (i+2,j+1),
        (i+3,j+1),
        (i+1,j-2),
        (i+2,j-2),
        (i-1,j-2),
        (i-2,j-2),
        (i+1,j+2),
        (i+2,j+2),
        (i-1,j+2),
        (i-2,j+2),
        (i-1,j-3),
        (i+1,j-3),
        (i-1,j+3),
        (i+1,j+3),
    ]

    return neighborhood
