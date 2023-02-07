import os
import pandas as pd
import numpy as np
from PIL import Image
from six import string_types
from math import ceil

from skimage.io import imread,imsave
from skimage.filters import gaussian
from skimage import color
from skimage.draw import disk
from skimage.util import img_as_ubyte
from skimage.transform import resize
from scipy.stats import entropy
from scipy.stats import percentileofscore as pct
from skimage.feature import greycomatrix, greycoprops
from sklearn.neighbors import KernelDensity

try:
    import tifffile as tiff
    import cv2
except:
    print("for roughness extraction, must install 'opencv-python' and 'tifffile' modules")

try:
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.applications.resnet50 import ResNet50
except:
    print("for neural feature extraction, must install 'tensorflow' module")

from .data import _typecheck,_pathfilter
from .plottools import _progressBar

#------------------------------------------------------------------------------

def extract(feature,
            pathcol=None,aggregate=True,scale=True,verbose=False,**kwargs):
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
    elif feature=='condition':
        return _condition(pathcol,verbose,**kwargs)
    elif feature=='roughness':
        return _roughness(pathcol,verbose,**kwargs)

#------------------------------------------------------------------------------

def _iterextract(pathcol,cols,breaks,pct,func,verbose=False,**kwargs):

    n = len(pathcol)
    ncols = len(cols)
    dictlist = []
    counter=0
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
            vals = func(imgpath,**kwargs)
            if ncols > 1:
                d = dict(zip(cols,vals))
            else:
                d = vals

        except Exception as e:
            print(e)
            vals = [None] * ncols
            if ncols > 1:
                d = dict(zip(cols,vals))
            else:
                d = vals

        dictlist.append(d)

    if ncols > 1:
        outstructure = pd.DataFrame.from_dict(dictlist)
        outstructure.index = pathcol.index
    else:
        outstructure = pd.Series(dictlist,index=pathcol.index)

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

def _scale(img,side):
    """Scales images to  'side' pixels max side for feature extraction. This
       function is distinct from resize() in data.py and does not save any
       images to file."""

    h,w = img.shape[0],img.shape[1] # note weird order
    if any([h>side,w>side]):
        if h>w:
            ratio = side / float(h)
            newh = side
            neww = int( w * ratio )
        elif w>h:
            ratio = side / float(w)
            neww = side
            newh = int( h * ratio )
        elif w==h:
            newh = side
            neww = side
        return resize(img,(newh,neww))
    else:
        return img

def _featscale(ser, input_range, output_range):
    """Scales a series to a specified range (default 0-1)"""
    a = output_range[0]
    b = output_range[1]

    if any([a>b, a==b]):
        raise ValueError("'output_range' must be a list or tuple of (min,max), with max > min")

    if input_range is None:
        input_range = (min(ser[ser.notnull()]),max(ser[ser.notnull()]))
    elif any([input_range[0]>input_range[1], input_range[0]==input_range[1]]):
        raise ValueError("'input_range' must be a list or tuple of (min,max), with max > min")

    ser_adj = ser.map(lambda x:x-input_range[0])
    adj_max = input_range[1] - input_range[0]

    try:
        ser_adj = ser_adj.map(lambda x: (x/adj_max)*(b-a)+a)
    except:
        # above will fail if all values are the same; if so, return zeros
        ser_adj = np.zeros(len(ser))

    return ser_adj

def _pct(ser):
    ser_notnull = ser[ser.notnull()]
    ser = ser.map(lambda x: pct(ser_notnull,x)/100) # pct returns 0-100

    return ser

def norm(arr, normtype='featscale', input_range=None, output_range=(0,1)):
    _typecheck(**locals())

    if isinstance(arr,pd.DataFrame):
        if normtype=='featscale':
            return arr.apply(_featscale, axis=0, input_range=input_range, output_range=output_range)
        elif normtype=='pct':
            if output_range != (0,1):
                print("""Warning: 'output_range' is ignored when normtype='pct'""")
            if input_range is not None:
                print("""Warning: 'input_range' is ignored when normtype='pct'""")
            return arr.apply(_pct, axis=0)

    elif isinstance(arr,pd.Series):
        if normtype=='featscale':
            return _featscale(arr, input_range=input_range, output_range=output_range)
        elif normtype=='pct':
            if output_range != (0,1):
                print("""Warning: 'output_range' is ignored when normtype='pct'""")
            if input_range is not None:
                print("""Warning: 'input_range' is ignored when normtype='pct'""")
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
            cols = [0]
            return _iterextract(pathcol,cols,breaks,pct,_hsv_mean,verbose,
                                scale=scale,axis=2)

        elif aggregate==False:
            cols = list(range(10))
            return _iterextract(pathcol,cols,breaks,pct,_hsv_10bin,verbose,
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
            cols = [0]
            return _iterextract(pathcol,cols,breaks,pct,_hsv_mean,verbose,
                                scale=scale,axis=1)

        elif aggregate==False:
            cols = list(range(10))
            return _iterextract(pathcol,cols,breaks,pct,_hsv_10bin,verbose,
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
            cols = [0]
            return _iterextract(pathcol,cols,breaks,pct,_huepeak,verbose,
                                scale=scale)

        elif aggregate==False:
            cols = ["red","orange","yellow","green","cyan","blue","purple",
                    "magenta","highred"]

            featdf = _iterextract(pathcol,cols,breaks,pct,_hue_8bin,verbose,
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
        cols = [0]
        breaks,pct = _progressBar(pathcol)
        return _iterextract(pathcol,cols,breaks,pct,_entropy,verbose,
                            scale=scale,axis=2)

def _entropy(imgpath,scale,axis=None):
    img = _imgprocess(imgpath,scale)
    return entropy(np.histogram(img[:,:,axis],bins=10)[0])

def _std_brightness(pathcol,scale,verbose):
    """Returns standard deviation of brightness"""

    if isinstance(pathcol,string_types):
        return _std(pathcol,scale,axis=2)

    elif isinstance(pathcol,pd.Series):
        cols = [0]
        breaks,pct = _progressBar(pathcol)
        return _iterextract(pathcol,cols,breaks,pct,_std,verbose,
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
       cols = [0]
       breaks,pct = _progressBar(pathcol)
       return _iterextract(pathcol,cols,breaks,pct,_greycoprops,verbose,
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

    # need pooling, otherwise the model returns a 7x7 feature map
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    if isinstance(pathcol,string_types):
        return _featvector(pathcol,model)

    elif isinstance(pathcol,pd.Series):
        breaks,pct = _progressBar(pathcol)
        cols = list(range(2048))
        featdf = _iterextract(pathcol,cols,breaks,pct,_featvector,verbose,
                              model=model)
        return featdf

def _featvector(imgpath,model):
    img = image.load_img(imgpath, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return model.predict(x)[0]

#------------------------------------------------------------------------------

def _condition(pathcol,verbose,k=1,savemap=True,side=512):
    """Returns median lightness of the k% darkest pixels, and the
       median saturation of the k% lightest pixels. If savemap is True,
       also returns the path to the saved map."""

    if isinstance(pathcol,string_types):
        return _ktop(pathcol,k,savemap,side)

    elif isinstance(pathcol,pd.Series):
        breaks,pct = _progressBar(pathcol)
        cols = ['lowtone','saturation']
        if savemap:
            cols.append('mappath')

        return _iterextract(pathcol,cols,breaks,pct,_ktop,
                            verbose,k=k,savemap=savemap,side=side)

def _ktop(imgpath,k,savemap,side):
    
    img = imread(imgpath)
    img_hsv = color.rgb2hsv(img)

    img = img_as_ubyte(_scale(img,side))
    img_hsv = _scale(img_hsv,side)

    npixels = int(img_hsv.shape[0] * img_hsv.shape[1] * k/100)
    valimage = img_hsv[:,:,2]

    dark_threshold = sorted(valimage.flatten())[npixels]
    dark_pixels = valimage < dark_threshold
    
    light_threshold = sorted(valimage.flatten())[-npixels]
    light_pixels = valimage > light_threshold

    lowtone = np.median(valimage[dark_pixels])
    saturation = np.median(img_hsv[:,:,1][light_pixels])

    if savemap is not False:

        img[dark_pixels] = [255,0,255] # magenta
        img[light_pixels] = [0,255,255] # cyan 
        
        imgdir = os.path.dirname(imgpath)
        imgname = os.path.basename(imgpath)

        if isinstance(savemap,string_types):
            mapdir = os.path.join(imgdir,'_maps_'+savemap)
        else:
            mapdir = os.path.join(imgdir,'_maps')
        if not os.path.exists(mapdir):
            os.makedirs(mapdir)
        mappath = os.path.join(mapdir,imgname[:-4]+".png")
        imsave(mappath,img)

        return lowtone,saturation,mappath

    return lowtone,saturation

# def _condition(pathcol,scale,verbose,sigma=6,savemap=False,side=256):
#     """Returns brightness at 'dmax' (the darkest spot) and saturation at 'dmin';
#        used for measuring photo fading and yellowing"""

#     if isinstance(pathcol,string_types):
#         return _condition_convolution(pathcol,scale,sigma,savemap,side)

#     elif isinstance(pathcol,pd.Series):
#         breaks,pct = _progressBar(pathcol)
#         cols = ['contrast','satdmin']
#         if savemap:
#             cols.append('mappath')

#         return _iterextract(pathcol,cols,breaks,pct,_condition_convolution,
#                             verbose,scale=scale,sigma=sigma,savemap=savemap,side=side)

# def _condition_convolution(imgpath,scale,sigma,savemap,side):
    
#     img = imread(imgpath)
#     img_hsv = color.rgb2hsv(img)

#     if scale==True:
#         img_hsv = _scale(img_hsv,side)

#         if savemap:
#             img = _scale(img,side)

#     imgblur_hsv = gaussian(img_hsv,sigma=sigma,channel_axis=2)
#     contrast = np.max(imgblur_hsv[:,:,2]) - np.min(imgblur_hsv[:,:,2])
#     dmin = np.argmax(imgblur_hsv[:,:,2])
#     dmax = np.argmin(imgblur_hsv[:,:,2])
#     satdmin = imgblur_hsv[:,:,1].flatten()[dmin]

#     if savemap is not False:
#         imgblur = gaussian(img,sigma=sigma,channel_axis=2)
#         dmin = np.unravel_index(dmin,imgblur.shape[:2])
#         dmax = np.unravel_index(dmax,imgblur.shape[:2])
        
#         # draw black circle around dmin in imgblur
#         rr,cc = disk((dmin[0],dmin[1]),radius=10,shape=imgblur.shape[:2])
#         imgblur[rr,cc] = [0,0,0]

#         # draw white circle around dmax in imgblur
#         rr,cc = disk((dmax[0],dmax[1]),radius=10,shape=imgblur.shape[:2])
#         imgblur[rr,cc] = [1,1,1]

#         imgdir = os.path.dirname(imgpath)
#         imgname = os.path.basename(imgpath)

#         if isinstance(savemap,string_types):
#             mapdir = os.path.join(imgdir,'_maps_'+savemap)
#         else:
#             mapdir = os.path.join(imgdir,'_maps')
#         if not os.path.exists(mapdir):
#             os.makedirs(mapdir)
#         mappath = os.path.join(mapdir,imgname[:-4]+".png")
#         imsave(mappath,imgblur)

#         return contrast,satdmin,mappath

#     return contrast,satdmin

#------------------------------------------------------------------------------

def _roughness(pathcol,verbose,
               N=768,gain=250,low_pass_sigma=501,high_pass_sigma=21,low_pass_apply='divide'):

    """Returns standard deviation of pixel brightness after some pre-processing
    and bandpass filtering. Only works with TIFF files currently. Intended for
    use with raking light microscopy images. Used for approximating Sq as
    defined in surface metrology (root mean square height). The roughness values
    computed here have a ~0.9 Pearson correlation with Sq.
    """

    if isinstance(pathcol,string_types):
        return _bandpass_std(pathcol,N,gain,low_pass_sigma,high_pass_sigma,low_pass_apply)

    elif isinstance(pathcol,pd.Series):
        cols = [0]
        breaks,pct = _progressBar(pathcol)
        return _iterextract(pathcol,cols,breaks,pct,_bandpass_std,verbose,
                            N=N,gain=gain,low_pass_sigma=low_pass_sigma,
                            high_pass_sigma=high_pass_sigma,low_pass_apply=low_pass_apply)

def _crop_array(array,N):

    h, w = array.shape

    left  = w/2 - N/2
    upper = h/2 - N/2
    right = w/2 + N/2
    lower = h/2 + N/2

    return array[int(upper):int(lower),int(left):int(right)]

def _read_process_image(imgpath,gain,N,low_pass_sigma,high_pass_sigma,low_pass_apply):

    try:
        tif_array = tiff.imread(imgpath)
    except:
        tif_array = np.asarray(Image.open(imgpath)) # used if imagecodecs is missing above

    if tif_array.shape[2] == 3:
        tif_array = color.rgb2gray(tif_array)
    elif tif_array.shape[2] == 4:
        tif_array = color.rgb2gray(tif_array[:,:,:3]) # some have alpha layer for some reason

    #tif_array = img_as_ubyte(tif_array)

    # if tif_array.shape[1] < 2448:
    #    N = 1024

    # define sigma for Gaussian blurs to low pass and high pass the data
    #low_pass_sigma = 151
    #low_pass_sigma = 201
    #high_pass_sigma = 5

    # Crop array to extract middle NxN portion of image
    # Adding extra to allow for smooth filtering
    tif_array = _crop_array(tif_array,N+low_pass_sigma)

    # Normalize by total intensity
    tif_array = (gain*tif_array)/(np.sum(tif_array))*(N**2)

    #Subtract or divide low-pass to remove low order waviness
    #Use divide for NN, will be darker

    if low_pass_apply == 'divide':
        tif_array = tif_array / cv2.GaussianBlur(tif_array,(low_pass_sigma,low_pass_sigma),0)
    elif low_pass_apply == 'subtract':
        tif_array = tif_array - cv2.GaussianBlur(tif_array,(low_pass_sigma,low_pass_sigma),0)

    #High-pass data
    tif_array = cv2.GaussianBlur(tif_array,(high_pass_sigma,high_pass_sigma),0)
    tif_array = _crop_array(tif_array,N)

    return tif_array

def _bandpass_std(imgpath,N,gain,low_pass_sigma,high_pass_sigma,low_pass_apply):

    # Scaling factor used in normalization step (found by trial)
    #gain = 250

    # Extracted image will be NxN = 1024x1024 (middle chunk of the image)
    #N = 1024
    #N = 1365

    img = _read_process_image(imgpath,gain,N,low_pass_sigma,high_pass_sigma,low_pass_apply)

    return np.std(img)
