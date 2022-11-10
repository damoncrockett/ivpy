import os
import pandas as pd
import numpy as np
from PIL import Image
from six import string_types
#import image_slicer
from glob2 import glob
import matplotlib.pyplot as plt

from .data import _pathfilter,_typecheck
from .extract import _read_process_image
from skimage.color import rgb2gray
from skimage.io import imsave
import tifffile as tiff
import warnings
warnings.filterwarnings('ignore')

#-------------------------------------------------------------------------------

def resize(savedir=None,pathcol=None,thumb=256,verbose=False,include_dir=False):
    """Creates thumbnails of images, saves to 'savedir'. Had considered default
       'savedir', but decided user should always supply 'savedir', putting
       the write responsibility on them."""

    if savedir==None:
        raise ValueError("Must supply 'savedir'")
    elif savedir is not None:
        if os.path.isdir(savedir)==True:
            pass
        else:
            try:
                os.mkdir(savedir)
            except:
                raise ValueError("'savedir' must be a valid filepath")

    _typecheck(**locals())
    pathcol = _pathfilter(pathcol)

    if isinstance(pathcol,string_types):
        return _resize(pathcol,savedir,thumb,include_dir)

    elif isinstance(pathcol,pd.Series):
        pathcol_resized = pd.Series(index=pathcol.index)
        n = len(pathcol)
        for j,i in enumerate(pathcol.index):
            impath = pathcol.loc[i]
            if verbose==True:
                print(j+1,'of',n,impath)
            pathcol_resized.loc[i] = _resize(impath,savedir,thumb,include_dir)
        return pathcol_resized

def _resize(impath,savedir,thumb,include_dir):
    try:
        im = Image.open(impath)
        im.thumbnail((thumb,thumb),Image.LANCZOS)
        if include_dir:
            basename = '_'.join(impath.split("/"))
        else:
            basename = os.path.basename(impath)

        savestring = savedir + "/" + basename
        im.save(savestring)
        return savestring
    except:
        return None

#-------------------------------------------------------------------------------

def tifpass(savedir=None,pathcol=None,verbose=False,gain=250,N=1365,
            include_dir=False,low_pass_sigma=201,high_pass_sigma=5,
            low_pass_apply='subtract',plainsave=False):
    """Creates cropped, normalized, bandpassed versions of texturescope TIFFs,
    saves to 'savedir'."""

    if savedir==None:
        raise ValueError("Must supply 'savedir'")
    elif savedir is not None:
        if os.path.isdir(savedir)==True:
            pass
        else:
            try:
                os.mkdir(savedir)
            except:
                raise ValueError("'savedir' must be a valid filepath")

    _typecheck(**locals())
    pathcol = _pathfilter(pathcol)

    if isinstance(pathcol,string_types):
        return _tifpass(pathcol,savedir,gain,N,
                        include_dir,low_pass_sigma,high_pass_sigma,
                        low_pass_apply,plainsave)

    elif isinstance(pathcol,pd.Series):
        pathcol_tifpassed = pd.Series(index=pathcol.index)
        n = len(pathcol)
        for j,i in enumerate(pathcol.index):
            impath = pathcol.loc[i]
            if verbose==True:
                print(j+1,'of',n,impath)
            pathcol_tifpassed.loc[i] = _tifpass(impath,savedir,gain,N,include_dir,
                                                low_pass_sigma,high_pass_sigma,
                                                low_pass_apply,plainsave)
        return pathcol_tifpassed

def _tifpass(impath,savedir,gain,N,include_dir,low_pass_sigma,high_pass_sigma,low_pass_apply,plainsave):
    try:
        img = _read_process_image(impath,gain,N,low_pass_sigma,high_pass_sigma,low_pass_apply)

        if include_dir:
            basename = '_'.join(impath.split("/"))
        else:
            basename = os.path.basename(impath)
        savestring = savedir + "/" + basename

        if plainsave:
            im = Image.fromarray(img)
            im.save(savestring)
        else:
            vmin= -3*np.std(img)
            vmax=  3*np.std(img)
            savestring = savestring[:-3] + 'jpg'
            _ = plt.imsave(savestring, img, vmin=vmin, vmax=vmax, cmap='gray')

        return savestring
    except Exception as e:
        print(e)
        return None

#-------------------------------------------------------------------------------

def tifprocess(savedir=None,pathcol=None,verbose=False,N=1365,include_dir=False):
    """Creates cropped, exposure-corrected versions of texturescope TIFFs,
    saves to 'savedir'."""

    if savedir==None:
        raise ValueError("Must supply 'savedir'")
    elif savedir is not None:
        if os.path.isdir(savedir)==True:
            pass
        else:
            try:
                os.mkdir(savedir)
            except:
                raise ValueError("'savedir' must be a valid filepath")

    _typecheck(**locals())
    pathcol = _pathfilter(pathcol)

    if isinstance(pathcol,string_types):
        return _tifprocess(pathcol,savedir,N,include_dir)

    elif isinstance(pathcol,pd.Series):
        pathcol_tifprocessed = []
        n = len(pathcol)
        for j,i in enumerate(pathcol.index):
            impath = pathcol.loc[i]
            if verbose==True:
                print(j+1,'of',n,impath)
            pathcol_tifprocessed.append(_tifprocess(impath,savedir,N,include_dir))
        return pd.Series(pathcol_tifprocessed,index=pathcol.index)

def _tifprocess(impath,savedir,N,include_dir):
    try:
        img = rgb2gray(tiff.imread(impath))
        if img.shape[1] < 2448:
            N = 1024
        img = _crop_array(img,N)
        img = _exposure_correction(img)

        if include_dir:
            basename = '_'.join(impath.split("/"))
        else:
            basename = os.path.basename(impath)[:-3] + 'jpg'

        savestring = os.path.join(savedir,basename)
        _ = imsave(savestring,img)

        return savestring
    except Exception as e:
        print(e)
        return None

def _crop_array(array,N):

    h, w = array.shape

    left  = w/2 - N/2
    upper = h/2 - N/2
    right = w/2 + N/2
    lower = h/2 + N/2

    return array[int(upper):int(lower),int(left):int(right)]

def _exposure_correction(img):
    tmp = img.flatten()
    y, _ = np.histogram(tmp, bins=np.linspace(0,1,101))
    peak = np.argmax(y)
    exposure_correction = 0.5/(peak/100)

    imgout = img*exposure_correction

    imgout[imgout < 0] = 0
    imgout[imgout > 1] = 1

    return imgout

#-------------------------------------------------------------------------------

"""
def shatter(savedir=None,pathcol=None,k=64,verbose=False):

    if savedir==None:
        raise ValueError("Must supply 'savedir'")
    elif savedir is not None:
        if os.path.isdir(savedir)==True:
            pass
        else:
            try:
                os.mkdir(savedir)
            except:
                raise ValueError("'savedir' must be a valid filepath")

    _typecheck(**locals())
    pathcol = _pathfilter(pathcol)

    if isinstance(pathcol,string_types):
        tf = _shatter(pathcol,savedir,k)
        return tf

    elif isinstance(pathcol,pd.Series):

        counter = 0
        n = len(pathcol)
        for i in pathcol.index:
            counter+=1
            impath = pathcol.loc[i]
            if verbose==True:
                print(counter,'of',n,impath)

            tf = _shatter(impath,savedir,k,verbose)
            if counter==1:
                df = tf
            else:
                df = df.append(tf)

        return df.reset_index(drop=True)

def _shatter(impath,savedir,k,verbose):

    im = Image.open(impath)
    basename = os.path.basename(impath)[:-4]

    try:
        tiles = image_slicer.slice(impath,k,save=False)
        m = len(tiles)

        if verbose==True:
            print('recovered',m,'tiles from',basename)

        tilelist = []
        for tile in tiles:
            savestring = savedir + '/' + basename + '_' + str(tile.number) + '_' + str(tile.position[0]) + '-' + str(tile.position[1]) + '.jpg'

            try:
                tile.image.save(savestring)
                if verbose==True:
                    print(savestring)
            except Exception as e:
                print(e)

            d = {}
            d['sourceimg'] = impath
            d['sourceimgbase'] = basename
            d['tilepath'] = savestring
            d['tilenumber'] = tile.number
            d['tileposition'] = tile.position
            d['tilesize'] = tile.image.size
            d['tilecoords'] = tile.coords

            tilelist.append(d)

        tf = pd.DataFrame.from_dict(tilelist)

    except Exception as e:
        print(e)
        tf = pd.DataFrame(columns=['sourceimg',
                                   'sourceimgbase',
                                   'tilecoords',
                                   'tilenumber',
                                   'tilepath',
                                   'tileposition',
                                   'tilesize'])

    return tf
"""
