import os
import pandas as pd
from PIL import Image
from six import string_types
import image_slicer
from glob2 import glob

from .data import _pathfilter,_typecheck

#-------------------------------------------------------------------------------

def resize(savedir=None,pathcol=None,thumb=256):
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
        return _resize(pathcol,savedir,thumb)

    elif isinstance(pathcol,pd.Series):
        pathcol_resized = pd.Series(index=pathcol.index)
        for i in pathcol.index:
            impath = pathcol.loc[i]
            pathcol_resized.loc[i] = _resize(impath,savedir,thumb)
        return pathcol_resized

def _resize(impath,savedir,thumb):
    try:
        im = Image.open(impath)
        im.thumbnail((thumb,thumb),Image.ANTIALIAS)
        basename = os.path.basename(impath)
        savestring = savedir + "/" + basename
        im.save(savestring)
        return savestring
    except:
        return None
#-------------------------------------------------------------------------------

def shatter(savedir=None,pathcol=None,k=64):

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
        _shatter(pathcol,savedir,k)
        return _collect_shatter(savedir)

    elif isinstance(pathcol,pd.Series):
        for i in pathcol.index:
            impath = pathcol.loc[i]
            _shatter(impath,savedir,k)
        return _collect_shatter(savedir)

def _shatter(impath,savedir,k):
    im = Image.open(impath)
    basename = os.path.basename(impath)[:-4]
    try:
        tiles = image_slicer.slice(impath,k,save=False)
        image_slicer.save_tiles(tiles,directory=savedir,prefix=basename)
    except Exception as e:
        print(e)

def _collect_shatter(savedir):
    allshatter = glob(os.path.join(savedir,'*.png'))
    tmp = pd.DataFrame({'localpath':allshatter})
    basenames = [os.path.basename(item) for item in tmp.localpath]
    tmp['basename'] = [item.split('_')[0] for item in basenames]
    return tmp
