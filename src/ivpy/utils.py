import os
import pandas as pd
from PIL import Image
from six import string_types
#import image_slicer
from glob2 import glob

from .data import _pathfilter,_typecheck

#-------------------------------------------------------------------------------

def resize(savedir=None,pathcol=None,thumb=256,verbose=False):
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
        n = len(pathcol)
        for j,i in enumerate(pathcol.index):
            impath = pathcol.loc[i]
            if verbose==True:
                print(j+1,'of',n,impath)
            pathcol_resized.loc[i] = _resize(impath,savedir,thumb)
        return pathcol_resized

def _resize(impath,savedir,thumb):
    try:
        im = Image.open(impath)
        im.thumbnail((thumb,thumb),Image.LANCZOS)
        basename = os.path.basename(impath)
        savestring = savedir + "/" + basename
        im.save(savestring)
        return savestring
    except:
        return None
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
