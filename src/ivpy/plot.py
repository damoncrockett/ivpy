from PIL import Image
from numpy import sqrt,repeat,arange
import pandas as pd
from shapely.geometry import Point
from copy import deepcopy

from .data import _typecheck,_colfilter,_bin
from .plottools import _scalecart,_scalepol
from .plottools import _gridcoords,_paste,_getsizes,_round

#------------------------------------------------------------------------------

def show(pathcol=None,
         featcol=None,
         thumb=False,
         sample=False,
         idx=False,
         bg='#4a4a4a',
         ascending=False):

    """
    Shows either a single image by index or a pathcol, possibly sampled,
    as a scrolling, sortable rect montage

    Args:
        pathcol (int,Series) --- single index or col of image paths to be shown
        featcol (str,Series) --- sorting column
        thumb (int) --- pixel value for thumbnail side
        sample (int) --- integer size of sample
        ascending (Boolean) --- sorting order
    """

    _typecheck(**locals())
    pathcol,featcol,ycol = _colfilter(pathcol,
                                      featcol=featcol,
                                      sample=sample,
                                      ascending=ascending)

    if isinstance(pathcol, str): # single pathstring
        im = Image.open(pathcol)
        if thumb!=False:
            im.thumbnail((thumb,thumb),Image.ANTIALIAS)
        return im
    else:
        if thumb==False:
            thumb = 100
            ncols = int(980/thumb) # hard-coded bc of Jupyter cell sizes
        else:
            ncols = int(980/thumb) # n.b. Python 3 defaults to float divide

        n = len(pathcol)
        w,h,coords = _gridcoords(n,ncols,thumb)
        canvas = Image.new('RGB',(w,h),bg)
        _paste(pathcol,thumb,idx,canvas,coords)

        return canvas

def montage(pathcol=None,
            featcol=None,
            thumb=100,
            sample=False,
            idx=False,
            bg='#4a4a4a',
            shape='square',
            ascending=False):

    """
    Square or circular montage of images

    Args:
        pathcol (Series) --- col of image paths to be plotted
        featcol (str,Series) --- sorting column
        thumb (int) --- pixel value for thumbnail side
        sample (int) --- integer size of sample
        idx (Boolean) --- whether to print index on image
        bg (color) --- background color
        shape (str) --- square or circular montage
        ascending (Boolean) --- sorting order
    """

    _typecheck(**locals())
    pathcol,featcol,ycol = _colfilter(pathcol,
                                      featcol=featcol,
                                      sample=sample,
                                      ascending=ascending)
    n = len(pathcol)

    if shape=='square':

        ncols = int(sqrt(n))
        w,h,coords = _gridcoords(n,ncols,thumb)
        canvas = Image.new('RGB',(w,h),bg)
        _paste(pathcol,thumb,idx,canvas,coords)

    elif shape=='circle':

        side = int(sqrt(n)) + 5 # may have to tweak this
        x,y = range(side)*side,repeat(range(side),side)
        gridlist = [Point(item) for item in zip(x,y)]

        canvas = Image.new('RGB',(side*thumb,side*thumb),bg)

        # plot center image
        maximus = Point(side/2,side/2)
        coords = [(int(maximus.x*thumb),int(maximus.y*thumb))]
        _paste(pathcol[:1],thumb,idx,canvas,coords)
        gridlist.remove(maximus)

        # compute distances from center; sort by distance
        dists = [maximus.distance(item) for item in gridlist]
        tmp = pd.DataFrame({"gridlist":gridlist,"dists":dists})
        tmp.sort_values(by='dists',inplace=True) # ascending
        gridlist = tmp.gridlist.iloc[:n-1] # n-1 bc we removed maximus

        # plot remaining images
        coords = [(int(item.x*thumb),int(item.y*thumb)) for item in gridlist]
        _paste(pathcol[1:],thumb,idx,canvas,coords)

    return canvas

def compose(*args,**kwargs):

    """
    Composes PIL canvases into metacanvas

    Args:
        *args --- any number of canvases, given by name or plot function
        ncols (int) --- number of columns in metacanvas (optional)
        rounding (str) --- when ncols is None, round ncols 'up' or 'down'
        thumb (int) --- pixel value for thumbnail side
        bg (color) --- background color
    """

    _typecheck(**locals()) # won't typecheck args
    typelist = [isinstance(item, Image.Image) for item in args]
    if not all(typelist):
        raise TypeError("Arguments passed to 'compose' must be PIL Images")

    n = len(args)
    rounding = kwargs.get('rounding', 'up')
    ncols = kwargs.get( 'ncols', _round(sqrt(n),direction=rounding) )
    thumb = kwargs.get( 'thumb', max(_getsizes(args)) )
    bg = kwargs.get('bg', '#4a4a4a')

    if ncols > n:
        raise ValueError("'ncols' cannot be larger than number of plots")

    w,h,coords = _gridcoords(n,ncols,thumb)
    metacanvas = Image.new('RGB',(w,h),bg)

    for i in range(n):
        canvas = args[i]
        tmp = deepcopy(canvas) # copy because thumbnail always inplace
        tmp.thumbnail((thumb,thumb),Image.ANTIALIAS)
        metacanvas.paste(tmp,coords[i])

    return metacanvas

def histogram(featcol,
              pathcol=None,
              ycol=None,
              thumb=28,
              nbins=35,
              sample=False,
              idx=False,
              ascending=False,
              bg="#4a4a4a",
              quantile=False,
              coordinates='cartesian'): # not yet implemented

    """
    Cartesian or polar histogram of images

    Args:
        featcol (str,Series) --- histogram axis; must be supplied
        pathcol (Series) --- col of image paths to be plotted
        ycol (str,Series) --- vertical sorting feature if desired
        thumb (int) --- pixel value for thumbnail side
        nbins (int) --- number of bins used to discretize featcol;
            can alternatively be entered as array of bin edges
        sample (int) --- integer size of sample
        idx (Boolean) --- whether to print index on image
        ascending (Boolean) --- vertical sorting direction
        bg (color) --- background color
        quantile (Boolean) --- whether binning is quantile-based;
            if True, produces nearly even spread across bins
        coordinates (str) --- 'cartesian' or 'polar'
    """

    _typecheck(**locals())
    pathcol,featcol,ycol = _colfilter(pathcol,
                                      featcol=featcol,
                                      ycol=ycol,
                                      sample=sample)

    if quantile==False:
        xbin = pd.cut(featcol,nbins,labels=False)
    elif quantile==True:
        xbin = pd.qcut(featcol,nbins,labels=False,duplicates='drop')

    bins = xbin.unique()
    binmax = xbin.value_counts().max()
    plotheight = thumb * binmax
    canvas = Image.new('RGB',(thumb*nbins,plotheight),bg)

    for binlabel in bins:
        if ycol is not None:
            ycol_bin = ycol[xbin==binlabel]
            ycol_bin = ycol_bin.sort_values(ascending=ascending)
            pathcol_bin = pathcol.loc[ycol_bin.index]
        else:
            pathcol_bin = pathcol[xbin==binlabel]

        n = len(pathcol_bin)
        xcoord = thumb * binlabel
        ycoord = plotheight - thumb # bc paste loc is UPPER left corner
        ycoords = arange(ycoord,plotheight-thumb*(n+1),-thumb)
        coords = [tuple((xcoord,item)) for item in ycoords]
        _paste(pathcol_bin,thumb,idx,canvas,coords)

    return canvas

def scatter(featcol,
            ycol,
            pathcol=None,
            thumb=32,
            side=500,
            sample=False,
            idx=False,
            gridded=False,
            xdomain=None,
            ydomain=None,
            xbins=None,
            ybins=None,
            bg="#4a4a4a",
            coordinates='cartesian'):

    """
    Cartesian or polar scatterplot of images

    Args:
        featcol (str,Series) --- x-axis; must be supplied
        ycol (str,Series) --- y-axis; must be supplied
        pathcol (Series) --- col of image paths to be plotted
        thumb (int) --- pixel value for thumbnail side
        side (int) --- length of plot side in pixels; all plots enforced square
        sample (int) --- integer size of sample
        idx (Boolean) --- whether to print index on image
        xdomain (list,tuple) --- xmin and xmax; defaults to data extremes
        ydomain (list,tuple) --- ymin and ymax; defaults to data extremes
        xbins (int,seq) --- 'bins' argument passed to pd.cut()
        ybins (int,seq) --- 'bins' argument passed to pd.cut()
        bg (color) --- background color
        coordinates (str) --- 'cartesian' or 'polar'
    """

    _typecheck(**locals())
    pathcol,featcol,ycol = _colfilter(pathcol,
                                      featcol=featcol,
                                      ycol=ycol,
                                      sample=sample)

    if xbins is not None:
        featcol = _bin(featcol,xbins)
    if ybins is not None:
        ycol = _bin(ycol,ybins)

    if coordinates=='cartesian':
        x,y = _scalecart(featcol,ycol,xdomain,ydomain,side,thumb)
        phis = None # a bit hacky but can't think of a better way yet
    elif coordinates=='polar':
        x,y,phis = _scalepol(featcol,ycol,xdomain,ydomain,side,thumb)
    coords = zip(x,y)
    canvas = Image.new('RGB',(side,side),bg) # fixed size
    _paste(pathcol,thumb,idx,canvas,coords,coordinates,phis)

    return canvas
