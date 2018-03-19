from PIL import Image
from numpy import sqrt,repeat,arange
import pandas as pd
from shapely.geometry import Point
from copy import deepcopy

from .data import _typecheck,_colfilter,_bin
from .plottools import _scalecart,_scalepol,_bin2phi,_bin2phideg,_pol2cart
from .plottools import _gridcoords,_gridcoordscircle,_paste,_getsizes,_round
from .plottools import _histcoordscart,_histcoordspolar,_gridcoordscirclemax

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def show(pathcol=None,
         featcol=None,
         xdomain=None,
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
        xdomain (list,tuple) --- xmin and xmax; defaults to data extremes
        thumb (int) --- pixel value for thumbnail side
        sample (int) --- integer size of sample
        ascending (Boolean) --- sorting order
    """

    _typecheck(**locals())
    pathcol,featcol,ycol = _colfilter(pathcol,
                                      featcol=featcol,
                                      xdomain=xdomain,
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

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

def montage(pathcol=None,
            featcol=None,
            xdomain=None,
            thumb=100,
            sample=False,
            idx=False,
            bg='#4a4a4a',
            shape='square',
            ascending=False,
            facetcol=None):

    """
    Square or circular montage of images

    Args:
        pathcol (Series) --- col of image paths to be plotted
        featcol (str,Series) --- sorting column
        xdomain (list,tuple) --- xmin and xmax; defaults to data extremes
        thumb (int) --- pixel value for thumbnail side
        sample (int) --- integer size of sample
        idx (Boolean) --- whether to print index on image
        bg (color) --- background color
        shape (str) --- square or circular montage
        ascending (Boolean) --- sorting order
        facetcol (str,Series) --- col to split data into plot facets
    """

    _typecheck(**locals())
    pathcol,featcol,ycol,facetcol = _colfilter(pathcol,
                                               featcol=featcol,
                                               xdomain=xdomain,
                                               sample=sample,
                                               ascending=ascending,
                                               facetcol=facetcol)
    n = len(pathcol)

    if shape=='square':

        ncols = int(sqrt(n))
        w,h,coords = _gridcoords(n,ncols,thumb)
        canvas = Image.new('RGB',(w,h),bg)
        _paste(pathcol,thumb,idx,canvas,coords)

    elif shape=='circle':

        side = int(sqrt(n)) + 5 # may have to tweak this
        canvas = Image.new('RGB',(side*thumb,side*thumb),bg)

        # center image
        gridlist,maximus,coords = _gridcoordscirclemax(side,thumb)
        _paste(pathcol[:1],thumb,idx,canvas,coords)
        gridlist.remove(maximus)

        # remaining images
        coords = _gridcoordscircle(n,maximus,gridlist,thumb)
        _paste(pathcol[1:],thumb,idx,canvas,coords)

    return canvas

#------------------------------------------------------------------------------

def histogram(featcol,
              xdomain=None,
              pathcol=None,
              ycol=None,
              ydomain=None,
              thumb=28,
              bins=35,
              sample=False,
              idx=False,
              ascending=False,
              bg="#4a4a4a",
              coordinates='cartesian',
              facetcol=None):

    """
    Cartesian or polar histogram of images

    Args:
        featcol (str,Series) --- histogram axis; must be supplied
        xdomain (list,tuple) --- xmin and xmax; defaults to data extremes
        pathcol (Series) --- col of image paths to be plotted
        ycol (str,Series) --- vertical sorting feature if desired
        ydomain (list,tuple) --- ymin and ymax; defaults to data extremes
        thumb (int) --- pixel value for thumbnail side
        bins (int,seq) --- number of bins used to discretize featcol;
            can alternatively be entered as array of bin edges
        sample (int) --- integer size of sample
        idx (Boolean) --- whether to print index on image
        ascending (Boolean) --- vertical sorting direction
        bg (color) --- background color
        coordinates (str) --- 'cartesian' or 'polar'
        facetcol (str,Series) --- col to split data into plot facets
    """

    _typecheck(**locals())
    pathcol,featcol,ycol,facetcol = _colfilter(pathcol,
                                               featcol=featcol,
                                               xdomain=xdomain,
                                               ycol=ycol,
                                               ydomain=ydomain,
                                               sample=sample,
                                               facetcol=facetcol)

    """
    This is domain expansion. The histogram ydomain can be contracted; it simply
    removes data points. But it cannot be expanded, since y in a histogram is
    not a proper axis. The user can expand the xdomain either using that kwarg
    or by submitting a set of domain-expanding bin edges. If user gives xdomain
    and an integer 'bins' argument, that xdomain is split into equal-width bins.
    If the user submits other bin edges, those are the edges, regardless of
    whether they match the submitted xdomain. This makes it possible, for
    example, to restrict the domain using 'xdomain' and expand the plotting
    space using 'bins'.
    """
    if xdomain is not None:
        xrange = xdomain[1]-xdomain[0]
        if isinstance(bins,int):
            # n.b.: this is slightly different than giving int to pd.cut
            increment = float(xrange)/bins
            bins = arange(xdomain[0],xdomain[1]+increment,increment)

    xbin = pd.cut(featcol,bins,labels=False,include_lowest=True)
    nbins = len(pd.cut(featcol,bins,include_lowest=True).value_counts())
    nonemptybins = xbin.unique() # will ignore empty bins
    binmax = xbin.value_counts().max()

    if coordinates=='cartesian':
        plotheight = thumb * binmax
        canvas = Image.new('RGB',(thumb*nbins,plotheight),bg)
    elif coordinates=='polar':
        canvas = Image.new('RGB',(binmax*2*thumb+thumb,binmax*2*thumb+thumb),bg)

    for binlabel in nonemptybins:
        if ycol is not None:
            ycol_bin = ycol[xbin==binlabel]
            ycol_bin = ycol_bin.sort_values(ascending=ascending)
            pathcol_bin = pathcol.loc[ycol_bin.index]
        else:
            pathcol_bin = pathcol[xbin==binlabel]

        n = len(pathcol_bin)
        if coordinates=='cartesian':
            coords = _histcoordscart(n,binlabel,plotheight,thumb)
            _paste(pathcol_bin,thumb,idx,canvas,coords,coordinates)
        elif coordinates=='polar':
            coords = _histcoordspolar(n,binlabel,binmax,nbins,thumb)
            _paste(pathcol_bin,thumb,idx,canvas,coords,coordinates,phis)

    return canvas

#------------------------------------------------------------------------------

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
            coordinates='cartesian',
            facetcol=None):

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
        facetcol (str,Series) --- col to split data into plot facets
    """

    _typecheck(**locals())
    pathcol,featcol,ycol,facetcol = _colfilter(pathcol,
                                               featcol=featcol,
                                               xdomain=xdomain,
                                               ycol=ycol,
                                               ydomain=ydomain,
                                               sample=sample,
                                               facetcol=facetcol)

    if xbins is not None:
        featcol = _bin(featcol,xbins)
    if ybins is not None:
        ycol = _bin(ycol,ybins)

    canvas = Image.new('RGB',(side,side),bg) # fixed size
    # xdomain and ydomain only active at this stage if expanding
    if coordinates=='cartesian':
        x,y = _scalecart(featcol,ycol,xdomain,ydomain,side,thumb)
        coords = zip(x,y)
        _paste(pathcol,thumb,idx,canvas,coords,coordinates)
    elif coordinates=='polar':
        x,y,phis = _scalepol(featcol,ycol,xdomain,ydomain,side,thumb)
        coords = zip(x,y)
        _paste(pathcol,thumb,idx,canvas,coords,coordinates,phis)

    return canvas
