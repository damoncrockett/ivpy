from PIL import Image
from numpy import sqrt
from copy import deepcopy

from .data import _typecheck,_colfilter,_bin,_facet
from .plottools import _gridcoords,_paste,_getsizes,_round
from .plottools import _montage,_histogram,_scatter,_mat

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def show(pathcol=None,
         xcol=None,
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
        xcol (str,Series) --- sorting column
        xdomain (list,tuple) --- xmin and xmax; defaults to data extremes
        thumb (int) --- pixel value for thumbnail side
        sample (int) --- integer size of sample
        ascending (Boolean) --- sorting order
    """

    _typecheck(**locals())
    pathcol,xcol,ycol = _colfilter(pathcol,
                                      xcol=xcol,
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

        Currently, an issue needing fixing is that when 'compose' is called
        by a plotting function, the user cannot directly control facet size.
        Indirectly, the user can crank up 'thumb' in the original plot call.
    """

    _typecheck(**locals()) # won't typecheck args
    # can be (canvas,matdict) tuples if called within a plotting function
    typelist = [isinstance(item,(Image.Image,tuple)) for item in args]
    if not all(typelist):
        raise TypeError("Arguments passed to 'compose' must be PIL Images")

    n = len(args)
    rounding = kwargs.get('rounding', 'up')
    ncols = kwargs.get('ncols',_round(sqrt(n),direction=rounding))
    bg = kwargs.get('bg', '#4a4a4a')

    if ncols > n:
        raise ValueError("'ncols' cannot be larger than number of plots")

    # if user-called
    if isinstance(args[0],Image.Image):
        thumb = kwargs.get('thumb',min(_getsizes(args)))
        w,h,coords = _gridcoords(n,ncols,thumb)
        metacanvas = Image.new('RGB',(w,h),bg)

        for i in range(n):
            canvas = args[i]
            tmp = deepcopy(canvas) # copy because thumbnail always inplace
            tmp.thumbnail((thumb,thumb),Image.ANTIALIAS)
            metacanvas.paste(tmp,coords[i])

    # if called by plotting function
    elif isinstance(args[0],tuple):
        thumb = kwargs.get('thumb',min(_getsizes([item[0] for item in args])))

        mattedfacets = []
        for arg in args:
            canvas = arg[0]
            matdict = arg[1]
            canvas.thumbnail((thumb,thumb),Image.ANTIALIAS)
            mattedfacets.append(_mat(canvas,**matdict))

        side = mattedfacets[0].width # any side in the list is fine, all same
        w,h,coords = _gridcoords(n,ncols,side)
        metacanvas = Image.new('RGB',(w,h),bg)

        for i in range(n):
            canvas = mattedfacets[i]
            metacanvas.paste(canvas,coords[i])

    return metacanvas

#------------------------------------------------------------------------------

def montage(pathcol=None,
            xcol=None,
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
        xcol (str,Series) --- sorting column
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
    pathcol,xcol,ycol,facetcol = _colfilter(pathcol,
                                               xcol=xcol,
                                               xdomain=xdomain,
                                               sample=sample,
                                               ascending=ascending,
                                               facetcol=facetcol)

    if facetcol is None:
        return _montage(**locals())
    elif facetcol is not None:
        facetlist = _facet(**locals())
        plotlist = [_montage(**facet) for facet in facetlist]
        return compose(*plotlist)

#------------------------------------------------------------------------------

def histogram(xcol,
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
              facetcol=None,
              xlabel=None):

    """
    Cartesian or polar histogram of images

    Args:
        xcol (str,Series) --- histogram axis; must be supplied
        xdomain (list,tuple) --- xmin and xmax; defaults to data extremes
        pathcol (Series) --- col of image paths to be plotted
        ycol (str,Series) --- vertical sorting feature if desired
        ydomain (list,tuple) --- ymin and ymax; defaults to data extremes
        thumb (int) --- pixel value for thumbnail side
        bins (int,seq) --- number of bins used to discretize xcol;
            can alternatively be entered as array of bin edges
        sample (int) --- integer size of sample
        idx (Boolean) --- whether to print index on image
        ascending (Boolean) --- vertical sorting direction
        bg (color) --- background color
        coordinates (str) --- 'cartesian' or 'polar'
        facetcol (str,Series) --- col to split data into plot facets
        xlabel (Boolean) --- whether to include bin labels
    """

    _typecheck(**locals())
    pathcol,xcol,ycol,facetcol = _colfilter(pathcol,
                                               xcol=xcol,
                                               xdomain=xdomain,
                                               ycol=ycol,
                                               ydomain=ydomain,
                                               sample=sample,
                                               facetcol=facetcol)

    if facetcol is None:
        return _histogram(**locals())
    elif facetcol is not None:
        facetlist = _facet(**locals())
        plotlist = [_histogram(**facet) for facet in facetlist]
        return compose(*plotlist)

#------------------------------------------------------------------------------

def scatter(xcol,
            ycol,
            pathcol=None,
            thumb=32,
            side=500,
            sample=False,
            idx=False,
            xdomain=None,
            ydomain=None,
            xbins=None,
            ybins=None,
            bg="#4a4a4a",
            coordinates='cartesian',
            facetcol=None,
            xlabel=None,
            ylabel=None):

    """
    Cartesian or polar scatterplot of images

    Args:
        xcol (str,Series) --- x-axis; must be supplied
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
        xlabel (Boolean) --- whether to include x-axis labels
        ylabel (Boolean) --- whether to include y-axis labels
    """

    _typecheck(**locals())
    pathcol,xcol,ycol,facetcol = _colfilter(pathcol,
                                               xcol=xcol,
                                               xdomain=xdomain,
                                               ycol=ycol,
                                               ydomain=ydomain,
                                               sample=sample,
                                               facetcol=facetcol)

    if facetcol is None:
        return _scatter(**locals())
    elif facetcol is not None:
        facetlist = _facet(**locals())
        plotlist = [_scatter(**facet) for facet in facetlist]
        return compose(*plotlist)
