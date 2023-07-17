from PIL import Image,ImageDraw
from pandas import Series
from numpy import sqrt,arange,ndarray,mean
from copy import deepcopy
from six import string_types

from .data import _typecheck,_colfilter,_facet
from .plottools import _gridcoords,_paste,_getsizes,_round
from .plottools import _border,_montage,_histogram,_scatter,_facetcompose
from .plottools import _titlesize,_entitle,_bottom_left_corner

seq_types = (list,tuple,ndarray,Series)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def show(pathcol=None,
         xcol=None,
         notecol=None,
         xdomain=None,
         thumb=False,
         sample=False,
         idx=False,
         bg='#212121',
         ascending=False):

    """
    Shows either a single image by index or a pathcol, possibly sampled,
    as a scrolling, sortable rect montage

    Args:
        pathcol (int,Series) --- single index or col of image paths to be shown
        xcol (str,Series) --- sorting column
        notecol (str,Series) --- annotation column
        xdomain (list,tuple) --- xmin and xmax; defaults to data extremes
        thumb (int) --- pixel value for thumbnail side
        sample (int) --- integer size of sample
        idx (Boolean) --- whether to print indices on images
        bg (color) --- background color
        ascending (Boolean) --- sorting order
    """

    try:
        _typecheck(**locals()['kwargs'])
    except:
        _typecheck(**locals())

    pathcol,xcol,ycol,facetcol,notecol = _colfilter(pathcol,
                                            xcol=xcol,
                                            notecol=notecol,
                                            xdomain=xdomain,
                                            sample=sample,
                                            ascending=ascending)

    if isinstance(pathcol, str): # single pathstring
        im = Image.open(pathcol)
        if thumb!=False:
            im.thumbnail((thumb,thumb),Image.Resampling.LANCZOS)
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
        _paste(pathcol,thumb,idx,canvas,coords,notecol=notecol)

        return canvas

#------------------------------------------------------------------------------

def montage(pathcol=None,
            xcol=None,
            xdomain=None,
            thumb=100,
            sample=False,
            idx=False,
            bg='#212121',
            shape='square',
            ascending=False,
            facetcol=None,
            notecol=None,
            title=None,
            border=False):

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
        notecol (str,Series) --- annotation column
        title (str) --- plot title
        border (Boolean) --- whether to border facets
    """

    try:
        _typecheck(**locals()['kwargs'])
    except:
        _typecheck(**locals())

    pathcol,xcol,ycol,facetcol,notecol = _colfilter(pathcol,
                                               xcol=xcol,
                                               xdomain=xdomain,
                                               sample=sample,
                                               ascending=ascending,
                                               facetcol=facetcol,
                                               notecol=notecol)

    if facetcol is None:
        canvas = _montage(**locals())

    elif facetcol is not None:
        facetlist,_ = _facet(**locals())
        plotlist = [_montage(**facet) for facet in facetlist]
        canvas = _facetcompose(*plotlist,bg=bg,border=border)

    if title is not None:
        font,_,fontHeight = _titlesize(canvas)
        canvas = _entitle(canvas,title,font,fontHeight,bg)

    return canvas

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
              bg="#212121",
              coordinates='cartesian',
              facetcol=None,
              notecol=None,
              xaxis=None,
              yaxis=None,
              flip=False,
              dot=False,
              bincols=1,
              border=False,
              title=None,
              axislines=False):

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
        notecol (str,Series) --- annotation column
        xaxis (Boolean,int) --- whether to include bin labels or number to include
        yaxis (Boolean,int) --- whether to include bin counts or number to include
        flip (Boolean) --- whether to flip images vertically; for 'under'
            histogram
        dot (Boolean) --- whether to use uniform dots as plotting units
        bincols (int) --- number of columns per bin; usually 1, higher if some
            bins are excessively large
        border (Boolean) --- whether to border facets
        title (str) --- plot title
        axislines (Boolean) --- whether to draw axis lines
    """

    try:
        _typecheck(**locals()['kwargs'])
    except:
        _typecheck(**locals())

    pathcol,xcol,ycol,facetcol,notecol = _colfilter(pathcol,
                                               xcol=xcol,
                                               xdomain=xdomain,
                                               ycol=ycol,
                                               ydomain=ydomain,
                                               sample=sample,
                                               facetcol=facetcol,
                                               notecol=notecol)

    if xdomain is None:
        xdomain = (xcol.min(),xcol.max())
    
    if facetcol is None:
        canvas = _histogram(**locals())

    elif facetcol is not None:

        if flip==True:
            raise ValueError("Cannot flip images in a faceted plot")

        facetlist,binmax = _facet(**locals(),plottype='histogram')
        plotlist = [_histogram(**facet,binmax=binmax) for facet in facetlist]
        canvas = _facetcompose(*plotlist,border=border,bg=bg)

    if title is not None:
        font,_,fontHeight = _titlesize(canvas)
        canvas = _entitle(canvas,title,font,fontHeight)

    return canvas

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
            bg="#212121",
            coordinates='cartesian',
            facetcol=None,
            notecol=None,
            xaxis=None,
            yaxis=None,
            dot=False,
            border=False,
            title=None,
            axislines=False):

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
        notecol (str,Series) --- annotation column
        xaxis (Boolean,int) --- whether to include x-axis labels or number to include
        yaxis (Boolean,int) --- whether to include y-axis labels or number to inlucde
        dot (Boolean) --- whether to use uniform dots as plotting units
        border (Boolean) --- whether to border plots
        title (str) --- plot title
        axislines (Boolean) --- whether to draw axis lines
    """

    try:
        _typecheck(**locals()['kwargs'])
    except:
        _typecheck(**locals())

    pathcol,xcol,ycol,facetcol,notecol = _colfilter(pathcol,
                                               xcol=xcol,
                                               xdomain=xdomain,
                                               ycol=ycol,
                                               ydomain=ydomain,
                                               sample=sample,
                                               facetcol=facetcol,
                                               notecol=notecol,
                                               scatter=True)

    if xdomain is None:
        xdomain = (xcol.min(),xcol.max())
    if ydomain is None:
        ydomain = (ycol.min(),ycol.max())

    if facetcol is None:
        canvas = _scatter(**locals())

    elif facetcol is not None:
        facetlist,_ = _facet(**locals())
        plotlist = [_scatter(**facet) for facet in facetlist]
        canvas = _facetcompose(*plotlist,border=border,bg=bg)

    if title is not None:
        font,_,fontHeight = _titlesize(canvas)
        canvas = _entitle(canvas,title,font,fontHeight)

    return canvas

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def line(*args,**kwargs):

    """
    Simple line plot. Will eliminate null values.

    Args:
        *args --- sequences to be plotted as lines
        side (int) --- length of plot side in pixels; all plots enforced square
        bg (color) --- background color
        fill (color or sequence of colors) --- line color
        width (int or sequence of ints) --- line weight
    """

    try:
        _typecheck(**locals()['kwargs'])
    except:
        _typecheck(**locals())

    typelist = [isinstance(item,(seq_types)) for item in args]
    if not all(typelist):
        raise TypeError("Arguments passed to 'line' must be sequences")

    fill = kwargs.get('fill', 'white')
    width = kwargs.get('width', 1)
    side = kwargs.get('side', 400)
    bg = kwargs.get('bg', '#212121')

    if bg=='transparent':
        canvas = Image.new('RGBA',(side,side),None) # fixed size
    else:
        canvas = Image.new('RGB',(side,side),bg) # fixed size

    draw = ImageDraw.Draw(canvas)

    anynull = [mean(Series(arg).notnull())!=1 for arg in args]
    if any(anynull):
        raise ValueError("Cannot pass null sequence values to 'line'")

    lens = list(set([len(arg) for arg in args]))
    n = lens[0]
    if len(lens) > 1:
        raise ValueError("All sequences passed to 'line' must be the same length")

    ymax = max([max(item) for item in args])
    ymin = min([min(item) for item in args])
    yrange = ymax - ymin

    for i,arg in enumerate(args):
        incr = side / (n-1)
        xs = arange(0,side+incr,incr)
        xs = [int(item) for item in xs[:n]]
        ys = [int( (1 - (item-ymin) / yrange ) * side ) for item in arg]
        coords = zip(xs,ys)

        if isinstance(fill,seq_types):
            fcolor = fill[i]
        else:
            fcolor = fill

        if isinstance(width,seq_types):
            lwidth = width[i]
        else:
            lwidth = width

        draw.line(list(coords),fill=fcolor,width=lwidth,joint='curve')

    return canvas

#------------------------------------------------------------------------------

def compose(*args,ncols=None,rounding='down',thumb=None,bg='#212121',border=False):

    """
    Composes PIL canvases into metacanvas

    Args:
        *args --- any number of canvases, given by name or plot function
        ncols (int) --- number of columns in metacanvas (optional)
        rounding (str) --- when ncols is None, round ncols 'up' or 'down'
        thumb (int) --- pixel value for thumbnail side
        bg (color) --- background color
        border (Boolean) --- whether to border plots
    """

    typelist = [isinstance(item,Image.Image) for item in args]
    if not all(typelist):
        raise TypeError("Arguments passed to 'compose' must be PIL Images")

    n = len(args)
    if ncols is None:
        ncols = _round(sqrt(n),direction=rounding)
    if thumb is None:
        thumb = min(_getsizes(args))

    try:
        _typecheck(**locals()['kwargs'])
    except:
        _typecheck(**locals())

    if ncols > n:
        raise ValueError("'ncols' cannot be larger than number of plots")

    thumbargs = [deepcopy(arg) for arg in args]
    for thumbarg in thumbargs:
        thumbarg.thumbnail((thumb,thumb),Image.Resampling.LANCZOS)

    w,h,coords = _gridcoords(n,ncols,thumb)
    
    if bg is None:
        metacanvas = Image.new('RGBA',(w,h),bg)
    else:
        metacanvas = Image.new('RGB',(w,h),bg)

    for i in range(n):
        canvas = thumbargs[i]
        if canvas.size!=(thumb,thumb):
            canvas = _bottom_left_corner(canvas,thumb,bg)
        if border:
            canvas = _border(canvas,bg=bg)
       
        if bg is None:
            metacanvas.paste(canvas,coords[i],canvas)
        else:
            metacanvas.paste(canvas,coords[i])

    return metacanvas

#------------------------------------------------------------------------------

def overlay(*args,**kwargs):

    """
    Overlay images

    Args:
        *args --- images to be overlaid
        side (int) --- length of plot side in pixels; all plots enforced square
        bg (color) --- background color
    """

    try:
        _typecheck(**locals()['kwargs'])
    except:
        _typecheck(**locals())

    side = kwargs.get('side', 400)
    bg = kwargs.get('bg', 'white')

    if bg=='transparent':
        canvas = Image.new('RGBA',(side,side),None) # fixed size
    else:
        canvas = Image.new('RGB',(side,side),bg) # fixed size

    for arg in args:

        if isinstance(arg,string_types):
            arg = Image.open(arg)
            
        if arg.size != (side,side):
            arg = arg.resize((side,side))
        
        canvas.paste(arg,(0,0),arg)

    return canvas
