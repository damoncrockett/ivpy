import pandas as pd
import os
from PIL import Image,ImageDraw,ImageFont,ImageColor
from numpy import repeat, sqrt, arange, radians, cos, sin, linspace
import numpy as np
from math import ceil
from six import string_types
from copy import deepcopy

try:
    import requests
except:
    print("'requests' module not installed")

from .data import _bin, _typecheck

int_types = (int,np.int8,np.int16,np.int32,np.int64,
             np.uint8,np.uint16,np.uint32,np.uint64)

seq_types = (list,tuple,np.ndarray,pd.Series)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def _border(im,bg):

    if _islight(bg):
        fill = 'black'
    else:
        fill = 'white'

    _typecheck(**locals())

    draw = ImageDraw.Draw(im)
    # n.b.: lines are drawn under and to the right of the starting pixel
    draw.line([(im.width,im.height-1),(0,im.height-1)],fill=fill,width=1)
    draw.line([(0,im.height),(0,0)],fill=fill,width=1)

    return im

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

"""
The underscored plotting functions take a raft of kwargs, most of which are idle
because they were used to sort, sample, etc., which determines what the cols
look like. But they need to be listed here because the underscore functions are
called using locals(), which will include all kwargs passed to the user funcs.
"""

def _montage(pathcol=None,
             xcol=None,
             xdomain=None,
             ycol=None,
             ydomain=None,
             thumb=None,
             sample=None,
             idx=None,
             bg=None,
             shape=None,
             ascending=None,
             facetcol=None,
             facettitle=None,
             notecol=None,
             border=None,
             title=None):

    n = len(pathcol)

    if shape=='square':
        ncols = ceil(sqrt(n))
        w,h,coords = _gridcoords(n,ncols,thumb)
        canvas = Image.new('RGB',(w,h),bg)
        _paste(pathcol,thumb,idx,canvas,coords,notecol=notecol)
    elif shape=='rect':
        ncols = ceil( sqrt( n / 0.5625 ) )
        w,h,coords = _gridcoords(n,ncols,thumb)
        canvas = Image.new('RGB',(w,h),bg)
        _paste(pathcol,thumb,idx,canvas,coords,notecol=notecol)
    elif shape=='circle':
        side = int(sqrt(n)) + 5 # may have to tweak this
        canvas = Image.new('RGB',(side*thumb,side*thumb),bg)

        # center image
        gridlist,maximus,coords = _gridcoordscirclemax(side,thumb)
        _paste(pathcol[:1],thumb,idx,canvas,coords,notecol=notecol)
        gridlist.remove(maximus)

        # remaining images
        coords = _gridcoordscircle(n,maximus,gridlist,thumb)
        _paste(pathcol[1:],thumb,idx,canvas,coords,notecol=notecol)
    else:
        # if shape is none of the above, it will be an integer number of columns
        w,h,coords = _gridcoords(n,shape,thumb)
        canvas = Image.new('RGB',(w,h),bg)
        _paste(pathcol,thumb,idx,canvas,coords,notecol=notecol)

    if facetcol is None:
        return canvas

    elif facetcol is not None:
        matdict = {'bg':bg,
                   'facettitle':facettitle,
                   'plottype':'montage'}

        return canvas,matdict

#-------------------------------------------------------------------------------

def _histogram(xcol=None,
               xdomain=None,
               pathcol=None,
               ycol=None,
               ydomain=None,
               thumb=None,
               bins=None,
               sample=None,
               idx=None,
               ascending=None,
               bg=None,
               coordinates=None,
               facetcol=None,
               facettitle=None,
               xaxis=None,
               yaxis=None,
               notecol=None,
               flip=None,
               dot=None,
               bincols=None,
               border=None,
               binmax=None,
               title=None,
               axislines=None):

    """
    If user submitted bin sequence leaves out some rows, user must pass xdomain
    argument, because data subsetting not allowable this late in the process.
    """
    if isinstance(bins,(seq_types)):
        if any([xcol.max()>bins[-1],xcol.min()<bins[0]]):
            raise ValueError("""Submitted bin edges do not capture all the data.
                                Domain contraction requires 'xdomain' argument,
                                and bin edges must span the domain.
                                """)

    """
    This (below) is domain expansion. The histogram ydomain can be contracted:
    it simply removes data points. But it cannot be expanded, since y in a
    histogram is not a proper axis. The user can expand the xdomain either using
    that kwarg or by submitting a set of domain-expanding bin edges. If user
    gives xdomain and an integer 'bins' argument, that xdomain is split into
    equal-width bins. If the user submits other bin edges, those are the edges,
    regardless of whether they match the submitted xdomain. This makes it
    possible, for example, to restrict the domain using 'xdomain' and expand the
    plotting space using 'bins'.
    """
    if xdomain is not None:
        if isinstance(bins,int):
            xbin = pd.cut(xcol,linspace(xdomain[0],xdomain[1],bins+1),labels=False,include_lowest=True)
    else:
        xbin = pd.cut(xcol,bins,labels=False,include_lowest=True)

    nonemptybins = xbin.unique() # will ignore empty bins

    if binmax is None:
        binmax = xbin.value_counts().max()

    if isinstance(bins,int):
        nbins = bins
    else:
        nbins = len(bins) - 1

    if bincols > 1:
        binmax = ceil(binmax / bincols)
        plotwidth = (bincols + 1) * thumb * nbins
    else:
        plotwidth = thumb * nbins

    if coordinates=='cartesian':
        plotheight = thumb * binmax
        canvas = Image.new('RGB',(plotwidth,plotheight),bg)
    elif coordinates=='polar':
        if flip==True:
            raise ValueError("If 'flip' is true, 'coordinates' must be 'cartesian'")
        canvas = Image.new('RGB',(binmax*2*thumb+thumb,binmax*2*thumb+thumb),bg)

    for binlabel in nonemptybins:
        if ycol is not None:
            ycol_bin = ycol[xbin==binlabel]
            ycol_bin = ycol_bin.sort_values(ascending=ascending)
            pathcol_bin = pathcol.loc[ycol_bin.index]
        elif ycol is None:
            pathcol_bin = pathcol[xbin==binlabel]

        n = len(pathcol_bin)

        if coordinates=='cartesian':
            if bincols > 1:
                w,h,coords = _gridcoordsup(n,bincols,thumb)
                binbar = Image.new('RGB',(w,h),bg)
                _paste(pathcol_bin,thumb,idx,binbar,coords,notecol=notecol,flip=flip,dot=dot)

                xbar = binlabel * (bincols + 1) * thumb
                ybar = plotheight - h
                canvas.paste(binbar,(xbar,ybar))
            else:
                coords = _histcoordscart(n,binlabel,plotheight,thumb)
                _paste(pathcol_bin,thumb,idx,canvas,coords,coordinates,
                       notecol=notecol,flip=flip,dot=dot)
        elif coordinates=='polar':
            coords,phis = _histcoordspolar(n,binlabel,binmax,nbins,thumb)
            _paste(pathcol_bin,thumb,idx,canvas,coords,coordinates,phis,
                   notecol=notecol,dot=dot)

    if facetcol is None:
        if flip==True:
            return canvas.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM) # note that a flipped canvas cannot have axis labels
        else:
            if any([xaxis is not None,border is not None]):
                canvas = _facetmat(canvas,bg=bg,xaxis=xaxis,yaxis=yaxis,
                                   plottype='histogram',xtitle=xcol.name,
                                   ytitle='count',axislines=axislines,
                                   xdomain=xdomain,binmax=binmax)

            return canvas

    elif facetcol is not None:
        matdict = {'bg':bg,
                   'facettitle':facettitle,
                   'xaxis':xaxis,
                   'yaxis':yaxis,
                   'plottype':'histogram',
                   'xtitle':xcol.name,
                   'ytitle':'count',
                   'axislines':axislines,
                   'xdomain':xdomain,
                   'binmax':binmax}

        return canvas,matdict

#-------------------------------------------------------------------------------

def _scatter(xcol=None,
             ycol=None,
             pathcol=None,
             thumb=None,
             side=None,
             sample=None,
             idx=None,
             xdomain=None,
             ydomain=None,
             xbins=None,
             ybins=None,
             bg=None,
             coordinates=None,
             facetcol=None,
             facettitle=None,
             xaxis=None,
             yaxis=None,
             notecol=None,
             dot=None,
             border=None,
             title=None,
             axislines=None):

    if xbins is not None:
        xcol = _bin(xcol,xbins)
    if ybins is not None:
        ycol = _bin(ycol,ybins)

    if isinstance(side,(list,tuple)):
        if coordinates!='cartesian':
            raise TypeError("If sides unequal, 'coordinates' must be 'cartesian'")
    elif isinstance(side,int_types):
        side = (side,side)

    canvas = Image.new('RGB',side,bg) # fixed size

    # xdomain and ydomain only active at this stage if expanding
    if coordinates=='cartesian':
        x,y = _scalecart(xcol,ycol,xdomain,ydomain,side,thumb)
        coords = list(zip(x,y)) # py3 zip
        _paste(pathcol,thumb,idx,canvas,coords,coordinates,notecol=notecol,dot=dot)
    elif coordinates=='polar':
        x,y,phis = _scalepol(xcol,ycol,xdomain,ydomain,side,thumb)
        coords = list(zip(x,y)) # py3 zip
        _paste(pathcol,thumb,idx,canvas,coords,coordinates,phis,notecol=notecol,dot=dot)

    if facetcol is None:
        if any([xaxis is not None,border is not None]):
            canvas = _facetmat(canvas,bg=bg,xaxis=xaxis,yaxis=yaxis,
                              plottype='scatter',xtitle=xcol.name,
                              ytitle=ycol.name,axislines=axislines,
                              xdomain=xdomain,ydomain=ydomain)

        return canvas

    elif facetcol is not None:
        matdict = {'bg':bg,
                   'facettitle':facettitle,
                   'xaxis':xaxis,
                   'yaxis':yaxis,
                   'plottype':'scatter',
                   'xtitle':xcol.name,
                   'ytitle':ycol.name,
                   'axislines':axislines,
                   'xdomain':xdomain,
                   'ydomain':ydomain}

        return canvas,matdict

#-------------------------------------------------------------------------------

def _facetcompose(*args,border=None,bg=None):

    # item[0] in each arg is the Image; item[1] is matdict
    thumb = max( _getsizes([item[0] for item in args]) )
    for arg in args:
        arg[0].thumbnail((thumb,thumb),Image.Resampling.LANCZOS)

    # below is necessary because hist facets can be different heights
    maxheight = max([item[0].height for item in args])

    """
    If plottype == scatter, all facets have the same widths and heights.
    If plottype == histogram, all facet widths are the same, but heights can differ.
    If plottype == montage, both heights and widths can differ.

    In the loop below, plots are same-sized (set to the largest),and then matted
    (where titles, axes are added). Another loop composes them into a single plot.
    """

    mattedfacets = []
    for arg in args:
        canvas = arg[0]
        matdict = arg[1]

        if all([matdict['plottype']=='histogram',canvas.height < maxheight]):
            maxtemplate = Image.new('RGB',(canvas.width,maxheight),bg)
            maxtemplate.paste(canvas,(0,maxheight-canvas.height))
            mattedfacets.append(_facetmat(maxtemplate,**matdict))
        elif all([matdict['plottype']=='montage',any([canvas.width < thumb, canvas.height < thumb])]):
            maxtemplate = Image.new('RGB',(thumb,thumb),bg)
            halfwdiff = int( (thumb - canvas.width) / 2 )
            halfhdiff = int( (thumb - canvas.height) / 2 )
            maxtemplate.paste(canvas,(halfwdiff,halfhdiff))
            mattedfacets.append(_facetmat(maxtemplate,**matdict))
        else:
            mattedfacets.append(_facetmat(canvas,**matdict))

    n = len(args)
    ncols = _round(sqrt(n),direction='down')
    w,h,coords = _gridcoords(n,ncols,mattedfacets[0].size) # any facet in the list is fine, all same
    metacanvas = Image.new('RGB',(w,h),bg)

    for i in range(n):
        canvas = mattedfacets[i]

        if border:
            canvas = _border(canvas,bg)

        metacanvas.paste(canvas,coords[i])

    return metacanvas

#-------------------------------------------------------------------------------

def _islight(color):

    if color is None:
        return True
    
    if isinstance(color,str):
        color = ImageColor.getrgb(color)

    return sum(color) > 382.5

def _facetmat(im,
         bg=None,
         facettitle=None,
         xaxis=None,
         yaxis=None,
         plottype=None,
         axislines=None,
         xtitle=None,
         ytitle=None,
         xdomain=None,
         ydomain=None,
         binmax=None):

    if axislines:
        im = _border(im,bg)

    font,pt,fontHeight = _titlesize(im)

    # if bg is light, textcolor is dark
    if _islight(bg):
        textcolor = 'black'
    else:
        textcolor = 'white'

    """
    Note below that axis titles are added automatically when and only when axes
    are specified. One possibility this rules out is simply specifying axis
    titles without also printing ticks and labels. My current position is that
    if this is all you need, you can just put 'Y by X' as the title of the plot,
    but I could see changing this in the future.
    """

    if xaxis is not None:
        if all([yaxis is None,plottype=='scatter']):
            raise ValueError("If 'xaxis' is not None, 'yaxis' cannot be None")
        elif all([yaxis is None,plottype=='histogram']):
            im = _axes(im,xaxis,6,pt,fontHeight,xtitle,ytitle,bg,textcolor,xdomain,binmax=binmax) # 4 bin count ticks if none specified
        else:
            im = _axes(im,xaxis,yaxis,pt,fontHeight,xtitle,ytitle,bg,textcolor,xdomain,ydomain=ydomain,binmax=binmax)

    if facettitle is not None:
        im = _entitle(im,facettitle,font,fontHeight,bg)

    return im

def _axes(im,xaxis,yaxis,pt,fontHeight,xtitle,ytitle,bg,textcolor,xdomain,ydomain=None,binmax=None):

    boxSize = fontHeight*2
    imsize = im.size
    ax = Image.new('RGB',(imsize[0]+boxSize*2,imsize[1]+boxSize*2),bg)
    ax.paste(im,(boxSize*2,0))

    # ticks
    xbox = Image.new('RGB',(imsize[0],boxSize),bg)
    ybox = Image.new('RGB',(boxSize,imsize[1]),bg)

    xboxdraw = ImageDraw.Draw(xbox)
    yboxdraw = ImageDraw.Draw(ybox)

    xticklocs = [int(item) for item in linspace(0,imsize[0],xaxis)][1:-1] # cut off endpoints
    for xtick in xticklocs:
        xboxdraw.line([(xtick,0),(xtick,int(boxSize/8))],fill=textcolor)

    yticklocs = [int(item) for item in linspace(0,imsize[1],yaxis)][1:-1] # cut off endpoints
    for ytick in yticklocs:
        yboxdraw.line([(boxSize,ytick),(boxSize-int(boxSize/8),ytick)],fill=textcolor)

    # ticklabels
    tickLabelFont = ImageFont.truetype(os.path.expanduser("~") + "/fonts/Roboto-Light.ttf",int(pt * 0.67))
    
    xmin = xdomain[0]
    xmax = xdomain[1]
    xlabels = [str(round(item,label_round(xmax))) for item in linspace(xmin,xmax,xaxis)][1:-1]
    for i,xlabel in enumerate(xlabels):
        xtick = xticklocs[i]
        bbox = tickLabelFont.getbbox(xlabel)
        labelFontWidth,labelFontHeight = getfontsize(bbox)
        xboxdraw.text((int(xtick-labelFontWidth/2),int(boxSize/8+labelFontHeight/2)),text=xlabel,font=tickLabelFont,fill=textcolor)

    if binmax:
        ylabels = [str(int(item)) for item in linspace(0,binmax,yaxis)][1:-1]
    else:
        ymin = ydomain[0]
        ymax = ydomain[1]
        ylabels = [str(round(item,label_round(ymax))) for item in linspace(ymin,ymax,yaxis)][1:-1]
    
    ylabels = list(reversed(ylabels))
    for i,ylabel in enumerate(ylabels):
        ytick = yticklocs[i]
        bbox = tickLabelFont.getbbox(ylabel)
        labelFontWidth,labelFontHeight = getfontsize(bbox)
        yboxdraw.text((boxSize-int(boxSize/8)-labelFontWidth-labelFontHeight/2,int(ytick-labelFontHeight/2)),text=ylabel,font=tickLabelFont,fill=textcolor)

    ax.paste(xbox,(boxSize*2,imsize[1]))
    ax.paste(ybox,(boxSize,0))

    # axis titles
    titleFont = ImageFont.truetype(os.path.expanduser("~") + "/fonts/Roboto-Light.ttf",pt)
    bbox_x = titleFont.getbbox(xtitle)
    bbox_y = titleFont.getbbox(ytitle)
    xAxisFontWidth,xAxisFontHeight = getfontsize(bbox_x)
    yAxisFontWidth,yAxisFontHeight = getfontsize(bbox_y)

    xlbox = Image.new('RGB',(imsize[0],boxSize),bg)
    ylbox = Image.new('RGB',(imsize[1],boxSize),bg)

    xlboxdraw = ImageDraw.Draw(xlbox)
    ylboxdraw = ImageDraw.Draw(ylbox)

    xlboxdraw.text((int(imsize[0]/2-xAxisFontWidth/2),int(xAxisFontHeight/4)),xtitle,font=titleFont,fill=textcolor)
    ylboxdraw.text((int(imsize[1]/2-yAxisFontWidth/2),int(yAxisFontHeight/4)),ytitle,font=titleFont,fill=textcolor)

    ax.paste(xlbox,(boxSize*2,imsize[1]+boxSize))

    ylbox = ylbox.rotate(90,expand=1)
    ax.paste(ylbox,(0,0))

    return ax

def _titlesize(im):

    side = max(im.size)
    pt = 0
    fontWidth = 0
    while fontWidth < side/4:
        pt+=1
        sampletext = "LANDSCAPE" # just some 9-letter word
        font = ImageFont.truetype(os.path.expanduser("~") + "/fonts/Roboto-Light.ttf",pt)
        bbox = font.getbbox(sampletext)
        fontWidth,fontHeight = getfontsize(bbox)

    return font,pt,fontHeight

def _entitle(im,title,font,fontHeight,bg):

    if _islight(bg):
        textcolor = 'black'
    else:
        textcolor = 'white'

    mat = Image.new('RGB',(im.width,im.height+fontHeight*2),bg)
    mat.paste(im,(0,fontHeight*2))
    draw = ImageDraw.Draw(mat)
    bbox = font.getbbox(title)
    titleFontWidth,titleFontHeight = getfontsize(bbox)
    draw.text((int(im.width/2-titleFontWidth/2),int(fontHeight-titleFontHeight/2)),title,font=font,fill=textcolor)

    return mat

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def label_round(lmax):
    lmax = abs(lmax)
    if int(lmax)==0:
        return 2
    else:
        if lmax / 10 >= 1:
            return None
        else:
            return 1

def _gridcoords(n,ncols,thumb):
    nrows = int( ceil( float(n) / ncols ) ) # final row may be incomplete

    if isinstance(thumb,int_types):
        item_width = thumb
        item_height = thumb
    elif isinstance(thumb,tuple):
        item_width = thumb[0]
        item_height = thumb[1]

    w,h = ncols*item_width,nrows*item_height

    xgrid = list(range(ncols)) * nrows # bc py3 range returns iterator
    ygrid = repeat(range(nrows),ncols)
    xgrid = xgrid[:n]
    ygrid = ygrid[:n]
    x = [item*item_width for item in xgrid]
    y = [item*item_height for item in ygrid]

    return w,h,list(zip(x,y)) # py3 zip

def _gridcoordsup(n,ncols,thumb):
    nrows = int( ceil( float(n) / ncols ) ) # final row may be incomplete
    w,h = ncols*thumb,nrows*thumb

    xgrid = list(range(ncols)) * nrows # bc py3 range returns iterator
    ygrid = repeat(list(reversed(range(nrows))),ncols)
    xgrid = xgrid[:n]
    ygrid = ygrid[:n]
    x = [item*thumb for item in xgrid]
    y = [item*thumb for item in ygrid]

    return w,h,list(zip(x,y)) # py3 zip

def _gridcoordscirclemax(side,thumb):
    x,y = list(range(side))*side,repeat(range(side),side) # py3 range
    gridlist = list(zip(x,y))
    maximus = (int(side/2),int(side/2)) # py3 defaults to float
    return gridlist,maximus,[(int(maximus[0]*thumb),int(maximus[1]*thumb))]

def _gridcoordscircle(n,maximus,gridlist,thumb):
    # compute distances from center; sort by distance
    maximusarray = np.array(maximus) # for computing distance
    dists = [np.linalg.norm(maximusarray-np.array(item)) for item in gridlist]
    tmp = pd.DataFrame({"gridlist":gridlist,"dists":dists})
    tmp.sort_values(by='dists',inplace=True) # ascending
    gridlist = tmp.gridlist.iloc[:n-1] # n-1 bc we removed maximus
    return [(int(item[0]*thumb),int(item[1]*thumb)) for item in gridlist]

def polar2cartesian(r: int, theta: int) -> tuple:
    return (r * cos(theta), r * sin(theta))

def _bin2phi(nbins,binnum):
    incr = float(360)/nbins
    return radians(incr*binnum)

def _bin2phideg(nbins,binnum):
    incr = float(360)/nbins
    return incr*binnum

def _histcoordscart(n,binlabel,plotheight,thumb):
    xcoord = thumb * binlabel
    ycoord = plotheight - thumb # bc paste loc is UPPER left corner
    ycoords = arange(ycoord,plotheight-thumb*(n+1),-thumb)
    return [tuple((xcoord,item)) for item in ycoords]

def _histcoordspolar(n,binlabel,binmax,nbins,thumb):
    rhos = arange(binmax,binmax-n-1,-1)
    phi = _bin2phi(nbins,binlabel)
    phis = repeat(_bin2phideg(nbins,binlabel),n)
    xycoords = [_polar2cartesian(rho,phi) for rho in rhos]
    x = [int((item[0]+binmax)*thumb) for item in xycoords]
    y = [int((binmax-item[1])*thumb) for item in xycoords]
    return list(zip(x,y)),phis # py3 zip

def _scalecart(xcol,ycol,xdomain,ydomain,side,thumb):
    xcolpct = _pct(xcol,xdomain)
    ycolpct = _pct(ycol,ydomain)
    xpasterange = side[0] - thumb # otherwise will cut off extremes
    ypasterange = side[1] - thumb # otherwise will cut off extremes
    x = [int(item*xpasterange) for item in xcolpct]
    y = [int((1-item)*ypasterange) for item in ycolpct]
    return x,y

def _scalepol(xcol,ycol,xdomain,ydomain,side,thumb):
    # get percentiles for col values
    xcolpct = _pct(xcol,xdomain)
    ycolpct = _pct(ycol,ydomain)
    # derive polar coordinates from percentiles and 360 degree std
    rhos = [item for item in xcolpct] # unit radius
    phis = [item*float(360) for item in ycolpct]
    phiradians = [radians(item) for item in phis]
    # convert these to xy coordinates in (-1,1) range
    polcoords = list(zip(rhos,phiradians))
    xycoords = [_polar2cartesian(item[0],item[1]) for item in polcoords]
    # convert to canvas coordinates
    pasterange = side[0] - thumb # otherwise will cut off extremes
    radius = float(pasterange)/2
    x = [int(item[0]*radius+radius) for item in xycoords]
    y = [int(radius-item[1]*radius) for item in xycoords]
    return x,y,phis

def _pct(col,domain):
    """This will fail on missing data"""
    if domain==None:
        dmin = col.min()
        dmax = col.max()
    else:
        # if we contract domain, this is equivalent to above
        dmin = domain[0]
        dmax = domain[1]
    drange = dmax - dmin
    return [(item - dmin) / float(drange) for item in col]

def getfontsize(bbox):
    fontWidth = bbox[2] - bbox[0]
    fontHeight = bbox[3] - bbox[1]

    return fontWidth,fontHeight

def _idx(im,i):
    pos = 0 # hard-code
    draw = ImageDraw.Draw(im)

    text = str(int(i))

    fontsize = int( im.width / 28 )
    if fontsize < 10:
        fontsize = 10
    font = ImageFont.truetype(os.path.expanduser("~") + "/fonts/Roboto-Light.ttf",fontsize)
    bbox = font.getbbox(text)
    fontWidth, fontHeight = getfontsize(bbox)

    draw.rectangle(
        [(pos,pos),(pos+fontWidth,pos+fontHeight)],
        fill='white',
        outline=None
    )

    draw.text((pos,pos),text,font=font,fill='black')

def _annote(im,note):
    draw = ImageDraw.Draw(im)
    text = str(note)
    textlist = text.split('\n')
    noterows = len(textlist)
    fontsize = int( im.width / 16 )
    if fontsize < 10:
        fontsize = 10
    font = ImageFont.truetype(os.path.expanduser("~") + "/fonts/Roboto-Light.ttf",fontsize )
    maxwidthtext = max(textlist,key=len)
    bbox = font.getbbox(maxwidthtext)
    fontWidth = getfontsize(bbox)[0]
    # 4 is default line spacing in PIL multiline_text
    fontHeight = max([getfontsize(font.getbbox(item))[1] for item in textlist]) + 4

    imHeight = im.height
    pos = imHeight - (fontHeight * noterows - 4) # rm unnecessary final space

    draw.rectangle(
        [(0,pos),(fontWidth,imHeight)],
        fill='#fef7db',
        outline=None
    )

    draw.multiline_text((0,pos),text,font=font,fill='dimgrey')

def _placeholder(thumb):
    im = Image.new('RGB',(thumb,thumb),'#969696')
    draw = ImageDraw.Draw(im)
    draw.line([(0,0),(thumb,thumb)],'#dddddd')
    return im

def _dot(thumb):
    if isinstance(thumb,tuple):
        thumb = max(thumb)
    im = Image.new('RGBA',(thumb,thumb),'rgba(0,0,0,0)')
    draw = ImageDraw.Draw(im)
    incr = int( thumb / 10 )
    radius = int( thumb / 20 )
    draw.rounded_rectangle([(incr,incr),(thumb-incr,thumb-incr)],
                           radius=radius,outline=None,fill='white')
    return im

def _paste(pathcol,thumb,idx,canvas,coords,
           coordinates=None,phis=None,notecol=None,flip=None,dot=None):
    if isinstance(pathcol, string_types): # bc this is allowable in _typecheck
        raise TypeError("'pathcol' must be a pandas Series")

    counter=-1
    for i in pathcol.index:
        counter+=1
        impath = pathcol.loc[i]
        try:
            if dot==True:
                im = _dot(thumb)
            elif isinstance(impath,string_types):
                if impath.startswith(("http://", "https://")):
                    response = requests.get(impath, stream=True)
                    im = Image.open(response.raw)
                else:
                    im = Image.open(impath)
            elif isinstance(impath,Image.Image):
                im = deepcopy(impath) # when pathcol is a list of PIL images
            else:
                im = _placeholder(thumb)
        except Exception as e:
            print(e)
            im = _placeholder(thumb)

        if isinstance(thumb,tuple):
            im.thumbnail((thumb[0],thumb[1]),Image.Resampling.LANCZOS)
        elif isinstance(thumb,int_types):        
            im.thumbnail((thumb,thumb),Image.Resampling.LANCZOS)
        
        im = im.convert('RGBA') # often unnecessary but for rotation and glyphs
        if idx==True: # idx labels placed after thumbnail
            _idx(im,i)
        if notecol is not None:
            note = notecol.loc[i]
            _annote(im,note)
        if flip==True:
            im = im.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
        if coordinates=='polar':
            phi = phis[counter]
            if 90 < phi < 270:
                phi = phi + 180 # avoids upside down images
            im = im.rotate(phi,expand=1) # expand so it won't clip the corners

        canvas.paste(im,coords[counter],im) # im is a mask for itself

def _round(x,direction='down'):
    if direction=='down':
        return int(x)
    elif direction=='up':
        return int(ceil(x)) # ceil returns float

def _getsizes(args):
    plotsizes = [item.size for item in args]
    return [item for sublist in plotsizes for item in sublist]

def _bottom_left_corner(im,thumb,bg):

    if bg is None:
        canvas = Image.new('RGBA',(thumb,thumb),bg)
        canvas.paste(im,(0,thumb-im.height),im)
    else:
        canvas = Image.new('RGB',(thumb,thumb),bg)
        canvas.paste(im,(0,thumb-im.height))
    
    return canvas

def _progressBar(pathcol):
    n = len(pathcol)
    breaks = [int(n * item) for item in arange(.05,1,.05)]
    pct = [str(int(item*100))+"%" for item in arange(.05,1,.05)]

    return breaks,pct
