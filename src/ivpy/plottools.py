import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from numpy import repeat,sqrt,arange,radians,cos,sin
import numpy as np
from math import ceil
from six import string_types
import requests

from .data import _bin, _typecheck

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def _border(im,fill='#4a4a4a',width=1):
    _typecheck(**locals())
    draw = ImageDraw.Draw(im)
    # n.b.: lines are drawn under and to the right of the starting pixel
    draw.line([(0,0),(im.width,0)],fill=fill,width=width)
    draw.line([(im.width-1,0),(im.width-1,im.height)],fill=fill,width=width)
    draw.line([(im.width,im.height-1),(0,im.height-1)],fill=fill,width=width)
    draw.line([(0,im.height),(0,0)],fill=fill,width=width)

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
             notecol=None):

    n = len(pathcol)

    if shape=='square':
        ncols = int(sqrt(n))
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
               notecol=None):

    """
    If user submitted bin sequence leaves out some rows, user must pass xdomain
    argument, because data subsetting not allowable this late in the process.
    """
    if isinstance(bins,(list,tuple,np.ndarray)):
        if any([xcol.max()>bins[-1],xcol.min()<bins[0]]):
            raise ValueError("""Submitted bin edges do not capture all the data.
                                Domain contraction requires 'xdomain' argument,
                                and bin edges must span the domain.
                                """)

    """
    This (below) is domain expansion. The histogram ydomain can be contracted;
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
        xrange = xdomain[1]-xdomain[0]
        if isinstance(bins,int):
            # n.b.: this is slightly different than giving int to pd.cut
            increment = float(xrange)/bins
            # range is overkill but don't have great way to always avoid NaNs
            # will probably need to revisit this in the future
            bins = arange(xdomain[0],xdomain[1]+increment*2,increment)

    xbin = pd.cut(xcol,bins,labels=False,include_lowest=True)
    nbins = len(pd.cut(xcol,bins,include_lowest=True).value_counts())
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
        elif ycol is None:
            pathcol_bin = pathcol[xbin==binlabel]

        n = len(pathcol_bin)

        if coordinates=='cartesian':
            coords = _histcoordscart(n,binlabel,plotheight,thumb)
            _paste(pathcol_bin,thumb,idx,canvas,coords,coordinates,
                   notecol=notecol)
        elif coordinates=='polar':
            coords,phis = _histcoordspolar(n,binlabel,binmax,nbins,thumb)
            _paste(pathcol_bin,thumb,idx,canvas,coords,coordinates,phis,
                   notecol=notecol)

    if facetcol is None:
        if xaxis is not None:
            canvas = _plotmat(canvas,
                          bg=bg,
                          facetcol=facetcol,
                          xaxis=xaxis,
                          plottype='histogram')

        return canvas

    elif facetcol is not None:
        matdict = {'bg':bg,
                   'facettitle':facettitle,
                   'xaxis':xaxis,
                   'plottype':'histogram'}

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
             notecol=None):

    if xbins is not None:
        xcol = _bin(xcol,xbins)
    if ybins is not None:
        ycol = _bin(ycol,ybins)

    canvas = Image.new('RGB',(side,side),bg) # fixed size

    # xdomain and ydomain only active at this stage if expanding
    if coordinates=='cartesian':
        x,y = _scalecart(xcol,ycol,xdomain,ydomain,side,thumb)
        coords = list(zip(x,y)) # py3 zip
        _paste(pathcol,thumb,idx,canvas,coords,coordinates,notecol=notecol)
    elif coordinates=='polar':
        x,y,phis = _scalepol(xcol,ycol,xdomain,ydomain,side,thumb)
        coords = list(zip(x,y)) # py3 zip
        _paste(pathcol,thumb,idx,canvas,coords,coordinates,phis,notecol=notecol)

    if facetcol is None:
        if any([xaxis is not None,yaxis is not None]):
            canvas = _plotmat(canvas,
                          bg=bg,
                          facetcol=facetcol,
                          xaxis=xaxis,
                          yaxis=yaxis)

        return canvas

    elif facetcol is not None:
        matdict = {'bg':bg,
                   'facettitle':facettitle,
                   'xaxis':xaxis,
                   'yaxis':yaxis}

        return canvas,matdict

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def _gridcoords(n,ncols,thumb):
    nrows = int( ceil( float(n) / ncols ) ) # final row may be incomplete
    w,h = ncols*thumb,nrows*thumb

    xgrid = list(range(ncols)) * nrows # bc py3 range returns iterator
    ygrid = repeat(range(nrows),ncols)
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

def _pol2cart(rho,phi):
    x = rho * cos(phi)
    y = rho * sin(phi)
    return(x,y)

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
    xycoords = [_pol2cart(rho,phi) for rho in rhos]
    x = [int((item[0]+binmax)*thumb) for item in xycoords]
    y = [int((binmax-item[1])*thumb) for item in xycoords]
    return list(zip(x,y)),phis # py3 zip

def _scalecart(xcol,ycol,xdomain,ydomain,side,thumb):
    xcolpct = _pct(xcol,xdomain)
    ycolpct = _pct(ycol,ydomain)
    pasterange = side - thumb # otherwise will cut off extremes
    x = [int(item*pasterange) for item in xcolpct]
    y = [int((1-item)*pasterange) for item in ycolpct]
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
    xycoords = [_pol2cart(item[0],item[1]) for item in polcoords]
    # convert to canvas coordinates
    pasterange = side - thumb # otherwise will cut off extremes
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

def _idx(im,i):
    pos = 0 # hard-code
    draw = ImageDraw.Draw(im)
    text = str(int(i))
    font = ImageFont.truetype('../fonts/VeraMono.ttf',12)
    fontWidth, fontHeight = font.getsize(text)

    draw.rectangle(
        [(pos,pos),(pos+fontWidth,pos+fontHeight)],
        fill='white',
        outline=None
    )

    draw.text((pos,pos),text,font=font,fill='black')

def _annote(im,note):
    draw = ImageDraw.Draw(im)
    text = str(note)
    font = ImageFont.truetype('../fonts/VeraMono.ttf',12)
    fontWidth, fontHeight = font.getsize(text)
    imHeight = im.height
    pos = imHeight - fontHeight

    draw.rectangle(
        [(0,pos),(fontWidth,imHeight)],
        fill='white',
        outline=None
    )

    draw.text((0,pos),text,font=font,fill='black')

def _placeholder(thumb):
    im = Image.new('RGB',(thumb,thumb),'#969696')
    draw = ImageDraw.Draw(im)
    draw.line([(0,0),(thumb,thumb)],'#dddddd')
    return im

def _paste(pathcol,thumb,idx,canvas,coords,
           coordinates=None,phis=None,notecol=None):
    if isinstance(pathcol, string_types): # bc this is allowable in _typecheck
        raise TypeError("'pathcol' must be a pandas Series")

    counter=-1
    for i in pathcol.index:
        counter+=1
        try:
            if pathcol.loc[i].startswith(("http://", "https://")):
              response = requests.get(pathcol.loc[i], stream=True)
              im = Image.open(response.raw)
            else:
              im = Image.open(pathcol.loc[i])
        except:
            im = _placeholder(thumb)
        im.thumbnail((thumb,thumb),Image.ANTIALIAS)
        im = im.convert('RGBA') # often unnecessary but for rotation and glyphs
        if idx==True: # idx labels placed after thumbnail
            _idx(im,i)
        if notecol is not None:
            note = notecol.loc[i]
            _annote(im,note)
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

def _plotmat(im,
         bg=None,
         facettitle=None,
         xaxis=None,
         yaxis=None,
         plottype=None):

    if im.width!=im.height:
        im = _premat(im,bg,plottype)

    #if plottype!='montage': # bc montages do not have axis boundaries
    #    im = _border(im)

    # we want a 9-letter word to span half the plot width
    pt = 0
    fontWidth = 0
    while fontWidth < im.width/2:
        pt+=1
        sampletext = "LANDSCAPE" # just some 9-letter word
        font = ImageFont.truetype('../fonts/VeraMono.ttf',pt)
        fontWidth,fontHeight = font.getsize(sampletext)

    side = im.height + fontHeight * 3 * 2 # 3 rows of text top and bottom
    mat = Image.new('RGB',(side,side),bg)
    halfwdiff = int( (side - im.width) / 2 )
    halfhdiff = int( (side - im.height) / 2 )
    mat.paste(im,(halfwdiff,halfhdiff))
    draw = ImageDraw.Draw(mat)

    if any([xaxis is not None, yaxis is not None]):
        mat = _axes(mat,draw,xaxis,yaxis)

    if facettitle is not None:
        text = facettitle
        fontWidth,fontHeight = font.getsize(text)
        draw.text((int((side-fontWidth)/2),fontHeight),text,font=font)

    return mat

def _premat(im,bg,plottype):
    side = max([im.width,im.height])
    premat = Image.new('RGB',(side,side),bg)

    if plottype=='histogram':
        premat.paste(im,(0,side-im.height))
    elif plottype=='montage':
        halfwdiff = int( (side - im.width) / 2 )
        halfhdiff = int( (side - im.height) / 2 )
        premat.paste(im,(halfwdiff,halfhdiff))

    return premat

def _axes(mat,draw,xaxis,yaxis):
    return None

def _progressBar(pathcol):
    n = len(pathcol)
    breaks = [int(n * item) for item in arange(.05,1,.05)]
    pct = [str(int(item*100))+"%" for item in arange(.05,1,.05)]

    return breaks,pct
