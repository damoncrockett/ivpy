import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from numpy import repeat,sqrt,arange,radians,cos,sin
from math import ceil
from shapely.geometry import Point

def _montage(pathcol=None,
             featcol=None,
             xdomain=None,
             ycol=None, # idle
             ydomain=None, # idle
             thumb=None,
             sample=None,
             idx=None,
             bg=None,
             shape=None,
             ascending=None,
             facetcol=None):

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

def _histogram(featcol=None,
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
               facetcol=None):

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
            # range is overkill but don't have great way to always avoid NaNs
            bins = arange(xdomain[0],xdomain[1]+increment*2,increment)

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
        elif ycol is None:
            pathcol_bin = pathcol[xbin==binlabel]

        n = len(pathcol_bin)

        if coordinates=='cartesian':
            coords = _histcoordscart(n,binlabel,plotheight,thumb)
            _paste(pathcol_bin,thumb,idx,canvas,coords,coordinates)
        elif coordinates=='polar':
            coords,phis = _histcoordspolar(n,binlabel,binmax,nbins,thumb)
            _paste(pathcol_bin,thumb,idx,canvas,coords,coordinates,phis)

    return canvas

def _scatter(featcol=None,
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
             facetcol=None):

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

def _gridcoords(n,ncols,thumb):
    nrows = int( ceil( float(n) / ncols ) ) # final row may be incomplete
    w,h = ncols*thumb,nrows*thumb

    xgrid = range(ncols) * nrows
    ygrid = repeat(range(nrows),ncols)
    xgrid = xgrid[:n]
    ygrid = ygrid[:n]
    x = [item*thumb for item in xgrid]
    y = [item*thumb for item in ygrid]

    return w,h,zip(x,y)

def _gridcoordscirclemax(side,thumb):
    x,y = range(side)*side,repeat(range(side),side)
    gridlist = [Point(item) for item in zip(x,y)]
    maximus = Point(side/2,side/2)
    return gridlist,maximus,[(int(maximus.x*thumb),int(maximus.y*thumb))]

def _gridcoordscircle(n,maximus,gridlist,thumb):
    # compute distances from center; sort by distance
    dists = [maximus.distance(item) for item in gridlist]
    tmp = pd.DataFrame({"gridlist":gridlist,"dists":dists})
    tmp.sort_values(by='dists',inplace=True) # ascending
    gridlist = tmp.gridlist.iloc[:n-1] # n-1 bc we removed maximus
    return [(int(item.x*thumb),int(item.y*thumb)) for item in gridlist]

def _pol2cart((rho,phi)):
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
    xycoords = [_pol2cart((rho,phi)) for rho in rhos]
    x = [int((item[0]+binmax)*thumb) for item in xycoords]
    y = [int((binmax-item[1])*thumb) for item in xycoords]
    return zip(x,y),phis

def _scalecart(featcol,ycol,xdomain,ydomain,side,thumb):
    featcolpct = _pct(featcol,xdomain)
    ycolpct = _pct(ycol,ydomain)
    pasterange = side - thumb # otherwise will cut off extremes
    x = [int(item*pasterange) for item in featcolpct]
    y = [int((1-item)*pasterange) for item in ycolpct]
    return x,y

def _scalepol(featcol,ycol,xdomain,ydomain,side,thumb):
    # get percentiles for col values
    featcolpct = _pct(featcol,xdomain)
    ycolpct = _pct(ycol,ydomain)
    # derive polar coordinates from percentiles and 360 degree std
    rhos = [item for item in featcolpct] # unit radius
    phis = [item*float(360) for item in ycolpct]
    phiradians = [radians(item) for item in phis]
    # convert these to xy coordinates in (-1,1) range
    xycoords = [_pol2cart(item) for item in zip(rhos,phiradians)]
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
    font = ImageFont.truetype('../fonts/VeraMono.ttf', 8)
    fontWidth, fontHeight = font.getsize(text)

    try:
        draw.rectangle(
            [(pos,pos),(pos+fontWidth,pos+fontHeight)],
            fill='#282828',
            outline=None
        )

        draw.text((pos,pos),text,font=font,fill='#efefef')

    except Exception as e:
        print e

def _placeholder(thumb):
    im = Image.new('RGB',(thumb,thumb),'#969696')
    draw = ImageDraw.Draw(im)
    draw.line([(0,0),(thumb,thumb)],'#dddddd')
    return im

def _paste(pathcol,thumb,idx,canvas,coords,coordinates=None,phis=None):
    counter=-1
    for i in pathcol.index:
        counter+=1
        try:
            im = Image.open(pathcol.loc[i])
        except:
            im = _placeholder(thumb)
        im.thumbnail((thumb,thumb),Image.ANTIALIAS)
        if idx==True: # idx labels placed after thumbnail
            _idx(im,i)
        if coordinates=='polar':
            phi = phis[counter]
            if 90 < phi < 270:
                phi = phi + 180 # avoids upside down images
            im = im.convert('RGBA') # need alpha layer to make rotation work
            im = im.rotate(phi,expand=1) # expand so it won't clip the corners
            canvas.paste(im,coords[counter],im) # im is a mask for itself
        else:
            canvas.paste(im,coords[counter])

def _round(x,direction='down'):
    if direction=='down':
        return int(x)
    elif direction=='up':
        return int(ceil(x)) # ceil returns float

def _getsizes(args):
    plotsizes = [item.size for item in args]
    return [item for sublist in plotsizes for item in sublist]

def _outline(im):
    draw = ImageDraw.Draw(im)
    # n.b.: lines are drawn under and to the right of the starting pixel
    draw.line([(0,0),(im.width,0)],'#dddddd',width=1)
    draw.line([(im.width-1,0),(im.width-1,im.height)],'#dddddd',width=1)
    draw.line([(im.width,im.height-1),(0,im.height-1)],'#dddddd',width=1)
    draw.line([(0,im.height),(0,0)],'#dddddd',width=1)

    return im
