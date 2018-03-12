from PIL import Image,ImageDraw,ImageFont
from numpy import sqrt,repeat
from math import ceil
from copy import deepcopy

def _scale(col,domain,side,thumb,y=False):
    """This will fail on missing data"""

    pinrange = side - thumb # otherwise will cut off extremes

    if domain==None:
        dmin = col.min()
        dmax = col.max()
    else:
        if not all(isinstance(domain,(list,tuple)),len(domain)==2):
            raise TypeError("'domain' must be two-item list or tuple")
        else:
            dmin = domain[0]
            dmax = domain[1]

    if y==False:
        return [ int( _pct(item,dmin,dmax) * pinrange ) for item in col]
    elif y==True:
        return [ int( (1 - _pct(item,dmin,dmax)) * pinrange ) for item in col]
    else:
        raise TypeError("'y' must be a Boolean")

def _pct(item,dmin,dmax):
    drange = dmax - dmin
    return (item - dmin) / float(drange)

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

def _round(x,direction='down'):
    if direction=='down':
        return int(x)
    elif direction=='up':
        return int(ceil(x)) # ceil returns float

def compose(*args,**kwargs):

    """
    Composes canvases into metacanvas

    Args:
        *args --- any number of canvases, given by name or plot function
        ncols (int) --- number of columns in metacanvas (optional)
        rounding (str) --- when ncols is None, round ncols 'up' or 'down'
        thumb (int) --- pixel value for thumbnail side
        bg (color) --- background color
    """

    typelist = [isinstance(item, Image.Image) for item in args]
    if not all(typelist):
        raise TypeError("Arguments passed to 'compose' must be PIL Images")

    n = len(args)

    try:
        ncols = kwargs['ncols']
    except:
        try:
            rounding = kwargs['rounding']
        except:
            rounding = 'up'
        ncols = _round(sqrt(n),direction=rounding)
    finally:
        if not isinstance(ncols, int):
            raise TypeError("'ncols' must be an integer")
        if ncols > n:
            raise ValueError("'ncols' cannot be larger than number of plots")

    nrows = int( ceil( float(n) / ncols ) ) # final row may be incomplete
    xgrid = range(ncols) * nrows
    ygrid = repeat(range(nrows),ncols)

    xgrid = xgrid[:n]
    ygrid = ygrid[:n]

    try:
        side = kwargs['thumb']
    except:
        plotsizes = [item.size for item in args]
        plotwidths = [item[0] for item in plotsizes]
        plotheights = [item[1] for item in plotsizes]
        side = max(max(plotwidths),max(plotheights))

    px_w = ncols * side
    px_h = nrows * side

    try:
        bg = kwargs['bg']
    except:
        bg = '#4a4a4a'

    metacanvas = Image.new('RGB',(px_w,px_h),bg)
    for i in range(n):
        canvas = args[i]
        tmp = deepcopy(canvas) # copy because thumbnail always inplace
        tmp.thumbnail((side,side),Image.ANTIALIAS)
        x = xgrid[i] * side
        y = ygrid[i] * side
        metacanvas.paste(tmp,(x,y))

    return metacanvas
