from PIL import Image,ImageDraw,ImageFont
from numpy import repeat
from math import ceil

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
        return [int( _pct(item,dmin,dmax) * pinrange ) for item in col]
    elif y==True:
        return [int( (1 - _pct(item,dmin,dmax)) * pinrange ) for item in col]
    else:
        raise TypeError("'y' must be a Boolean")

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

def _paste(pathcol,thumb,idx,canvas,coords):
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
        canvas.paste(im,coords[counter])

def _round(x,direction='down'):
    if direction=='down':
        return int(x)
    elif direction=='up':
        return int(ceil(x)) # ceil returns float

def _getsizes(args):
    plotsizes = [item.size for item in args]
    return [item for sublist in plotsizes for item in sublist]
