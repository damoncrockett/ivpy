from PIL import Image,ImageDraw,ImageFont

def _scale(col,domain,side,thumb,y=False):
    """This will fail on missing data""" 

    pinrange = side - thumb # otherwise will cut off extremes

    if domain==None:
        dmin = col.min()
        dmax = col.max()
    else:
        if not all(isinstance(domain,(list,tuple)),len(domain)==2):
            raise ValueError("'domain' must be two-item list or tuple")
        else:
            dmin = domain[0]
            dmax = domain[1]

    if y==False:
        return [ int( _pct(item,dmin,dmax) * pinrange ) for item in col]
    elif y==True:
        return [ int( (1 - _pct(item,dmin,dmax)) * pinrange ) for item in col]
    else:    
        raise ValueError("'y' must be a Boolean")

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

def _facet():
    return None