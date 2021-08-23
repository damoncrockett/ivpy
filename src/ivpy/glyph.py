import pandas as pd
from PIL import Image,ImageDraw
from .data import check_nan, _typecheck
from .plottools import _border
import os
import numpy as np

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def draw_glyphs(df,aes,savedir,
                glyphtype='radar',
                border=True,
                mat=True,
                crosshairs=True,
                side=200,
                alpha=1.0):
    """
    User-called glyph drawing function. Really just a wrapper for all of the
    constituent glyph drawing functions, of which there is currently only one.
    Currently requires 'df' as an argument, although I will probably make it
    possible to use the attached dataframe as well. That will require a new
    'filter' function, '_framefilter'.
    """

    _typecheck(**locals())

    try:
        os.mkdir(savedir)
    except:
        pass

    if glyphtype=='radar':
        gpaths = _draw_radar(df,aes,savedir,border,mat,crosshairs,side,alpha)

    return gpaths

#-------------------------------------------------------------------------------

def _draw_radar(df,aes,savedir,border,mat,crosshairs,side,alpha):

    alphargb = int(alpha*255)

    colors = [(40,109,192,alphargb),
              (141,168,67,alphargb),
              (189,83,25,alphargb),
              (99,170,255,alphargb),
              (255,191,0,alphargb),
              (58,14,88,alphargb),
              (201,240,127,alphargb)]

    color = aes.get('color')
    if color is not None:
        numcolors = len(df[color].value_counts().keys())
        if numcolors > len(colors):
            raise ValueError("Number of color categories cannot exceed 7")
        else:
            keys = list(df[color].value_counts().keys())
            vals = colors[:len(keys)]
            cmap = dict(zip(keys,vals))

    basename = aes.get('basename')
    left = aes.get('left')
    top = aes.get('top')
    right = aes.get('right')
    bottom = aes.get('bottom')
    lthumb = aes.get('lthumb')
    rthumb = aes.get('rthumb')
    topflag = aes.get('topflag')
    rightflag = aes.get('rightflag')

    gpaths = []
    for i in df.index:
        try:
            basename_i = df[basename].loc[i]
            left_i = df[left].loc[i]
            top_i = df[top].loc[i]
            right_i = df[right].loc[i]
        except:
            raise ValueError("""'aes' must contain 'basename', 'left', 'top',
            and 'right'""")

        if bottom is not None:
            bottom_i = df[bottom].loc[i]

        polypts = [left_i,top_i,right_i,bottom_i]

        try:
            colorval = df[color].loc[i]
            if colorval is None:
                c = (51,51,51,alphargb)
            elif check_nan(colorval): # for NaN values in color column
                c = (51,51,51,alphargb)
            elif not check_nan(colorval):
                c = cmap[colorval]
        except:
            c = colors[0]

        glyph = _radar(polypts,crosshairs,c,side)

        if border:
            glyph = _border(glyph,width=round(side/200))
        if mat:
            glyph = _mat(glyph)

        if lthumb is not None:
            lthumb_i = df[lthumb].loc[i]
            if not check_nan(lthumb_i):
                try:
                    lthumb_i = Image.open(lthumb_i)
                    glyph = add_thumb(glyph,lthumb_i,'left')
                except:
                    pass
        if rthumb is not None:
            rthumb_i = df[rthumb].loc[i]
            if not check_nan(rthumb_i):
                try:
                    rthumb_i = Image.open(rthumb_i)
                    glyph = add_thumb(glyph,rthumb_i,'right')
                except:
                    pass

        """
        Currently the flag settings are not generalized. Here, the top flag
        indicates a color measurement using mode M2; the gloss flag indicates
        that the gloss measurement was taken at dmin or dmax, rather than dmid.
        """
        if topflag is not None:
            topflag_i = df[topflag].loc[i]
            if not check_nan(topflag_i):
                glyph = add_flag(glyph,'top',outline='white',fill=None)
        if rightflag is not None:
            rightflag_i = df[rightflag].loc[i]
            if not check_nan(rightflag_i):
                if rightflag_i=='dmin':
                    fill = 'white'
                elif rightflag_i=='dmax':
                    fill = 'black'
                glyph = add_flag(glyph,'right',outline=None,fill=fill)

        savestring = savedir + str(basename_i) + ".png"
        glyph.save(savestring)
        gpaths.append(savestring)

    return gpaths

def _radar(polypts,crosshairs,fill,
          side=200,
          outline='black',
          crosshairfill='black'):

    """
    Function where the basic radar glyph is drawn. Not meant to be user-called,
    though that could change as the glyph module develops.
    """

    im = Image.new('RGBA', (side,side), None)
    draw = ImageDraw.Draw(im)
    adj = int( side / 20 )
    halfside = int( side / 2 )
    coords = _radarcoords(polypts,halfside)

    if len(coords)==1:
        x,y = list(coords)[0][0], list(coords)[0][1]
        draw.ellipse([(x-adj,y-adj),(x+adj,y+adj)], fill=fill)
    elif len(coords)==2:
        if any([list(coords.index)==[0,2],list(coords.index)==[1,3]]):
            draw.line(list(coords), fill=fill, width=adj*2)
        else:
            coords = list(coords)
            coords.append((halfside,halfside))
            draw.polygon(coords, fill=fill, outline=outline)
    elif len(coords) > 2:
        draw.polygon(list(coords), fill=fill, outline=outline)

    if crosshairs:
        draw.line([(halfside,0),(halfside,side)],
                  fill=crosshairfill,
                  width=round(side/200))
        draw.line([(0,halfside),(side,halfside)],
                  fill=crosshairfill,
                  width=round(side/200))

    return im

def _radarcoords(polypts,halfside):
    assert len(polypts) > 2

    left = _radar_axis_position(False,False,polypts[0],halfside)
    top = _radar_axis_position(True,False,polypts[1],halfside)
    right = _radar_axis_position(False,True,1-polypts[2],halfside)
    bottom = _radar_axis_position(True,True,polypts[3],halfside)

    coords = pd.Series([left,top,right,bottom])
    coords = coords[coords.notnull()]

    return coords

def _radar_axis_position(vert,pos,val,halfside):
    if check_nan(val):
        return None
    elif not check_nan(val):
        prop = val * halfside
        if pos:
            extent = prop + halfside
        elif not pos:
            extent = halfside - prop
        if vert:
            return (halfside, extent)
        elif not vert:
            return (extent, halfside)

#-------------------------------------------------------------------------------

def _mat(im):
    w = im.width
    h = im.height

    w_incr = int( w * 0.1 )
    h_incr = int( h * 0.1 )

    newim = Image.new('RGBA',(w+w_incr*2,h+h_incr*2),None)
    newim.paste(im,(w_incr,h_incr),im)

    return newim

def add_thumb(im,thumb,corner):
    w = im.width
    thumb_side = int(w/4)
    thumb.thumbnail((thumb_side,thumb_side),Image.ANTIALIAS)

    if corner=='left':
        coords = (0,0)
    elif corner=='right':
        coords = (w-thumb_side,0)
    im.paste(thumb,coords)

    return im

def add_flag(im,loc,outline,fill):
    w = im.width
    h = im.height
    halfh = int(h/2)
    halfw = int(w/2)
    flag_adj = int( w / 50 )
    incr = int( flag_adj / 2)
    draw = ImageDraw.Draw(im)

    if loc=='right':
        flag_loc = [
            (w-flag_adj*2-incr,halfh-flag_adj),
            (w-incr,halfh+flag_adj)
        ]
    elif loc=='top':
        flag_loc = [
            (halfw-flag_adj,incr*2),
            (halfw+flag_adj,flag_adj*2+incr*2)
        ]

    draw.ellipse(flag_loc,outline=outline,fill=fill)
    return im
