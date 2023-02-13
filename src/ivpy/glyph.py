import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from .data import check_nan, _typecheck
import os
import numpy as np

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def draw_glyphs(df,aes,savedir,
                glyphtype='radar',
                gridlines=True,
                mat=True,
                radii=True,
                side=200,
                alpha=1.0,
                legend=True,
                savecolor=True,
                outline=None):
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
        gpaths = _draw_radar(df,aes,savedir,gridlines,mat,radii,side,alpha,legend,savecolor,outline)

    return gpaths

#-------------------------------------------------------------------------------

def _legend_entry(c,val,font,w,h,spacer_width):
    """
    Draw a single legend entry.
    """

    entry = Image.new('RGBA',(w,h),None)
    draw = ImageDraw.Draw(entry)

    draw.rounded_rectangle([(int(h/6),int(h/6)),(h-int(h/16),h-int(h/16))],fill=c,radius=int(h/8),outline='black',width=5)
    draw.text((h+spacer_width,0),val.upper(),fill='grey',font=font)
    
    return entry

def _legend(cmap):
    """
    Draw a legend for the radar glyphs.
    """

    font = ImageFont.truetype(os.path.expanduser("~") + "/fonts/Roboto-Light.ttf", 144)
    maxwidth = max([font.getsize(val.upper())[0] for val in cmap.keys()])
    maxheight = max([font.getsize(val.upper())[1] for val in cmap.keys()])
    vincr = int( maxheight * 0.1)
    spacer_width = 20
    entry_height = maxheight + vincr # PIL font height estimate can be short, cutting off bottom
    entry_width = maxwidth + spacer_width + entry_height + vincr # PIL font width estimate can be short, cutting off right side


    entries = []
    for key in cmap.keys():
        im = _legend_entry(cmap[key],key,font,entry_width,entry_height,spacer_width)
        entries.append(im)

    # here, we are using 'vincr' for a different purpose: to add vertical space between entries
    legend = Image.new('RGBA',(entry_width,(entry_height+vincr*2)*len(entries)),None)
    
    for i,im in enumerate(entries):
        legend.paste(im,(0, i * (entry_height+vincr*2)),im)

    return _mat(legend,rgba=False)

def get_legend(df,col):

    glyph_pathlist = list(df[col])
    
    if len(set([os.path.dirname(path) for path in glyph_pathlist])) > 1:
        print("Glyphs are not all in the same directory; retrieving legend from first glyph")

    legend_path = os.path.join(os.path.dirname(glyph_pathlist[0]),'_legend.png')
    
    if not os.path.exists(legend_path):
        raise ValueError("Legend does not exist; glyph legends are drawn only if 'color' is set in 'aes'")
    
    return Image.open(legend_path)


def _draw_radar(df,aes,savedir,gridlines,mat,radii,side,alpha,legend,savecolor,outline):

    alphargb = int(alpha*255)

    colors = [(40,109,192,alphargb),
              (141,168,67,alphargb),
              (189,83,25,alphargb),
              (99,170,255,alphargb),
              (255,191,0,alphargb),
              (58,14,88,alphargb),
              (201,240,127,alphargb),
              (124,70,160,alphargb),
              (58,166,9,alphargb),
              (223,42,105,alphargb)]

    color = aes.get('color')
    if color is not None:
        numcolors = len(df[color].value_counts().keys())
        if numcolors > len(colors):
            lendiff = numcolors - len(colors)
            dummycolors = [(0,0,0,alphargb) for i in range(lendiff)]
            colors = colors + dummycolors

            print("Warning: more than 10 colors specified; groups outside of the 10 largest will be drawn in black.")

        keys = list(df[color].value_counts().keys())
        vals = colors[:len(keys)]
        cmap = dict(zip(keys,vals))
        
        if legend:
            _legend(cmap).save(os.path.join(savedir,'_legend.png'))

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
    gpath_colors = []
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
                c = (160,160,160,alphargb)
            elif check_nan(colorval): # for NaN values in color column
                c = (160,160,160,alphargb)
            elif not check_nan(colorval):
                c = cmap[colorval]
        except:
            #c = colors[0]
            c = (0,0,0,alphargb)

        glyph = _radar(polypts,radii,gridlines,c,outline,side)

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

        savestring = os.path.join(savedir, str(basename_i) + ".png")
        glyph.save(savestring)
        gpaths.append(savestring)

        if savecolor:
            gpath_colors.append(c)

    if savecolor:
        return pd.DataFrame({'gpath':gpaths,'gpath_color':gpath_colors},index=df.index)
    else:
        return pd.Series(gpaths,index=df.index)

def _radar(polypts,radii,gridlines,radarfill,outline,side=200,radiifill='grey'):

    """
    Function where the basic radar glyph is drawn. Not meant to be user-called,
    though that could change as the glyph module develops.
    """

    im = Image.new('RGBA', (side,side), None)
    draw = ImageDraw.Draw(im)
    
    adj = int( side / 20 )
    halfside = int( side / 2 )

    # plot lines first, below the glyph

    if radii:
        draw.line([(halfside,0),(halfside,side)],
                  fill=radiifill,
                  width=round(side/200))
        draw.line([(0,halfside),(side,halfside)],
                  fill=radiifill,
                  width=round(side/200))

    if gridlines:

        sSixth = side / 6;
        sThird = side / 3;
        sHalf = side / 2;
        sTwoThird = side * 2/3;
        sFiveSixth = side * 5/6;

        draw.line([(sHalf,0),(side,sHalf)],fill=radiifill,width=round(side/200))
        draw.line([(side,sHalf),(sHalf,side)],fill=radiifill,width=round(side/200))
        draw.line([(sHalf,side),(0,sHalf)],fill=radiifill,width=round(side/200))
        draw.line([(0,sHalf),(sHalf,0)],fill=radiifill,width=round(side/200))

        draw.line([(sHalf,sSixth),(sFiveSixth,sHalf)],fill=radiifill,width=round(side/200))
        draw.line([(sFiveSixth,sHalf),(sHalf,sFiveSixth)],fill=radiifill,width=round(side/200))
        draw.line([(sHalf,sFiveSixth),(sSixth,sHalf)],fill=radiifill,width=round(side/200))
        draw.line([(sSixth,sHalf),(sHalf,sSixth)],fill=radiifill,width=round(side/200))

        draw.line([(sHalf,sThird),(sTwoThird,sHalf)],fill=radiifill,width=round(side/200))
        draw.line([(sTwoThird,sHalf),(sHalf,sTwoThird)],fill=radiifill,width=round(side/200))
        draw.line([(sHalf,sTwoThird),(sThird,sHalf)],fill=radiifill,width=round(side/200))
        draw.line([(sThird,sHalf),(sHalf,sThird)],fill=radiifill,width=round(side/200))

    # then the glyph atop it

    coords = _radarcoords(polypts,halfside)

    if len(coords)==1:
        x,y = list(coords)[0][0], list(coords)[0][1]
        draw.ellipse([(x-adj,y-adj),(x+adj,y+adj)], fill=radarfill)
    elif len(coords)==2:
        if any([list(coords.index)==[0,2],list(coords.index)==[1,3]]):
            draw.line(list(coords), fill=radarfill, width=adj*2)
        else:
            coords = list(coords)
            coords.append((halfside,halfside))
            draw.polygon(coords, fill=radarfill, outline=outline, width=int(side/100))
    elif len(coords) > 2:
        draw.polygon(list(coords), fill=radarfill, outline=outline, width=int(side/100))

    return im

def _radarcoords(polypts,halfside):
    assert len(polypts) > 2

    left = _radar_axis_position(False,False,polypts[0],halfside)
    top = _radar_axis_position(True,False,polypts[1],halfside)
    #right = _radar_axis_position(False,True,1-polypts[2],halfside) # gloss inversion
    right = _radar_axis_position(False,True,polypts[2],halfside)
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

def _mat(im,rgba=True):
    w = im.width
    h = im.height

    w_incr = int( w * 0.1 )
    h_incr = int( h * 0.1 )

    if rgba:
        newim = Image.new('RGBA',(w+w_incr*2,h+h_incr*2),None)
    else:
        # this for some reason prevents losing rect color when you mat a legend
        newim = Image.new('RGB',(w+w_incr*2,h+h_incr*2),"white")
        
    newim.paste(im,(w_incr,h_incr),im)
    
    return newim

def add_thumb(im,thumb,corner):
    w = im.width
    thumb_side = int(w/4)
    thumb.thumbnail((thumb_side,thumb_side),Image.Resampling.LANCZOS)

    if corner=='left':
        coords = (0,0)
    elif corner=='right':
        coords = (w-thumb_side,0)
    
    # if thumb is transparent, paste it with alpha
    if thumb.mode=='RGBA':
        im.paste(thumb,coords,thumb)
    else:
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
