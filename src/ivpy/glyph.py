import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from .data import _typecheck
from .plottools import polar2cartesian
import os
from numpy import radians, arange, linspace, random

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

FONTDIR = os.path.expanduser("~") + "/fonts/"

def get_font(fonttype="Roboto-Light", fontsize=40):
    font = ImageFont.truetype(FONTDIR + fonttype + ".ttf", fontsize)

    return font

#-------------------------------------------------------------------------------

def draw_glyphs(X,
                colorcol=None,
                savedir=None,
                savecol=None,
                glyphtype='radar',
                mat=True,
                side=800,
                alpha=1.0,
                legend=True,
                **kwargs):
    
    """
    User-called glyph drawing function. Really just a wrapper for all of the
    constituent glyph drawing functions, of which there is currently only one.
    Currently requires 'df' as an argument, although I will probably make it
    possible to use the attached dataframe as well. That will require a new
    'filter' function, '_framefilter'.
    """

    _typecheck(**locals())

    if not isinstance(X,pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    
    assert all([X.min().min() >= 0, X.max().max() <= 1]), "X must be normalized to [0,1]"

    alphargb = int(alpha*255)
    
    colors = [
        (40,109,192,alphargb),
        (141,168,67,alphargb),
        (189,83,25,alphargb),
        (99,170,255,alphargb),
        (255,191,0,alphargb),
        (58,14,88,alphargb),
        (201,240,127,alphargb),
        (124,70,160,alphargb),
        (58,166,9,alphargb),
        (223,42,105,alphargb)
        ]

    if colorcol is not None:
        assert X.index.equals(colorcol.index), "colorcol must have the same index as X"
        colorkeys = list(colorcol.value_counts().keys())
        numcolors = len(colorkeys)
        if numcolors > len(colors):
            lendiff = numcolors - len(colors)
            randomcolors = [get_random_color(alphargb) for _ in range(lendiff)]
            colors = colors + randomcolors

            print("Warning: more than 10 colors specified; groups outside of the 10 largest will be drawn in random colors.")

        vals = colors[:len(colorkeys)]
        cmap = dict(zip(colorkeys,vals))

        colorcol = pd.Series([cmap[val] for val in colorcol],index=colorcol.index)
        colorcol.loc[colorcol.isnull()] = (160,160,160,alphargb)

        if legend:
            _legend(cmap).save('_legend.png')
    else:
        colorcol = pd.Series([(160,160,160,alphargb)] * len(X.index),index=X.index)

    if savedir:
        try:
            os.mkdir(savedir)
        except:
            pass

    gpaths = []
    for i in X.index:
        if glyphtype=='radar':
            vertex_list = list(X.loc[i])
            g = radar(vertex_list, colorcol.loc[i], side, mat=mat, **kwargs)

        if savedir:
            if savecol:
                savepath = os.path.join(savedir,savecol.loc[i]+'.png')
            else:
                savepath = os.path.join(savedir,str(i)+'.png')
                
            g.save(savepath)
            gpaths.append(savepath)
        else:
            gpaths.append(g)

    return gpaths

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def radar(vertex_list: list, fill=(160,160,160,255), side=800, mat=True, **kwargs) -> Image:
    """
    args:
    
    vertex_list: (list) at least 3 normalized radius positions
    fill: (tuple) polygon fill color
    side: (int) canvas pixels per side
    outline: (str) polygon outline color
    outlinewidth: (int) polygon outline width
    gridlinefill: (str) gridline color
    gridlinewidth: (int) gridline width
    radii: (bool) draw radar axes
    gridlines: (bool or int) draw radar gridlines; either False or n_ticks
    axislabels: (bool or seq) draw axis labels; either False or a sequence of labels
    mat: (bool) add padding to canvas
    
    return:
    
    canvas: (PIL.Image or filepath) radar glyph or filepath to radar glyph
    """

    assert isinstance(fill,tuple), "fill must be a tuple of RGBA values"
    assert len(fill) == 4, "fill must be a 4-tuple of RGBA values"

    outline = kwargs.get('outline','black')
    outlinewidth = kwargs.get('outlinewidth',2)
    gridlinefill = kwargs.get('gridlinefill','lightgray')
    gridlinewidth = kwargs.get('gridlinewidth',1)
    radii = kwargs.get('radii',True)
    gridlines = kwargs.get('gridlines',4)
    axislabels = kwargs.get('axislabels',False)
    
    n_axes = len(vertex_list)
            
    canvas = Image.new('RGBA', (side,side), None)
            
    canvas = draw_polygon(canvas, vertex_list, fill, outline, outlinewidth)
    
    if radii:
        canvas = draw_radii(canvas, n_axes, gridlinefill, gridlinewidth)
        
    if gridlines:
        canvas = draw_gridlines(canvas, n_axes, gridlinefill, gridlinewidth, gridlines)
        
    if axislabels:
        canvas = draw_axis_labels(canvas, axislabels, gridlinefill)

    if mat:
        canvas = _mat(canvas, rgba=True)
        
    return canvas

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def get_radar_vertices(vertex_list: list, side=800) -> list:
    """
    args:
    
    vertex_list: (list) at least 3 normalized radius positions
    side: (int) canvas pixels per side
    
    return:
    
    image_vertices: (list) a list of canvas coordinates
    """
    
    n_axes = len(vertex_list)
    if n_axes < 3:
        raise ValueError("`vertex_list` must contain at least 3 values")
        
    is_odd = n_axes % 2 != 0
    angle_increment = 360 / n_axes
    
    if is_odd or n_axes == 4:
        starting_theta = 90
    else:
        starting_theta = 90 - angle_increment / 2
        
    polar_thetas = [item-360 if item >= 360 else item for item in arange(starting_theta,
                                                         starting_theta+360,
                                                         angle_increment)]
    
    polar_thetas_radians = [radians(item) for item in polar_thetas]
    
    polar_vertices = list(zip(vertex_list,polar_thetas_radians))
    cartesian_vertices = [polar2cartesian(*item) for item in polar_vertices]
    halbside = side / 2
    cartesian_image_vertices = [(item[0]*halbside, item[1]*halbside) for item in cartesian_vertices]
    origin = (halbside,halbside)
    image_vertices = [(origin[0]+item[0], origin[1]-item[1]) for item in cartesian_image_vertices]
    image_vertices = [(int(item[0]),int(item[1])) for item in image_vertices]
    
    return image_vertices

def draw_radii(canvas: Image, n_axes: int, fill: str, width: int) -> Image:
    side = canvas.width
    origin = (side/2,side/2)
    
    draw = ImageDraw.Draw(canvas)
    
    image_vertices = get_radar_vertices([1] * n_axes, side)
    for image_vertex in image_vertices:
        draw.line([origin,image_vertex],fill=fill,width=width)
        
    return canvas

def draw_gridlines(canvas: Image, n_axes: int, fill: str, width: int, n_ticks: int) -> Image:
    side = canvas.width
    
    draw = ImageDraw.Draw(canvas)
    
    for r in linspace(0,1,n_ticks):
        image_vertices = get_radar_vertices([r] * n_axes, side)
        for i in range(len(image_vertices)):
            draw.line([image_vertices[i-1],image_vertices[i]],fill=fill,width=width)
        
    return canvas

def draw_polygon(canvas: Image, vertex_list: list, fill: str, outline: str, width: int) -> Image:
    side = canvas.width
    
    draw = ImageDraw.Draw(canvas)
    
    image_vertices = get_radar_vertices(vertex_list, side)
    
    draw.polygon(image_vertices,
                 fill=fill,
                 outline=outline,
                 width=width)
    
    return canvas

def draw_axis_labels(canvas: Image, label_list: list, fill: str) -> Image:
    
    canvas = _mat(canvas, rgba=False)
    side = canvas.width
    
    draw = ImageDraw.Draw(canvas)
    
    image_vertices = get_radar_vertices([0.95] * len(label_list), side)
    
    fontsize = int(side/20)
    font = get_font(fontsize=fontsize)
    for i,label in enumerate(label_list):
        draw.text(image_vertices[i],label,fill=fill,font=font)
        
    return canvas

#-------------------------------------------------------------------------------

def _mat(im, incr=0.1, bg='white', rgba=True):
    w = im.width
    h = im.height

    w_incr = int( w * incr )
    h_incr = int( h * incr )

    if rgba:
        newim = Image.new('RGBA',(w+w_incr*2,h+h_incr*2),None)
        newim.paste(im,(w_incr,h_incr),im)
    else:
        # this for some reason prevents losing rect color when you mat a legend
        newim = Image.new('RGB',(w+w_incr*2,h+h_incr*2),bg)
        newim.paste(im,(w_incr,h_incr))
    
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

    font = get_font(fontsize=144)
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

def get_random_color(alphargb):
    """
    Get a random color.
    """

    return tuple(*random.randint(0,255,3), alphargb)

