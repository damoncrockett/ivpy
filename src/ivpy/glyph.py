import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from .data import _typecheck
from .plottools import polar2cartesian
import os
from numpy import radians, arange, linspace, random
import re

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

FONTDIR = os.path.expanduser("~") + "/fonts/"

def get_font(fonttype="Roboto-Light", fontsize=40):
    font = ImageFont.truetype(FONTDIR + fonttype + ".ttf", fontsize)

    return font

#-------------------------------------------------------------------------------

def draw_glyphs(X,
                fill=(160,160,160,255),
                savedir=None,
                savecol=None,
                glyphtype='radar',
                mat=True,
                side=800,
                alpha=1.0,
                legend=False,
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

    if isinstance(fill, pd.Series):
        assert X.index.equals(fill.index), "`fill` must have the same index as `X`"
        
        # if user supplies a pandas Series of valid colors, use them; otherwise, generate a color map
        try: 
            fill = pd.Series([anycolor_to_rgba(item,alphargb) for item in fill], index=fill.index, dtype='object')
        except:
            colorkeys = list(fill.value_counts().keys())
            numcolors = len(colorkeys)
            if numcolors > len(colors):
                lendiff = numcolors - len(colors)
                randomcolors = [get_random_color(alphargb) for _ in range(lendiff)]
                colors = colors + randomcolors

                print("Warning: more than 10 colors specified; groups outside of the 10 largest will be drawn in random colors.")

            vals = colors[:len(colorkeys)]
            cmap = dict(zip(colorkeys,vals))

            if legend:
                _legend(cmap).save('_legend.png')

            fill = pd.Series([cmap[val] for val in fill],index=fill.index)
        
        na_indices = fill.loc[fill.isna()].index
        nafiller_series = pd.Series([(160,160,160,alphargb)] * len(na_indices), index=na_indices, dtype='object')
        fill.loc[na_indices] = nafiller_series
   
    elif isinstance(fill, (tuple, str)): 
        fill = pd.Series([anycolor_to_rgba(fill,alphargb)] * len(X.index),index=X.index, dtype='object')

    if savedir:
        try:
            os.mkdir(savedir)
        except:
            pass

    gpaths = []
    for i in X.index:
        if glyphtype=='radar':
            vertex_list = list(X.loc[i])
            g = radar(vertex_list, fill.loc[i], side, mat=mat, **kwargs)

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

def radar(vertex_list: list, fill=(160,160,160,255), side=800, mat=True, alpha=1.0, **kwargs) -> Image:
    """
    args:
    
    vertex_list: (list) at least 3 normalized radius positions
    fill: (tuple, str) polygon fill color
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

    _typecheck(**locals())

    if fill is not None:
        # stricter than typecheck, which allows for a Series
        if not isinstance(fill, (tuple, str)):
            raise ValueError("`fill` must be a tuple or string")
        
        alphargb = int(alpha*255)

        try:
            fill = anycolor_to_rgba(fill,alphargb)
        except:
            raise ValueError("`fill` must be a valid color")
    
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

def anycolor_to_rgba(color, alphargb=255):
    
    # Normalize the color string if it's string type
    if isinstance(color, str):
        color = color.lower().strip()

        # Check for RGB/RGBA format
        rgb_match = re.match(r"rgba?\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})(?:,\s*(\d+(\.\d+)?))?\)$", color)
        if rgb_match:
            r, g, b = map(int, rgb_match.group(1, 2, 3))
            a = int(float(rgb_match.group(4))) if rgb_match.group(4) else alphargb
            return (r, g, b, a)

        # Convert named colors to hex
        if color in NAMED_COLORS:
            color = NAMED_COLORS[color]

        # Convert 3-char hex to 6-char hex
        if len(color) == 4 and color[0] == "#":
            color = "#" + "".join([c + c for c in color[1:]])

        # Check if it's a valid 6-character hex string
        if len(color) == 7 and color[0] == "#":
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            return (r, g, b, alphargb)
    
    elif isinstance(color, tuple) and len(color) in [3, 4]:
        r, g, b = color[:3]
        a = color[3] if len(color) == 4 else alphargb
        if all(0 <= v <= 255 for v in (r, g, b, a)):
            return (r, g, b, a)

    raise ValueError("Invalid color format")

# Dictionary of HTML named colors
NAMED_COLORS = {
    "aliceblue": "#f0f8ff",
    "antiquewhite": "#faebd7",
    "aqua": "#00ffff",
    "aquamarine": "#7fffd4",
    "azure": "#f0ffff",
    "beige": "#f5f5dc",
    "bisque": "#ffe4c4",
    "black": "#000000",
    "blanchedalmond": "#ffebcd",
    "blue": "#0000ff",
    "blueviolet": "#8a2be2",
    "brown": "#a52a2a",
    "burlywood": "#deb887",
    "cadetblue": "#5f9ea0",
    "chartreuse": "#7fff00",
    "chocolate": "#d2691e",
    "coral": "#ff7f50",
    "cornflowerblue": "#6495ed",
    "cornsilk": "#fff8dc",
    "crimson": "#dc143c",
    "cyan": "#00ffff",
    "darkblue": "#00008b",
    "darkcyan": "#008b8b",
    "darkgoldenrod": "#b8860b",
    "darkgray": "#a9a9a9",
    "darkgreen": "#006400",
    "darkgrey": "#a9a9a9",
    "darkkhaki": "#bdb76b",
    "darkmagenta": "#8b008b",
    "darkolivegreen": "#556b2f",
    "darkorange": "#ff8c00",
    "darkorchid": "#9932cc",
    "darkred": "#8b0000",
    "darksalmon": "#e9967a",
    "darkseagreen": "#8fbc8f",
    "darkslateblue": "#483d8b",
    "darkslategray": "#2f4f4f",
    "darkslategrey": "#2f4f4f",
    "darkturquoise": "#00ced1",
    "darkviolet": "#9400d3",
    "deeppink": "#ff1493",
    "deepskyblue": "#00bfff",
    "dimgray": "#696969",
    "dimgrey": "#696969",
    "dodgerblue": "#1e90ff",
    "firebrick": "#b22222",
    "floralwhite": "#fffaf0",
    "forestgreen": "#228b22",
    "fuchsia": "#ff00ff",
    "gainsboro": "#dcdcdc",
    "ghostwhite": "#f8f8ff",
    "gold": "#ffd700",
    "goldenrod": "#daa520",
    "gray": "#808080",
    "green": "#008000",
    "greenyellow": "#adff2f",
    "grey": "#808080",
    "honeydew": "#f0fff0",
    "hotpink": "#ff69b4",
    "indianred ": "#cd5c5c",
    "indigo ": "#4b0082",
    "ivory": "#fffff0",
    "khaki": "#f0e68c",
    "lavender": "#e6e6fa",
    "lavenderblush": "#fff0f5",
    "lawngreen": "#7cfc00",
    "lemonchiffon": "#fffacd",
    "lightblue": "#add8e6",
    "lightcoral": "#f08080",
    "lightcyan": "#e0ffff",
    "lightgoldenrodyellow": "#fafad2",
    "lightgray": "#d3d3d3",
    "lightgreen": "#90ee90",
    "lightgrey": "#d3d3d3",
    "lightpink": "#ffb6c1",
    "lightsalmon": "#ffa07a",
    "lightseagreen": "#20b2aa",
    "lightskyblue": "#87cefa",
    "lightslategray": "#778899",
    "lightslategrey": "#778899",
    "lightsteelblue": "#b0c4de",
    "lightyellow": "#ffffe0",
    "lime": "#00ff00",
    "limegreen": "#32cd32",
    "linen": "#faf0e6",
    "magenta": "#ff00ff",
    "maroon": "#800000",
    "mediumaquamarine": "#66cdaa",
    "mediumblue": "#0000cd",
    "mediumorchid": "#ba55d3",
    "mediumpurple": "#9370db",
    "mediumseagreen": "#3cb371",
    "mediumslateblue": "#7b68ee",
    "mediumspringgreen": "#00fa9a",
    "mediumturquoise": "#48d1cc",
    "mediumvioletred": "#c71585",
    "midnightblue": "#191970",
    "mintcream": "#f5fffa",
    "mistyrose": "#ffe4e1",
    "moccasin": "#ffe4b5",
    "navajowhite": "#ffdead",
    "navy": "#000080",
    "oldlace": "#fdf5e6",
    "olive": "#808000",
    "olivedrab": "#6b8e23",
    "orange": "#ffa500",
    "orangered": "#ff4500",
    "orchid": "#da70d6",
    "palegoldenrod": "#eee8aa",
    "palegreen": "#98fb98",
    "paleturquoise": "#afeeee",
    "palevioletred": "#db7093",
    "papayawhip": "#ffefd5",
    "peachpuff": "#ffdab9",
    "peru": "#cd853f",
    "pink": "#ffc0cb",
    "plum": "#dda0dd",
    "powderblue": "#b0e0e6",
    "purple": "#800080",
    "rebeccapurple": "#663399",
    "red": "#ff0000",
    "rosybrown": "#bc8f8f",
    "royalblue": "#4169e1",
    "saddlebrown": "#8b4513",
    "salmon": "#fa8072",
    "sandybrown": "#f4a460",
    "seagreen": "#2e8b57",
    "seashell": "#fff5ee",
    "sienna": "#a0522d",
    "silver": "#c0c0c0",
    "skyblue": "#87ceeb",
    "slateblue": "#6a5acd",
    "slategray": "#708090",
    "slategrey": "#708090",
    "snow": "#fffafa",
    "springgreen": "#00ff7f",
    "steelblue": "#4682b4",
    "tan": "#d2b48c",
    "teal": "#008080",
    "thistle": "#d8bfd8",
    "tomato": "#ff6347",
    "turquoise": "#40e0d0",
    "violet": "#ee82ee",
    "wheat": "#f5deb3",
    "white": "#ffffff",
    "whitesmoke": "#f5f5f5",
    "yellow": "#ffff00",
    "yellowgreen": "#9acd32"
    }