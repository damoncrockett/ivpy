from PIL import Image,ImageDraw,ImageFont
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage import color
from shapely.geometry import Point

from .data import _colfilter

def show(idx,pathcol=None,thumb=False):
    
    pathcol,featcol,ycol = _colfilter(pathcol)
    
    im = Image.open(pathcol.loc[idx])

    if thumb!=False:
        im.thumbnail((thumb,thumb),Image.ANTIALIAS)
    
    return im

def montage(pathcol=None,
            featcol=None,
            thumb=100,
            sample=False,
            idx=False,
            bg="#4a4a4a",
            shape='rect',
            ascending=False):
    
    # only first argument is positional
    pathcol,featcol,ycol = _colfilter(pathcol,
                                      featcol=featcol,
                                      sample=sample,
                                      ascending=ascending)

    if shape=='rect':

        if isinstance(thumb, int):
            xgrid = 980 / thumb # hard-coded bc of Jupyter cell sizes
        elif isinstance(thumb, float):
            xgrid = 980 / int(thumb)
            warnings.warn("""Variable 'thumb' given as a float,
                             rounded to nearest integer""")
        else:
            raise ValueError("Variable 'thumb' must be an integer")

        nrows = len(pathcol) / xgrid + 1 # python int divide always rounds down
        x = range(xgrid) * nrows
        x = [item * thumb for item in x]
        x = x[:len(pathcol)]
        y = np.repeat(range(nrows),xgrid)
        y = [item * thumb for item in y]
        y = y[:len(pathcol)]
        coords = zip(x,y)
        
        canvas = Image.new('RGB',(xgrid*thumb,nrows*thumb),bg)

        counter=-1
        for i in pathcol.index:
            counter+=1
            im = Image.open(pathcol.loc[i])
            im.thumbnail((thumb,thumb),Image.ANTIALIAS)
            
            # idx labels placed after thumbnail
            if idx==True:
                _idx(im,i)

            canvas.paste(im,coords[counter])

    elif shape=='circle':
        
        n = len(pathcol)
        side = int(np.sqrt(n)) + 5 # may have to tweak this
        x, y = range(side) * side, np.repeat(range(side),side)
        grid_list = pd.DataFrame({"x":x,"y":y})

        points = []
        for i in grid_list.index:
            points.append(Point(grid_list.x.loc[i],grid_list.y.loc[i]))
        grid_list['point'] = points
        open_grid = list(grid_list.point)

        canvas = Image.new('RGB',(side*thumb, side*thumb), bg)

        maximus = Point(side/2,side/2)
        im = Image.open(pathcol.iloc[0])
        im.thumbnail((thumb,thumb),Image.ANTIALIAS)
        x = int(maximus.x) * thumb
        y = int(maximus.y) * thumb

        # idx labels placed after thumbnail
        if idx==True:
            _idx(im,pathcol.index[0])

        canvas.paste(im,(x,y))
        open_grid.remove(maximus)

        for i in pathcol.index[1:]:
            im = Image.open(pathcol.loc[i])
            im.thumbnail((thumb,thumb),Image.ANTIALIAS)
            closest_open = min(open_grid,key=lambda x: maximus.distance(x))
            x = int(closest_open.x) * thumb
            y = int(closest_open.y) * thumb
            
            # idx labels placed after thumbnail
            if idx==True:
                _idx(im,i)

            canvas.paste(im,(x,y))
            open_grid.remove(closest_open)

    else:
        raise ValueError("Shape value must be either 'rect' or 'circle'")
    
    return canvas  

# thumb and bin defaults multiply to 980, the Jupyter cell width
def histogram(featcol,
              pathcol=None,
              ycol=None,
              thumb=28, 
              nbins=35, 
              sample=False,
              idx=False,
              ascending=False,
              bg="#4a4a4a",
              quantile=False,
              coordinates='cartesian'): # not yet implemented
    
    # only first argument is positional
    pathcol,featcol,ycol = _colfilter(pathcol,
                                      featcol=featcol,
                                      ycol=ycol,
                                      sample=sample)

    if quantile==False:
        xbin = pd.cut(featcol,nbins,labels=False)
    elif quantile==True:
        xbin = pd.qcut(featcol,nbins,labels=False,duplicates='drop')
    else:
        raise ValueError("'quantile' must be a Boolean")

    bins = xbin.unique()
    binmax = xbin.value_counts().max()
    px_w = thumb * nbins
    px_h = thumb * binmax

    canvas = Image.new('RGB',(px_w,px_h),bg)

    for binlabel in bins:
        if ycol is not None:
            ycol_bin = ycol[xbin==binlabel]
            ycol_bin = ycol_bin.sort_values(ascending=ascending)
            pathcol_bin = pathcol.loc[ycol_bin.index]
        else:
            pathcol_bin = pathcol[xbin==binlabel]

        y_coord = px_h - thumb # bc paste loc is upper left corner
        x_coord = thumb * binlabel

        for i in pathcol_bin.index:
            im = Image.open(pathcol_bin.loc[i])
            im.thumbnail((thumb,thumb),Image.ANTIALIAS)
            
            if idx==True:
                _idx(im,i)

            canvas.paste(im,(x_coord,y_coord))
            y_coord = y_coord - thumb

    return canvas

def scatter(featcol,
            ycol,
            pathcol=None,
            thumb=32,
            side=500,
            sample=False,
            idx=False,
            gridded=False,
            xdomain=None,
            ydomain=None,
            coordinates='cartesian'):
    
    # only first argument is positional
    pathcol,featcol,ycol = _colfilter(pathcol,
                                      featcol=featcol,
                                      ycol=ycol,
                                      sample=sample) 

    # scaling
    if xdomain==None:
        xmin = featcol.min()
        xmax = featcol.max()
    else:
        if not all(isinstance(xdomain,(list,tuple)),len(xdomain)==2):
            raise ValueError("'xdomain' must be two-item list or tuple")
        else:
            xmin = xdomain[0]
            xmax = xdomain[1]

    if ydomain==None:
        ymin = ycol.min()
        ymax = ycol.max()
    else:
        if not all(isinstance(ydomain,(list,tuple)),len(ydomain)==2):
            raise ValueError("'ydomain' must be two-item list or tuple")
        else:
            ymin = ydomain[0]
            ymax = ydomain[1]



    y = [int(item * side) for item in yfeatcol]
    
    coords = zip(x,y)

    canvas = Image.new('RGB',(side,side),bg) # fixed size

    for i in range(len(paths)):
        im = Image.open(paths.iloc[i])
        im.thumbnail((thumb,thumb),Image.ANTIALIAS)
        canvas.paste(im,coords[i])

    return canvas

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






























# ende   