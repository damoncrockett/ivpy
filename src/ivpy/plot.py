from PIL import Image
import numpy as np
import pandas as pd
from shapely.geometry import Point

from .data import _colfilter
from .plottools import _scale, _pct, _idx, _placeholder

#------------------------------------------------------------------------------

def show(pathcol=None,
         featcol=None,
         thumb=False,
         sample=False,
         idx=False,
         bg='#4a4a4a',
         ascending=False):

    """
    Shows either a single image by index or a pathcol, possibly sampled,
    as a scrolling, sortable rect montage

    Args:
        shown (int,Series) --- single index or col of image paths to be shown
        featcol (str,Series) --- sorting column
        thumb (int) --- pixel value for thumbnail side
        sample (int) --- integer size of sample
        ascending (Boolean) --- sorting order
    """

    # only first argument is positional
    pathcol,featcol,ycol = _colfilter(pathcol,
                                      featcol=featcol,
                                      sample=sample,
                                      ascending=ascending)

    if isinstance(pathcol, str): # single pathstring

        im = Image.open(pathcol)

        if thumb!=False:
            im.thumbnail((thumb,thumb),Image.ANTIALIAS)

        return im

    else:

        if thumb==False:
            thumb = 100
            xgrid = 980 / thumb # hard-coded bc of Jupyter cell sizes
        elif isinstance(thumb, int):
            xgrid = 980 / thumb
        elif isinstance(thumb, float):
            xgrid = 980 / int(thumb)
            warnings.warn("""Variable 'thumb' given as a float,
                             rounded to nearest integer""")
        else:
            raise TypeError("'thumb' must be an integer")

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

            try:
                im = Image.open(pathcol.loc[i])
            except:
                im = _placeholder(thumb)

            im.thumbnail((thumb,thumb),Image.ANTIALIAS)

            # idx labels placed after thumbnail
            if idx==True:
                _idx(im,i)

            canvas.paste(im,coords[counter])

        return canvas

def montage(pathcol=None,
            featcol=None,
            thumb=100,
            sample=False,
            idx=False,
            bg='#4a4a4a',
            shape='square',
            ascending=False):

    """
    Square or circular montage of images

    Args:
        pathcol (Series) --- col of image paths to be plotted
        featcol (str,Series) --- sorting column
        thumb (int) --- pixel value for thumbnail side
        sample (int) --- integer size of sample
        idx (Boolean) --- whether to print index on image
        bg (color) --- background color
        shape (str) --- square or circular montage
        ascending (Boolean) --- sorting order
    """

    # only first argument is positional
    pathcol,featcol,ycol = _colfilter(pathcol,
                                      featcol=featcol,
                                      sample=sample,
                                      ascending=ascending)

    if shape=='square':

        xgrid = int ( np.sqrt( len(pathcol) ) )
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

            try:
                im = Image.open(pathcol.loc[i])
            except:
                im = _placeholder(thumb)

            im.thumbnail((thumb,thumb),Image.ANTIALIAS)

            # idx labels placed after thumbnail
            if idx==True:
                _idx(im,i)

            canvas.paste(im,coords[counter])

    elif shape=='circle':

        n = len(pathcol)
        side = int(np.sqrt(n)) + 5 # may have to tweak this
        x, y = range(side) * side, np.repeat(range(side),side)
        grid_list = [Point(item) for item in zip(x,y)]

        canvas = Image.new('RGB',(side*thumb, side*thumb), bg)

        # plot center image
        maximus = Point(side/2,side/2)

        try:
            im = Image.open(pathcol.iloc[0])
        except:
            im = _placeholder(thumb)

        im.thumbnail((thumb,thumb),Image.ANTIALIAS)
        x = int(maximus.x) * thumb
        y = int(maximus.y) * thumb

        # idx label placed after thumbnail
        if idx==True:
            _idx(im,pathcol.index[0])

        canvas.paste(im,(x,y))
        grid_list.remove(maximus)

        # compute distance from center for each point
        tmp = pd.DataFrame(
            {"grid_list":grid_list,
             "grid_distances":[maximus.distance(item) for item in grid_list]}
             )

        # sorting grid locations by computed distance
        tmp.sort_values(by='grid_distances',inplace=True) # ascending
        grid_list = tmp.grid_list.iloc[:n-1] # n-1 bc we removed maximus

        counter=-1
        for i in pathcol.index[1:]:
            counter+=1

            try:
                im = Image.open(pathcol.loc[i])
            except:
                im = _placeholder(thumb)

            im.thumbnail((thumb,thumb),Image.ANTIALIAS)

            x = int(grid_list.iloc[counter].x) * thumb
            y = int(grid_list.iloc[counter].y) * thumb

            # idx labels placed after thumbnail
            if idx==True:
                _idx(im,i)

            canvas.paste(im,(x,y))

    else:
        raise ValueError("'shape' must be either 'square' or 'circle'")

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

    """
    Cartesian or polar histogram of images

    Args:
        featcol (str,Series) --- histogram axis; must be supplied
        pathcol (Series) --- col of image paths to be plotted
        ycol (str,Series) --- vertical sorting feature if desired
        thumb (int) --- pixel value for thumbnail side
        nbins (int) --- number of bins used to discretize featcol;
            can alternatively be entered as array of bin edges
        sample (int) --- integer size of sample
        idx (Boolean) --- whether to print index on image
        ascending (Boolean) --- vertical sorting direction
        bg (color) --- background color
        quantile (Boolean) --- whether binning is quantile-based;
            if True, produces nearly even spread across bins
        coordinates (str) --- Cartesian or polar
    """

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
        raise TypeError("'quantile' must be a Boolean")

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

            try:
                im = Image.open(pathcol_bin.loc[i])
            except:
                im = _placeholder(thumb)

            im.thumbnail((thumb,thumb),Image.ANTIALIAS)

            if idx==True:
                _idx(im,i)

            canvas.paste(im,(x_coord,y_coord))
            y_coord = y_coord - thumb

    return canvas

# gridding not implemented
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
            bg="#4a4a4a",
            coordinates='cartesian'): # not yet implemented

    """
    Cartesian or polar scatterplot of images

    Args:
        featcol (str,Series) --- x-axis; must be supplied
        ycol (str,Series) --- y-axis; must be supplied
        pathcol (Series) --- col of image paths to be plotted
        thumb (int) --- pixel value for thumbnail side
        side (int) --- length of plot side in pixels; all plots enforced square
        sample (int) --- integer size of sample
        idx (Boolean) --- whether to print index on image
        gridded (Boolean) --- whether to snap image locations to a grid
        xdomain (list,tuple) --- xmin and xmax; defaults to data extremes
        ydomain (list,tuple) --- ymin and ymax; defaults to data extremes
        bg (color) --- background color
        coordinates (str) --- Cartesian or polar
    """

    # only first argument is positional
    pathcol,featcol,ycol = _colfilter(pathcol,
                                      featcol=featcol,
                                      ycol=ycol,
                                      sample=sample)

    x = _scale(featcol,xdomain,side,thumb)
    y = _scale(ycol,ydomain,side,thumb,y=True)
    coords = zip(x,y)

    canvas = Image.new('RGB',(side,side),bg) # fixed size

    counter=-1
    for i in pathcol.index:
        counter+=1

        try:
            im = Image.open(pathcol.loc[i])
        except:
            im = _placeholder(thumb)

        im.thumbnail((thumb,thumb),Image.ANTIALIAS)

        if idx==True:
            _idx(im,i)

        canvas.paste(im,coords[counter])

    return canvas
