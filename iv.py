from PIL import Image,ImageDraw
import numpy as np
import pandas as pd

def show(pathcol,idx,thumb=False):
    im = Image.open(pathcol.loc[idx])
    if thumb!=False:
        im.thumbnail((thumb,thumb),Image.ANTIALIAS)
    return im

def montage(pathcol,featdf,featcol,thumb=100,sample=False,idx=False):
    xgrid = 868 / thumb # hard-coded bc of Jupyter cell sizes
    featdf = featdf[featdf[featcol] > 0.0]

    if sample!=False:
        featdf = featdf.sample(n=sample)
    
    featdf = featdf.sort_values(by=featcol,ascending=False)
    
    paths = pathcol.loc[featdf.index] # will still be sorted
    nrows = len(paths) / xgrid + 1 # python int division always rounds down
    
    x = range(xgrid) * nrows
    x = [item * thumb for item in x]
    x = x[:len(paths)]
    y = np.repeat(range(nrows),xgrid)
    y = [item * thumb for item in y]
    y = y[:len(paths)]
    coords = zip(x,y)
    
    #canvas = Image.new('RGB',(xgrid*thumb,nrows*thumb),(50,50,50))
    canvas = Image.new('RGB',(xgrid*thumb,nrows*thumb),(255,255,255))
    
    for i,j in zip(range(len(paths)),featdf.index):
        im = Image.open(paths.iloc[i])
        im.thumbnail((thumb,thumb),Image.ANTIALIAS)
        
        if idx==True:
            pos = (7,7)
            draw = ImageDraw.Draw(im)
            try:
                draw.text(pos, str(int(j)), fill="black")
            except Exception as e:
                print e

        canvas.paste(im,coords[i])    
    return canvas    

"""
note: an issue with scales: tools like ggplot make a scale based on the
data you give them. But in general, I want my plot boundaries to match
the canonical ranges for things, even if the supplied data don't span 
the canonical range. Not clear what to do about this.
"""

def scatter(pathcol,xfeatdf,yfeatdf,xfeat,yfeat,thumb=32,side=500):
    """This function is hard coded for a particular use case."""
    
    idxs = xfeatdf.index[xfeatdf[xfeat] > 0.0]
    paths = pathcol.loc[idxs]
    xfeatcol = xfeatdf[xfeat].loc[idxs]
    yfeatcol = yfeatdf[yfeat].loc[idxs]

    canvas = Image.new('RGB',(side,side),(50,50,50)) # fixed size
    
    # mapping to new scale
    x = [int(item * side) for item in xfeatcol]
    
    if yfeat=='hue_peak':
        y = [int(item / float(360) * side) for item in yfeatcol]
    else:
        y = [int(item * side) for item in yfeatcol]
    
    coords = zip(x,y)

    for i in range(len(paths)):
        im = Image.open(paths.iloc[i])
        im.thumbnail((thumb,thumb),Image.ANTIALIAS)
        canvas.paste(im,coords[i])

    return canvas

def histogram(pathcol,xfeatdf,yfeatdf,xfeat,yfeat,thumb=40,nbins=20,ascendsort=False):
    idxs = xfeatdf.index[xfeatdf[xfeat] > 0.0]
    paths = pathcol.loc[idxs]
    xfeatcol = xfeatdf[xfeat].loc[idxs]
    yfeatcol = yfeatdf[yfeat].loc[idxs]

    xbin = pd.cut(xfeatcol,nbins,labels=False)
    bins = xbin.unique()
    
    binmax = xbin.value_counts().max()
    px_w = thumb * nbins
    px_h = thumb * binmax

    canvas = Image.new('RGB',(px_w,px_h),(50,50,50))

    for binlabel in bins:
        binyfeat = yfeatcol[xbin==binlabel]
        binyfeat = binyfeat.sort_values(ascending=ascendsort)
        binpaths = paths[binyfeat.index]

        y_coord = px_h - thumb
        x_coord = thumb * binlabel

        for path in binpaths:
            im = Image.open(path)
            im.thumbnail((thumb,thumb),Image.ANTIALIAS)
            canvas.paste(im,(x_coord,y_coord))
            y_coord = y_coord - thumb

    return canvas
