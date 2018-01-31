from PIL import Image,ImageDraw
import numpy as np
import pandas as pd
import copy
from skimage.io import imread
from skimage import color
import warnings

ATTACHED_DATAFRAME = None
ATTACHED_PATHCOL = None

def attach(df,pathcol=None):
    
    if pathcol is None:
        
        raise ValueError("""Must supply variable 'pathcol', either as a string
            that names a column of image paths in the supplied DataFrame, or
            as a pandas Series of image paths 
            """)

    global ATTACHED_DATAFRAME
    global ATTACHED_PATHCOL

    ATTACHED_DATAFRAME = copy.deepcopy(df) # deep to avoid change bleed
    
    if isinstance(pathcol, basestring):
        ATTACHED_PATHCOL = ATTACHED_DATAFRAME[pathcol]
    
    elif isinstance(pathcol, pd.Series):
        
        if len(pathcol)==len(ATTACHED_DATAFRAME):
            ATTACHED_PATHCOL = copy.deepcopy(pathcol)
        
        else:
            raise ValueError("Length of path list does not match length of DataFrame") 
    
    else:
        raise TypeError("Variable 'pathcol' must be either a string or a pandas Series")

def detach(df):
    
    global ATTACHED_DATAFRAME
    global ATTACHED_PATHCOL

    ATTACHED_DATAFRAME = None
    ATTACHED_PATHCOL = None

def show(idx,pathcol=None,thumb=False):
    
    global ATTACHED_PATHCOL

    if pathcol is not None:
        
        if isinstance(pathcol, pd.Series):
            im = Image.open(pathcol.loc[idx])
        
        else: 
            raise ValueError("""Variable 'pathcol' must be a pandas Series""")
        
    else:
        
        try:
            im = Image.open(ATTACHED_PATHCOL.loc[idx])
        
        except:
            raise ValueError("""If no dataframe is attached, must supply a value 
                for 'pathcol'
                """)
    
    if thumb!=False:
        im.thumbnail((thumb,thumb),Image.ANTIALIAS)
    
    return im

def montage(pathcol=None,featcol=None,thumb=100,sample=False,idx=False,bg="#4a4a4a"):
    
    global ATTACHED_DATAFRAME
    global ATTACHED_PATHCOL    

    """
    Parsing user supplied data. If a DataFrame is attached, user can pass an empty montage
    function which will plot all images in attached DataFrame in the order they appear there.
    If no DataFrame is attached, user must supply 'pathcol', otherwise the function has no 
    image files to open and plot.
    """
    if pathcol is None:
        
        if ATTACHED_PATHCOL is None:
            raise ValueError("No DataFrame attached. Must supply 'pathcol'")
        
        else:
            pathcol = copy.deepcopy(ATTACHED_PATHCOL)
    
    else:
        
        if not isinstance(pathcol, pd.Series):
            raise ValueError("If supplied, 'pathcol' must be a pandas Series")

    
    # if featcol is none, images will be plotted by pathcol index ordering
    if featcol is not None:
        
        if isinstance(featcol, pd.Series):
            
            if len(featcol)!=len(pathcol): 
                raise ValueError("Lists of image paths and image features must be same length")
        
        elif isinstance(featcol, basestring):
            
            if ATTACHED_DATAFRAME is None:
                raise ValueError("""No DataFrame attached. Variable 'featcol' must be a pandas
                    Series""")
            
            else:
                
                tmp = copy.deepcopy(ATTACHED_DATAFRAME)
                featcol = tmp[featcol]
                
                if len(featcol)!=len(pathcol): 
                    raise ValueError("""Lists of image paths and image features must be same
                        length""") 
        
        else:
            raise ValueError("If supplied, 'featcol' must be a string or a pandas Series")
    
    elif featcol is None:
        paths = copy.deepcopy(pathcol) # deep copy is possibly not necessary here     
    
    """
    If user supplies a number for 'sample', we sample pathcol and subset featcol to these 
    indices. We do this because later, we might sort by featcol. If so, we will want to then
    subset pathcol by featcol, and this won't work if pathcol is missing a bunch of the indices
    after sampling. So we must sample both together. We raise a warning just to keep the user
    informed of what's going on.
    """
    if sample!=False:
        
        if not isinstance(sample, int):
            raise ValueError("Sample size must be an integer")
        
        pathcol = pathcol.sample(n=sample)
        featcol = featcol.loc[pathcol.index]
        
        warnings.warn("""Sampling done on 'pathcol' and applied to 'featcol'. If these columns
            come from different DataFrames, make sure the indices match.""")

    """
    If user supplies featcol, we sort featcol and apply to pathcol. At this point, if featcol
    is not None, we know it is a pandas Series (either user-supplied or from the attached df).
    We also know it is the same length as pathcol - even if pathcol got sampled above.
    """
    if featcol is not None:
        
        featcol = featcol.sort_values(ascending=False)
        paths = pathcol.loc[featcol.index]
        warnings.warn("""Sorting done on 'featcol' and applied to 'pathcol'. If these columns
            come from different DataFrames, make sure the indices match.""")

    """
    Building the montage
    """
    if isinstance(thumb, int):
        xgrid = 868 / thumb # hard-coded bc of Jupyter cell sizes
    
    elif isinstance(thumb, float):
        xgrid = 868 / int(thumb)
        
        warnings.warn("Variable 'thumb' given as a float, rounded to nearest integer")
    
    else:
        raise ValueError("Variable 'thumb' must be an integer")

    nrows = len(paths) / xgrid + 1 # python int division always rounds down
    x = range(xgrid) * nrows
    x = [item * thumb for item in x]
    x = x[:len(paths)]
    y = np.repeat(range(nrows),xgrid)
    y = [item * thumb for item in y]
    y = y[:len(paths)]
    coords = zip(x,y)
    
    canvas = Image.new('RGB',(xgrid*thumb,nrows*thumb),bg)

    for i,j in zip(range(len(paths)),pathcol.index):
        im = Image.open(paths.iloc[i])
        im.thumbnail((thumb,thumb),Image.ANTIALIAS)
        
        if idx==True:
            pos = (7,7)
            draw = ImageDraw.Draw(im)
            try:
                draw.text(pos, str(int(j)), fill='#c0c0c0')
            except Exception as e:
                print e

        canvas.paste(im,coords[i])    
    
    return canvas  



###################################################################################

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
