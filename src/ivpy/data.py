import pandas as pd
import copy

ATTACHED_DATAFRAME = None
ATTACHED_PATHCOL = None

def _typecheck(**kwargs):

    """data table columns"""
    pathcol = kwargs.get('pathcol')
    featcol = kwargs.get('featcol')
    ycol = kwargs.get('ycol')
    facetcol = kwargs.get('facetcol')

    """type checking"""
    if pathcol is not None:
        if not isinstance(pathcol,(int,pd.Series)):
            raise TypeError("""'pathcol' must be an integer or
                               a pandas Series""")
    if featcol is not None:
        if not isinstance(featcol,(basestring,pd.Series)):
            raise TypeError("'featcol' must be a string or a pandas Series")
    if ycol is not None:
        if not isinstance(ycol,(basestring,pd.Series)):
            raise TypeError("'ycol' must be a string or a pandas Series")
    if facetcol is not None:
        if not isinstance(facetcol,(basestring,pd.Series)):
            raise TypeError("'facetcol' must be a string or a pandas Series")

    """plot settings"""
    thumb = kwargs.get('thumb')
    sample = kwargs.get('sample')
    idx = kwargs.get('idx')
    bg = kwargs.get('bg')
    ascending = kwargs.get('ascending')
    shape = kwargs.get('shape')
    ncols = kwargs.get('ncols')
    rounding = kwargs.get('rounding')
    nbins = kwargs.get('nbins')
    quantile = kwargs.get('quantile')
    coordinates = kwargs.get('coordinates')
    side = kwargs.get('side')
    gridded = kwargs.get('gridded')
    xdomain = kwargs.get('xdomain')
    ydomain = kwargs.get('ydomain')

    """type checking"""
    if thumb is not None:
        if not isinstance(thumb,int):
            raise TypeError("'thumb' must be an integer")
    if sample is not None:
        if not isinstance(sample,int):
            raise TypeError("'sample' must be an integer")
    if idx is not None:
        if not isinstance(idx,bool):
            raise TypeError("'idx' must be True or False")
    if bg is not None:
        if not isinstance(bg,(tuple,basestring)):
            raise TypeError("'bg' must be an RGB triplet or a string")
    if ascending is not None:
        if not isinstance(ascending,bool):
            raise TypeError("'ascending' must be True or False")
    if shape is not None:
        if not any([shape=='square',shape=='circle']):
            raise ValueError("'shape' must be 'circle' or 'square'")
    if ncols is not None:
        if not isinstance(ncols,int):
            raise TypeError("'ncols' must be an integer")
    if rounding is not None:
        if not any([rounding=='up',rounding=='down']):
            raise ValueError("'rounding' must be 'up' or 'down'")
    if nbins is not None:
        if not isinstance(nbins,int):
            raise TypeError("'nbins' must be an integer")
    if quantile is not None:
        if not isinstance(quantile,bool):
            raise TypeError("'quantile' must be True or False")
    if coordinates is not None:
        if not any([coordinates=='cartesian',coordinates=='polar']):
            raise TypeError("'coordinates' must be 'cartesian' or 'polar'")
    if side is not None:
        if not isinstance(side,int):
            raise TypeError("'side' must be an integer")
    if gridded is not None:
        if not isinstance(gridded,bool):
            raise TypeError("'gridded' must be True or False")
    if xdomain is not None:
        if not all([isinstance(xdomain,(list,tuple)),len(xdomain)==2]):
            raise TypeError("'xdomain' must be a two-item list or tuple")
    if ydomain is not None:
        if not all([isinstance(ydomain,(list,tuple)),len(ydomain)==2]):
            raise TypeError("'ydomain' must be a two-item list or tuple")

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
            raise ValueError("""Length of path list does not match length of
                                DataFrame""")

def detach(df):

    global ATTACHED_DATAFRAME
    global ATTACHED_PATHCOL

    ATTACHED_DATAFRAME = None
    ATTACHED_PATHCOL = None

def _colfilter(pathcol,featcol=None,ycol=None,sample=False,ascending=False):

    pathcol = _pathfilter(pathcol)
    featcol = _featfilter(pathcol,featcol)
    ycol = _featfilter(pathcol,ycol)

    pathcol,featcol,ycol = _sample(pathcol,featcol,ycol,sample)
    pathcol,featcol,ycol = _sort(pathcol,featcol,ycol,ascending)

    return pathcol,featcol,ycol

def _pathfilter(pathcol):

    global ATTACHED_PATHCOL

    """
    Managing user supplied data. If a DataFrame is attached, user can pass
    an empty montage function which will plot all images in attached
    DataFrame in the order they appear there. If no DataFrame is attached,
    user must supply 'pathcol', otherwise the function has no image files
    to open and plot.
    """

    if pathcol is None:
        if ATTACHED_PATHCOL is None:
            raise ValueError("No DataFrame attached; must supply 'pathcol'")
        else:
            pathcol = copy.deepcopy(ATTACHED_PATHCOL)
    else:
        if isinstance(pathcol, int): # for use in show()
            if ATTACHED_PATHCOL is None:
                raise ValueError("No DataFrame attached; must supply 'pathcol'")
            else:
                tmp = copy.deepcopy(ATTACHED_PATHCOL)
                pathcol = tmp.loc[pathcol]

    return pathcol

def _featfilter(pathcol,col):

    global ATTACHED_DATAFRAME

    if col is not None:
        if isinstance(col, pd.Series):
            if not col.index.equals(pathcol.index): # too strong a criterion?
                raise ValueError("""Image paths and image features must have
                                    same indices""")
        elif isinstance(col, basestring):
            if ATTACHED_DATAFRAME is None:
                raise TypeError("""No DataFrame attached. Feature variable
                                    must be a pandas Series""")
            else:
                tmp = copy.deepcopy(ATTACHED_DATAFRAME)
                col = tmp[col]
                if not col.index.equals(pathcol.index): # too strong criterion?
                    raise ValueError("""Image paths and image features must
                                        have same indices""")

    return col

def _sample(pathcol,featcol,ycol,sample):

    """
    If user supplies a number for 'sample', we sample pathcol and subset
    feat/ycol to these indices. We do this because later, we might sort
    by feat/ycol. If so, we will want to then subset pathcol by feat/ycol,
    and this won't work if pathcol is missing a bunch of the indices after
    sampling. So we must sample both together.
    """
    if sample!=False:
        pathcol = pathcol.sample(n=sample)
        if featcol is not None:
            featcol = featcol.loc[pathcol.index]
        if ycol is not None:
            ycol = ycol.loc[pathcol.index]

    return pathcol,featcol,ycol

def _sort(pathcol,featcol,ycol,ascending):

    """
    If user supplies featcol, we sort featcol and apply to path/ycol.
    At this point, if featcol is not None, we know it is a pandas Series
    (either user-supplied or from the attached df). We also know it is
    the same length as pathcol - even if pathcol got sampled above. Note
    that for scatterplots, sorting makes no difference. But it's easier
    to reuse this function for all plots. Note that your working DataFrame
    is unchanged.
    """
    if featcol is not None:
        featcol = featcol.sort_values(ascending=ascending)
        pathcol = pathcol.loc[featcol.index]

        if ycol is not None:
            ycol = ycol.loc[pathcol.index]

    return pathcol,featcol,ycol
