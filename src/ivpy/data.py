import pandas as pd
import numpy as np
import copy

ATTACHED_DATAFRAME = None
ATTACHED_PATHCOL = None

def _typecheck(**kwargs):

    """data table columns"""
    pathcol = kwargs.get('pathcol')
    xcol = kwargs.get('xcol')
    ycol = kwargs.get('ycol')
    facetcol = kwargs.get('facetcol')

    """type checking"""
    if pathcol is not None:
        if not isinstance(pathcol,(int,pd.Series)):
            raise TypeError("""'pathcol' must be an integer or
                               a pandas Series""")
    if xcol is not None:
        if not isinstance(xcol,(basestring,pd.Series)):
            raise TypeError("'xcol' must be a string or a pandas Series")
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
    bins = kwargs.get('bins')
    coordinates = kwargs.get('coordinates')
    side = kwargs.get('side')
    xdomain = kwargs.get('xdomain')
    ydomain = kwargs.get('ydomain')
    xbins = kwargs.get('xbins')
    ybins = kwargs.get('ybins')

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
    if bins is not None:
        if not isinstance(bins,(int,list,tuple,np.ndarray)):
            raise TypeError("'bins' must be an integer or a sequence")
    if coordinates is not None:
        if not any([coordinates=='cartesian',coordinates=='polar']):
            raise TypeError("'coordinates' must be 'cartesian' or 'polar'")
    if side is not None:
        if not isinstance(side,int):
            raise TypeError("'side' must be an integer")
    if xdomain is not None:
        if not all([isinstance(xdomain,(list,tuple)),len(xdomain)==2]):
            raise TypeError("'xdomain' must be a two-item list or tuple")
    if ydomain is not None:
        if not all([isinstance(ydomain,(list,tuple)),len(ydomain)==2]):
            raise TypeError("'ydomain' must be a two-item list or tuple")
    if xbins is not None:
        if not isinstance(xbins,(np.ndarray,list,tuple,int)):
            raise TypeError("'xbins' must be an integer or a sequence")
    if ybins is not None:
        if not isinstance(ybins,(np.ndarray,list,tuple,int)):
            raise TypeError("'ybins' must be an integer or a sequence")

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
        if pathcol.index.equals(ATTACHED_DATAFRAME.index):
            ATTACHED_PATHCOL = copy.deepcopy(pathcol)
        else:
            raise ValueError("""'pathcol' must have same indices as 'df'""")

def detach(df):

    global ATTACHED_DATAFRAME
    global ATTACHED_PATHCOL

    ATTACHED_DATAFRAME = None
    ATTACHED_PATHCOL = None

def _colfilter(pathcol,
               xcol=None,
               ycol=None,
               sample=False,
               ascending=False,
               xdomain=None,
               ydomain=None,
               facetcol=None):

    pathcol = _pathfilter(pathcol)
    xcol = _featfilter(pathcol,xcol)
    ycol = _featfilter(pathcol,ycol)
    facetcol = _featfilter(pathcol,facetcol)

    pathcol,xcol,ycol,facetcol = _sample(pathcol,xcol,ycol,
                                            sample,facetcol)
    pathcol,xcol,ycol,facetcol = _sort(pathcol,xcol,ycol,
                                          ascending,facetcol)
    pathcol,xcol,ycol,facetcol = _subset(pathcol,xcol,ycol,
                                            xdomain,ydomain,facetcol)

    return pathcol,xcol,ycol,facetcol

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

def _sample(pathcol,xcol,ycol,sample,facetcol):

    """
    If user supplies a number for 'sample', we sample pathcol and subset
    feat/ycol to these indices. We do this because later, we might sort
    by feat/ycol. If so, we will want to then subset pathcol by feat/ycol,
    and this won't work if pathcol is missing a bunch of the indices after
    sampling. So we must sample both together.
    """
    if sample!=False:
        pathcol = pathcol.sample(n=sample)
        if xcol is not None:
            xcol = xcol.loc[pathcol.index]
        if ycol is not None:
            ycol = ycol.loc[pathcol.index]
        if facetcol is not None:
            facetcol = facetcol.loc[pathcol.index]

    return pathcol,xcol,ycol,facetcol

def _sort(pathcol,xcol,ycol,ascending,facetcol):

    """
    If user supplies xcol, we sort xcol and apply to path/ycol.
    At this point, if xcol is not None, we know it is a pandas Series
    (either user-supplied or from the attached df). We also know it is
    the same length as pathcol - even if pathcol got sampled above. Note
    that for scatterplots, sorting makes no difference. But it's easier
    to reuse this function for all plots. Note that your working DataFrame
    is unchanged. Also note that it is never possible to sort globally by
    ycol. This is because only histogram and scatter have ycol, and for
    histograms, this sorting happens bin by bin.
    """
    if xcol is not None:
        xcol = xcol.sort_values(ascending=ascending)
        pathcol = pathcol.loc[xcol.index]

        if ycol is not None: # nb sorting by ycol not possible globally
            ycol = ycol.loc[xcol.index]

        if facetcol is not None:
            facetcol = facetcol.loc[xcol.index]

    return pathcol,xcol,ycol,facetcol

def _subset(pathcol,xcol,ycol,xdomain,ydomain,facetcol):
    """
    Because this function runs after pathfilter, featfilter, sample and sort,
    many checks have already taken place. If xcol and ycol are not None, we
    know they must be pandas Series and not strings.
    """

    if xdomain is not None:
        if xcol is None:
            raise ValueError("""If 'xdomain' is supplied, 'xcol' must be
                                supplied as well""")
        xcol = xcol[(xcol>=xdomain[0])&(xcol<=xdomain[1])]
        pathcol = pathcol.loc[xcol.index]

        if ycol is not None:
            ycol = ycol.loc[xcol.index]

        if facetcol is not None:
            facetcol = facetcol.loc[xcol.index]

    if ydomain is not None:
        if ycol is None:
            raise ValueError("""If 'ydomain' is supplied, 'ycol' must be
                                supplied as well""")
        ycol = ycol[(ycol>=ydomain[0])&(ycol<=ydomain[1])]
        pathcol = pathcol.loc[ycol.index]
        xcol = xcol.loc[ycol.index] # if ycol is not None, ditto xcol

        if facetcol is not None:
            facetcol = facetcol.loc[ycol.index]

    return pathcol,xcol,ycol,facetcol

def _bin(col,bins):
    col = pd.cut(col,bins)
    leftbinedges = [float(str(item).split(",")[0].lstrip("([")) for item in col]

    return pd.Series(leftbinedges,index=col.index)

def _facet(**kwargs):
    kwargdict = copy.deepcopy(kwargs)
    facetcol = kwargs.get('facetcol')
    pathcol = kwargs.get('pathcol')
    xcol = kwargs.get('xcol')
    ycol = kwargs.get('ycol')
    xdomain = kwargs.get('xdomain')
    ydomain = kwargs.get('ydomain')

    # if user passes 'xdomain' or 'ydomain', these assignments change nothing
    if xcol is not None:
        xdomain = (xcol.min(),xcol.max())
    if ycol is not None:
        ydomain = (ycol.min(),ycol.max())

    facetlist = []
    for val in facetcol.unique():
        tmp = copy.deepcopy(kwargdict)

        # this bit fixes plot axes across facets
        tmp['xdomain'] = xdomain
        tmp['ydomain'] = ydomain

        # generate facet
        facetedcol = facetcol[facetcol==val]
        tmp['pathcol'] = pathcol.loc[facetedcol.index]

        if xcol is not None:
            tmp['xcol'] = xcol.loc[facetedcol.index]
        if ycol is not None:
            tmp['ycol'] = ycol.loc[facetedcol.index]

        facetlist.append(tmp)

    return facetlist
