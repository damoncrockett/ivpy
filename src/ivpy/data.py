import pandas as pd
import copy

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
            raise ValueError("""Length of path list does not match length of 
                                DataFrame""") 
    else:
        raise TypeError("""Variable 'pathcol' must be either a string or a 
                           pandas Series""")

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
        if not isinstance(pathcol, pd.Series):
            raise ValueError("If supplied, 'pathcol' must be a pandas Series")

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
                raise ValueError("""No DataFrame attached. Feature variable 
                                    must be a pandas Series""")
            else:
                tmp = copy.deepcopy(ATTACHED_DATAFRAME)
                col = tmp[col]
                if not col.index.equals(pathcol.index): # too strong criterion?
                    raise ValueError("""Image paths and image features must 
                                        have same indices""")
        else:
            raise ValueError("""If supplied, feature variable must be a string 
                                or a pandas Series""")

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
        if not isinstance(sample, int):
            raise ValueError("Sample size must be an integer")
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















    # ende