import pandas as pd
import numpy as np
import copy
from PIL import Image
from six import string_types

ATTACHED_DATAFRAME = None
ATTACHED_PATHCOL = None

int_types = (int,np.int8,np.int16,np.int32,np.int64,
             np.uint8,np.uint16,np.uint32,np.uint64)
seq_types = (list,tuple,np.ndarray,pd.Series)
float_types = (float,np.float16,np.float32,np.float64)

def _typecheck(**kwargs):
    """
    This function is called before plotting, as a way to control the error
    messages users receive when they pass invalid arguments. All of the 'col'
    arguments can be None; many other arguments cannot.
    """

    """data table columns"""
    pathcol = kwargs.get('pathcol')
    xcol = kwargs.get('xcol')
    ycol = kwargs.get('ycol')
    facetcol = kwargs.get('facetcol')
    notecol = kwargs.get('notecol')
    clustercol = kwargs.get('clustercol')
    colorcol = kwargs.get('colorcol')
    savecol = kwargs.get('savecol')

    """type checking"""
    if pathcol is not None:
        if not isinstance(pathcol,(int_types,pd.Series)):
            raise TypeError("""'pathcol' must be an integer or
                               a pandas Series""")
    if xcol is not None:
        if not isinstance(xcol,(string_types,int_types,pd.Series)):
            raise TypeError("'xcol' must be a string, int, or a pandas Series")
    if ycol is not None:
        if not isinstance(ycol,(string_types,int_types,pd.Series)):
            raise TypeError("'ycol' must be a string, int, or a pandas Series")
    if facetcol is not None:
        if not isinstance(facetcol,(string_types,int_types,pd.Series)):
            raise TypeError("""'facetcol' must be a string, int, or
                               a pandas Series""")
    if notecol is not None:
        if not isinstance(notecol,(string_types,int_types,pd.Series)):
            raise TypeError("""'notecol' must be a string, int, or
                               a pandas Series""")
    if clustercol is not None:
        if not isinstance(clustercol,(string_types,int_types,pd.Series)):
            raise TypeError("""'clustercol' must be a string, int, or
                               a pandas Series""")
    if savecol is not None:
        if not isinstance(savecol,pd.Series):
            raise TypeError("""'savecol' must be a pandas Series""")

    """
    plot settings: We add default values to avoid error when kwarg is not
    passed. When None is passed as a default value, or the user passes None,
    the get() call will return None, and then None will have to face the
    tribunal below. Sometimes it's okay, sometimes not.
    """
    thumb = kwargs.get('thumb',100)
    sample = kwargs.get('sample',100)
    idx = kwargs.get('idx',False)
    bg = kwargs.get('bg')
    ascending = kwargs.get('ascending',False)
    shape = kwargs.get('shape','square')
    ncols = kwargs.get('ncols',2)
    rounding = kwargs.get('rounding','up')
    bins = kwargs.get('bins',35)
    coordinates = kwargs.get('coordinates','cartesian')
    side = kwargs.get('side',500)
    xdomain = kwargs.get('xdomain') # needs to be None sometimes
    ydomain = kwargs.get('ydomain') # needs to be None sometimes
    xbins = kwargs.get('xbins') # needs to be None sometimes
    ybins = kwargs.get('ybins') # needs to be None sometimes
    savedir = kwargs.get('savedir',None)
    feature = kwargs.get('feature','brightness')
    aggregate = kwargs.get('aggregate',True)
    scale = kwargs.get('scale',True)
    outline = kwargs.get('outline')
    X = kwargs.get('X',pd.DataFrame())
    method = kwargs.get('method','kmeans')
    k = kwargs.get('k',4)
    i = kwargs.get('i')
    centroids = kwargs.get('centroids') # will be None sometimes
    normtype = kwargs.get('normtype','featscale')
    C = kwargs.get('C',0)
    j = kwargs.get('j')
    glyphtype = kwargs.get('glyphtype','radar')
    df = kwargs.get('df',pd.DataFrame())
    aes = kwargs.get('aes',{})
    border = kwargs.get('border',True)
    mat = kwargs.get('mat',True)
    verbose = kwargs.get('verbose',False)
    radii = kwargs.get('radii',True)
    gridlines = kwargs.get('gridlines',True)
    outlinewidth = kwargs.get('outlinewidth',2)
    gridlinefill = kwargs.get('gridlinefill','lightgray')
    gridlinewidth = kwargs.get('gridlinewidth',1)
    axislabels = kwargs.get('axislabels',False)
    alpha = kwargs.get('alpha',1.0)
    plot = kwargs.get('plot',True)
    flip = kwargs.get('flip',False)
    dot = kwargs.get('dot',False)
    bincols = kwargs.get('bincols',1)
    xaxis = kwargs.get('xaxis')
    yaxis = kwargs.get('yaxis')
    sigma = kwargs.get('sigma',1)
    gain = kwargs.get('gain',250)
    N = kwargs.get('N',1365)
    include_dir = kwargs.get('include_dir',False)
    low_pass_sigma = kwargs.get('low_pass_sigma',201)
    high_pass_sigma = kwargs.get('high_pass_sigma',5)
    plainsave = kwargs.get('plainsave',False)
    low_pass_apply = kwargs.get('low_pass_apply','subtract')
    axislines = kwargs.get('axislines',False)
    input_range = kwargs.get('input_range')
    output_range = kwargs.get('output_range',(0,1))
    legend = kwargs.get('legend',True)
    savemap = kwargs.get('savemap',False)
    textcolor = kwargs.get('textcolor')
    fill = kwargs.get('fill')

    """type checking"""
    if thumb!=False: # can only be false in show()
        if not isinstance(thumb,(int_types,tuple)):
            raise TypeError("'thumb' must be an integer or a tuple")
    if not isinstance(sample,int_types):
        raise TypeError("'sample' must be an integer")
    elif sample==True: # necessary bc True counts as int for some reason
        raise TypeError("'sample' must be an integer")
    if not isinstance(idx,bool):
        raise TypeError("'idx' must be True or False")
    if bg is not None:
        if not isinstance(bg,(tuple,string_types)):
            raise TypeError("'bg' must be an RGB triplet or a string")
    if not isinstance(ascending,bool):
        raise TypeError("'ascending' must be True or False")
    if not any([shape=='square',shape=='circle',shape=='rect',isinstance(shape,int_types)]):
        raise ValueError("""'shape' must be 'circle', 'square', or 'rect' or an
                             integer number of columns""")
    if not isinstance(ncols,int_types):
        raise TypeError("'ncols' must be an integer")
    if not any([rounding=='up',rounding=='down']):
        raise ValueError("'rounding' must be 'up' or 'down'")
    if not isinstance(bins,(int_types,seq_types)):
        raise TypeError("'bins' must be an integer or a sequence")
    if not any([coordinates=='cartesian',coordinates=='polar']):
        raise TypeError("'coordinates' must be 'cartesian' or 'polar'")
    if not isinstance(side,(int_types,list,tuple)):
        raise TypeError("'side' must be an integer, list, or tuple")
    if isinstance(side,(list,tuple)):
        if not len(side)==2:
            raise TypeError("'side' must have exactly 2 items if list or tuple")
    if xdomain is not None:
        if not isinstance(xdomain,(list,tuple)):
            raise TypeError("'xdomain' must be a list or tuple")
        if not len(xdomain)==2:
            raise ValueError("'xdomain' must be a two-item list or tuple")
    if ydomain is not None:
        if not isinstance(ydomain,(list,tuple)):
            raise TypeError("'xdomain' must be a list or tuple")
        if not len(ydomain)==2:
            raise ValueError("'xdomain' must be a two-item list or tuple")
    if xbins is not None:
        if not isinstance(xbins,(int_types,seq_types)):
            raise TypeError("'xbins' must be an integer or a sequence")
    if ybins is not None:
        if not isinstance(ybins,(int_types,seq_types)):
            raise TypeError("'ybins' must be an integer or a sequence")
    if savedir is not None:
        if not isinstance(savedir,string_types):
            raise TypeError("'savedir' must be a directory string")

    feats = [
    'brightness','saturation','hue','entropy','std','contrast',
    'dissimilarity','homogeneity','ASM','energy','correlation',
    'neural','tags','condition','roughness'
    ]
    if feature not in feats:
        raise ValueError("""'feature' must be one of 'brightness',
        'saturation','hue','entropy','std','contrast','dissimilarity',
        'homogeneity','ASM','energy','correlation','neural', 'tags', 'condition',
        or 'roughness'""")

    if not isinstance(aggregate,bool):
        raise TypeError("'aggregate' must be True or False")
    if not isinstance(scale,bool):
        raise TypeError("'scale' must be True or False")
    if outline is not None:
        if not isinstance(outline,(tuple,string_types)):
            raise TypeError("'outline' must be an RGB triplet or a string")
    if not isinstance(X,(pd.Series,pd.DataFrame)):
        raise TypeError("Feature matrix X must be a pandas Series or DataFrame")

    methods = [
    'kmeans','hierarchical','affinity','birch',
    'dbscan','hdbscan','minibatch','meanshift',
    'spectral'
    ]

    if method not in methods:
        raise TypeError("""'method' must be one of 'kmeans', 'hierarchical',
        'affinity','birch','dbscan','hdbscan','minibatch','meanshift', 
        or 'spectral'""")

    if not isinstance(k,int_types):
        raise TypeError("'k' must be an integer")
    if i is not None:
        if not isinstance(i,(int_types,seq_types)):
            raise TypeError("'i' must be an integer or sequence")
    if centroids is not None:
        if not isinstance(centroids,seq_types):
            raise TypeError("If passed, 'centroids' must be a sequence")

    normtypes = ['featscale','pct']

    if normtype not in normtypes:
        raise TypeError("""'normtype' must be one of 'featscale','pct'""")
    if C is not None:
        if not isinstance(C,(int_types,seq_types)):
            raise TypeError("'C' must be an integer or sequence")
    if j is not None:
        if not isinstance(j,(int_types,seq_types)):
            raise TypeError("'j' must be an integer or sequence")
    if fill is not None:
        if not isinstance(fill,(tuple,string_types,pd.Series)):
            raise TypeError("'fill' must be an RGB/A tuple, a string, or a pandas Series")

    glyphtypes = ['radar']

    if glyphtype not in glyphtypes:
        raise TypeError("""'glyphtype' must be one of 'radar'""")
    if not isinstance(df,pd.DataFrame):
        raise TypeError("'df' must be a pandas DataFrame")
    if not isinstance(aes,dict):
        raise TypeError("'aes' must be a dictionary")
    if not isinstance(border,bool):
        raise TypeError("'border' must be True or False")
    if not isinstance(mat,bool):
        raise TypeError("'mat' must be True or False")
    if not isinstance(verbose,bool):
        raise TypeError("'aggregate' must be True or False")
    if not isinstance(radii,bool):
        raise TypeError("'radii' must be True or False")
    if not isinstance(gridlines,bool):
        raise TypeError("'gridlines' must be True or False")
    if not isinstance(outlinewidth,int_types):
        raise TypeError("'outlinewidth' must be an integer")
    if not isinstance(gridlinefill,(tuple,string_types)):
            raise TypeError("'gridlinefill' must be an RGB triplet or a string")
    if not isinstance(gridlinewidth,int_types):
        raise TypeError("'gridlinewidth' must be an integer")
    if not isinstance(axislabels,(bool,seq_types)):
        raise TypeError("'axislabels' must be True, False, or a sequence")
    if not isinstance(alpha,(int_types,float_types)):
        raise TypeError("'alpha' must be a number between 0 and 1")
    if not all([alpha <= 1, alpha >= 0]):
        raise TypeError("'alpha' must be a number between 0 and 1")
    if not isinstance(plot,(bool,string_types)):
        raise TypeError("'plot' must be True, False, 'show', or 'montage'")
    if isinstance(plot,string_types):
        if plot not in ['show','montage']:
            raise TypeError("If a string is passed to 'plot', it must be 'show' or 'montage'")
    if not isinstance(flip,bool):
        raise TypeError("'flip' must be True or False")
    if not isinstance(dot,bool):
        raise TypeError("'dot' must be True or False")
    if not isinstance(bincols,int_types):
        raise TypeError("'bincols' must be an integer")
    if xaxis is not None:
        if not isinstance(xaxis,(int_types,bool)):
            raise TypeError("'xaxis' must be True, False, or an integer")
    if yaxis is not None:
        if not isinstance(yaxis,(int_types,bool)):
            raise TypeError("'yaxis' must be True, False, or an integer")
    if not isinstance(sigma,int_types):
        raise TypeError("'sigma' must be an integer")
    if not isinstance(gain,int_types):
        raise TypeError("'gain' must be an integer")
    if not isinstance(N,int_types):
        raise TypeError("'N' must be an integer")
    if not isinstance(include_dir,bool):
        raise TypeError("'include_dir' must be True or False")
    if not isinstance(low_pass_sigma,int_types):
        raise TypeError("'low_pass_sigma' must be an integer")
    if not isinstance(high_pass_sigma,int_types):
        raise TypeError("'high_pass_sigma' must be an integer")
    if not isinstance(plainsave,bool):
        raise TypeError("'plainsave' must be True or False")
    if low_pass_apply not in ['subtract','divide']:
        raise TypeError("'low_pass_apply' must be either 'subtract' or 'divide'")
    if not isinstance(axislines,bool):
        raise TypeError("'axislines' must be True or False")
    if not isinstance(output_range,(list,tuple)):
        raise TypeError("'output_range' must be a list or tuple")
    if not len(output_range)==2:
        raise ValueError("'output_range' must be a two-item list or tuple")
    if input_range is not None:    
        if not isinstance(input_range,(list,tuple)):
            raise TypeError("'input_range' must be a list or tuple")
        if not len(input_range)==2:
            raise ValueError("'input_range' must be a two-item list or tuple")
    if not isinstance(legend,bool):
        raise TypeError("'legend' must be True or False")
    if not isinstance(savemap,(bool,string_types)):
        raise TypeError("'savemap' must True, False, or a string defining a directory suffix")
    if textcolor is not None:
        if not isinstance(textcolor,(tuple,string_types)):
            raise TypeError("'textcolor' must be an RGB triplet or a string")
    

def attach(df,pathcol=None):

    if pathcol is None:
        raise ValueError("""Must supply variable 'pathcol', either as a string
            or integer that names a column of image paths in the supplied
            DataFrame, or as a pandas Series of image paths
            """)

    global ATTACHED_DATAFRAME
    global ATTACHED_PATHCOL

    ATTACHED_DATAFRAME = df

    if isinstance(pathcol,string_types):
        ATTACHED_PATHCOL = ATTACHED_DATAFRAME[pathcol]
    elif isinstance(pathcol,int_types):
        try:
            ATTACHED_PATHCOL = ATTACHED_DATAFRAME[pathcol]
        except:
            ATTACHED_PATHCOL = ATTACHED_DATAFRAME.iloc[:,pathcol]
    elif isinstance(pathcol,pd.Series):
        if pathcol.index.equals(ATTACHED_DATAFRAME.index):
            ATTACHED_PATHCOL = pathcol
        else:
            raise ValueError("""'pathcol' must have same indices as 'df'""")
    else:
        raise TypeError("'pathcol' must be a string, integer, or pandas Series")

def detach(df):
    """Resets global variables to None. Not likely to be used often, since
    attaching a new df and pathcol just replaces the old."""

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
               facetcol=None,
               notecol=None,
               scatter=False):

    """Index matching and working with attach()"""
    pathcol = _pathfilter(pathcol)
    xcol = _featfilter(pathcol,xcol)
    ycol = _featfilter(pathcol,ycol)
    facetcol = _featfilter(pathcol,facetcol)
    notecol = _featfilter(pathcol,notecol)

    """Selecting and sorting"""
    pathcol,xcol,ycol,facetcol,notecol = _dropna(pathcol,xcol,ycol,facetcol,
                                                 notecol)
    pathcol,xcol,ycol,facetcol,notecol = _sample(pathcol,xcol,ycol,
                                                 sample,facetcol,notecol)
    pathcol,xcol,ycol,facetcol,notecol = _sort(pathcol,xcol,ycol,
                                               ascending,facetcol,notecol,
                                               scatter)
    pathcol,xcol,ycol,facetcol,notecol = _subset(pathcol,xcol,ycol,
                                                 xdomain,ydomain,facetcol,
                                                 notecol)

    return pathcol,xcol,ycol,facetcol,notecol

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
        elif isinstance(col, (string_types, int)):
            if ATTACHED_DATAFRAME is None:
                raise TypeError("""No DataFrame attached. Feature variable
                                    must be a pandas Series""")
            else:
                tmp = copy.deepcopy(ATTACHED_DATAFRAME)
                try:
                    col = tmp[col] # if col a str or df column labels are ints
                except:
                    col = tmp.iloc[:,col] # if col is int and df col labels not
                if not col.index.equals(pathcol.index): # too strong criterion?
                    raise ValueError("""Image paths and image features must
                                        have same indices""")

    return col

def _dropna(pathcol,xcol,ycol,facetcol,notecol):
    """
    At time this function is called, all cols have same indices. We drop any
    rows with null values in each column, except notecol, which is not a
    plotting position column and so isn't strictly necessary.
    """
    setlist = []
    if xcol is not None:
        if np.mean(xcol.isnull()) > 0:
            xcol = xcol.dropna()
            setlist.append(set(xcol.index))
        else:
            setlist.append(set(xcol.index))
    if ycol is not None:
        if np.mean(ycol.isnull()) > 0:
            ycol = ycol.dropna()
            setlist.append(set(ycol.index))
        else:
            setlist.append(set(ycol.index))
    if facetcol is not None:
        if np.mean(facetcol.isnull()) > 0:
            facetcol = facetcol.dropna()
            setlist.append(set(facetcol.index))
        else:
            setlist.append(set(facetcol.index))

    if len(setlist) > 0:
        indices = set.intersection(*setlist)
        ndiff = len(pathcol.index) - len(indices)
        if ndiff > 0:
            print("removed " + str(ndiff) + " rows with missing data")
            pathcol = pathcol.loc[indices]
            if xcol is not None:
                xcol = xcol.loc[indices]
            if ycol is not None:
                ycol = ycol.loc[indices]
            if facetcol is not None:
                facetcol = facetcol.loc[indices]
            if notecol is not None:
                notecol = notecol.loc[indices]

    return pathcol,xcol,ycol,facetcol,notecol

def _sample(pathcol,xcol,ycol,sample,facetcol,notecol):

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
        if notecol is not None:
            notecol = notecol.loc[pathcol.index]

    return pathcol,xcol,ycol,facetcol,notecol

def _sort(pathcol,xcol,ycol,ascending,facetcol,notecol,scatter):

    """
    If user supplies xcol, we sort xcol and apply to path/ycol.
    At this point, if xcol is not None, we know it is a pandas Series
    (either user-supplied or from the attached df). We also know it is
    the same length as pathcol - even if pathcol got sampled above.

    Note that for scatterplots, sorting is unnecessary, and in fact, I now
    believe it to be undesired behavior in case you want to control the stacking
    order of your scatterplot, which you might. So, if scatter=True, there is no
    sort.

    Note that your working DataFrame is unchanged. Also note that it is never
    possible to sort globally by ycol. This is because only histogram and
    scatter have ycol, and for histograms, this sorting happens bin by bin.
    """
    if xcol is not None:
        if scatter==False: # sort only if non-scatter
            xcol = xcol.sort_values(ascending=ascending)
        pathcol = pathcol.loc[xcol.index]
        if ycol is not None: # nb sorting by ycol not possible globally
            ycol = ycol.loc[xcol.index]
        if facetcol is not None:
            facetcol = facetcol.loc[xcol.index]
        if notecol is not None:
            notecol = notecol.loc[xcol.index]

    return pathcol,xcol,ycol,facetcol,notecol

def _subset(pathcol,xcol,ycol,xdomain,ydomain,facetcol,notecol):
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
        if notecol is not None:
            notecol = notecol.loc[xcol.index]
    
    if ydomain is not None:
        if ycol is None:
            raise ValueError("""If 'ydomain' is supplied, 'ycol' must be
                                supplied as well""")

        ycol = ycol[(ycol>=ydomain[0])&(ycol<=ydomain[1])]
        pathcol = pathcol.loc[ycol.index]
        xcol = xcol.loc[ycol.index] # if ycol is not None, ditto xcol
        if facetcol is not None:
            facetcol = facetcol.loc[ycol.index]
        if notecol is not None:
            notecol = notecol.loc[ycol.index]

    return pathcol,xcol,ycol,facetcol,notecol

def _bin(col,bins):
    colname = col.name
    col = pd.cut(col,bins)
    leftbinedges = [float(str(item).split(",")[0].lstrip("([")) for item in col]

    # we set `name` here bc needed for axis titles, original col.name lost otherwise
    return pd.Series(leftbinedges,index=col.index,name=colname)

def _facet(**kwargs):
    kwargdict = copy.deepcopy(kwargs)
    facetcol = kwargs.get('facetcol')
    pathcol = kwargs.get('pathcol')
    xcol = kwargs.get('xcol')
    ycol = kwargs.get('ycol')
    xdomain = kwargs.get('xdomain')
    ydomain = kwargs.get('ydomain')
    notecol = kwargs.get('notecol')
    plottype = kwargs.get('plottype')
    coordinates = kwargs.get('coordinates')
    bins = kwargs.get('bins')

    """
    If we have xcol and ycol, but no user-passed domains, then we set them to
    the data extremes so that each facet can use the same domain. Conversely, if
    we do have user-passed domains, they need to dictate the plotting ranges.
    """

    if xcol is not None:
        if xdomain is None:
            xdomain = (xcol.min(),xcol.max())
    if ycol is not None:
        if ydomain is None:
            ydomain = (ycol.min(),ycol.max())

    facetlist = []
    vcounts = facetcol.value_counts()
    
    binmax = None
    
    #for val in np.sort(facetcol.unique()): # incl NaN but NaNs probably removed
    for val in vcounts.index:
        tmp = copy.deepcopy(kwargdict)

        try:
            del tmp['plottype']
        except:
            pass

        tmp['facettitle'] = str(val)

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
        if notecol is not None:
            tmp['notecol'] = notecol.loc[facetedcol.index]

        ###---------- Getting binmax for polar histogram
        if plottype=='histogram':
            if xdomain is not None:
                if isinstance(bins,int):
                    xbin = pd.cut(tmp['xcol'],np.linspace(xdomain[0],xdomain[1],bins+1),labels=False,include_lowest=True)
            else:
                xbin = pd.cut(tmp['xcol'],bins,labels=False,include_lowest=True)

            nonemptybins = xbin.unique() # will ignore empty bins
            facet_binmax = xbin.value_counts().max()
            if binmax is None:
                binmax = facet_binmax
            elif binmax is not None:
                if facet_binmax > binmax:
                    binmax = facet_binmax

        ###----------

        facetlist.append(tmp)

    return facetlist,binmax

def check_nan(cell):
    """
    This function checks whether a single value is NaN. I have no clue why this
    is not natively possible in pandas.
    """

    try:
        if pd.isnull(pd.Series(cell)[0]):
            return True
        else:
            return False
    except:
        return True
