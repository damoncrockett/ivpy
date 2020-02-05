import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation,AgglomerativeClustering,Birch
from sklearn.cluster import DBSCAN,FeatureAgglomeration,KMeans,MiniBatchKMeans
from sklearn.cluster import MeanShift,SpectralClustering
from six import string_types
from .plot import show
from .data import _typecheck,_pathfilter,_featfilter,int_types,seq_types
from .data import check_nan

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

ATTACHED_CLUSTERFRAME = None
ATTACHED_CLUSTERCOL = None

def correct(df,clustercol=None):

    if clustercol is None:
        raise ValueError("""Must supply variable 'clustercol', either as a
        string or integer that names a column in the supplied DataFrame, or as a
        pandas Series""")

    global ATTACHED_CLUSTERCOL
    global ATTACHED_CLUSTERFRAME

    ATTACHED_CLUSTERFRAME = df # no deep copy bc we want df to change

    if isinstance(clustercol,string_types):
        ATTACHED_CLUSTERCOL = ATTACHED_CLUSTERFRAME[clustercol]
    elif isinstance(clustercol,int_types):
        try:
            ATTACHED_CLUSTERCOL = ATTACHED_CLUSTERFRAME[clustercol]
        except:
            ATTACHED_CLUSTERCOL = ATTACHED_CLUSTERFRAME.iloc[:,clustercol]
    elif isinstance(clustercol,pd.Series):
        if clustercol.index.equals(ATTACHED_CLUSTERFRAME.index):
            ATTACHED_CLUSTERCOL = clustercol
        else:
            raise ValueError("'clustercol' must have same indices as 'df'")
    else:
        raise TypeError("'clustercol' must be a string, int, or pandas Series")

def _clusterfilter(clustercol):

    global ATTACHED_CLUSTERCOL

    if clustercol is None:
        if ATTACHED_CLUSTERCOL is None:
            raise ValueError("Must call correct() first or supply 'clustercol'")
        else:
            clustercol = ATTACHED_CLUSTERCOL

    return clustercol

#------------------------------------------------------------------------------

def _reassign_i(item,reassignment,clustercol):
    assignment = clustercol.loc[item]
    if check_nan(assignment):
        print("Item at index",str(item),"has no cluster assignment")
    elif not check_nan(assignment):
        clustercol.loc[item] = reassignment
        if reassignment is None:
            print("Removed",str(item),"from cluster",str(int(assignment)))
        elif reassignment is not None:
            print("Moved",
                  str(item),
                  "from cluster",
                  str(int(assignment)),
                  "to cluster",
                  str(reassignment))

def _reassign_C(clusternum,reassignment,clustercol):
    n = len(clustercol[clustercol==clusternum])
    if n==0:
        print("Cluster",str(clusternum),"is empty")
    elif n > 0:
        clustercol[clustercol==clusternum] = reassignment
        if reassignment is None:
            print("Removed all",str(n),"members of cluster",str(clusternum))
        elif reassignment is not None:
            print("Moved all",
                  str(n),
                  "members of cluster",
                  str(clusternum),
                  "to cluster",
                  str(reassignment))

#------------------------------------------------------------------------------

def cut(i=None,C=None,clustercol=None):
    _typecheck(**locals())
    clustercol = _clusterfilter(clustercol)

    if all([i is None, C is None]):
        raise ValueError("Must supply either 'i' or 'C' or both")

    if i is not None:
        if isinstance(i,int_types):
            _reassign_i(i,None,clustercol)
        elif isinstance(i,seq_types):
            for item in i:
                _reassign_i(item,None,clustercol)

    if C is not None:
        if isinstance(C,int_types):
            _reassign_C(C,None,clustercol)
        elif isinstance(C,seq_types):
            for clusternum in C:
                _reassign_C(clusternum,None,clustercol)

#------------------------------------------------------------------------------

def to(i,C,clustercol=None):
    _typecheck(**locals())
    clustercol = _clusterfilter(clustercol)

    if isinstance(C,seq_types):
        raise ValueError("Must choose a single destination cluster 'C'")
    elif isinstance(C,int_types):
        if isinstance(i,int_types):
            _reassign_i(i,C,clustercol)
        elif isinstance(i,seq_types):
            for item in i:
                _reassign_i(item,C,clustercol)
    elif C is None:
        cut(i)

#------------------------------------------------------------------------------

def merge(*args,clustercol=None):
    _typecheck(**locals())
    clustercol = _clusterfilter(clustercol)

    typelist = [isinstance(item,int_types) for item in args]
    if not all(typelist):
        raise TypeError("Arguments passed to 'merge' must be integers")

    reassignment = args[-1]
    to_reassign = args[:-1]

    for arg in to_reassign:
        _reassign_C(arg,reassignment,clustercol)

#------------------------------------------------------------------------------

def new(*args,clustercol=None):
    _typecheck(**locals())
    clustercol = _clusterfilter(clustercol)
    typelist = [isinstance(item,(list,int_types)) for item in args]
    if not all(typelist):
        raise TypeError("""Arguments passed to 'new' must be integers or lists
        of integers""")

    args = [[item] if not isinstance(item,list) else item for item in args]
    args = [item for sublist in args for item in sublist]
    args = list(set(args)) # eliminate repeats
    typelist = [isinstance(item,int_types) for item in args]
    if not all(typelist):
        raise TypeError("""Arguments passed to 'new' must be integers or
        sequences of integers""")

    extant = [int(item) for item in clustercol.unique() if not check_nan(item)]

    clusternum = 0
    while clusternum in extant:
        clusternum+=1

    clustercol.loc[args] = clusternum
    print(str(len(args)),"items moved to new cluster",str(clusternum))

#------------------------------------------------------------------------------

def swap(i,j,clustercol=None):
    _typecheck(**locals())
    clustercol = _clusterfilter(clustercol)

    if isinstance(i,seq_types):
        assignment_i = set(clustercol.loc[i])
        if len(assignment_i) > 1:
            raise ValueError("""Indices passed to 'i' must belong to same
            cluster""")
    if isinstance(j,seq_types):
        assignment_j = set(clustercol.loc[j])
        if len(assignment_j) > 1:
            raise ValueError("""Indices passed to 'j' must belong to same
            cluster""")
    try:
        assignment_i = int(list(assignment_i)[0])
    except:
        assignment_i = int(clustercol.loc[i])
    try:
        assignment_j = int(list(assignment_j)[0])
    except:
        assignment_j = int(clustercol.loc[j])

    if assignment_i==assignment_j:
        raise ValueError("Items belong to same cluster")

    to(i,assignment_j)
    to(j,assignment_i)

#------------------------------------------------------------------------------

def roster(C,clustercol=None,pathcol=None,notecol=None,thumb=False):
    if isinstance(pathcol,int): # allowable for show(), blocked by _paste()
        raise TypeError("'pathcol' must be a pandas Series")
    _typecheck(**locals())

    pathcol = _pathfilter(pathcol)
    notecol = _featfilter(pathcol,notecol)
    clustercol = _clusterfilter(clustercol)

    if not isinstance(C,int_types):
        raise TypeError("Can only pass a single cluster number to 'roster'")

    idxs = clustercol.index[clustercol==C]

    if notecol is None:
        return show(pathcol=pathcol.loc[idxs],idx=True,thumb=thumb)
    elif notecol is not None:
        return show(pathcol=pathcol.loc[idxs],
                    idx=True,
                    notecol=notecol.loc[idxs],
                    thumb=thumb)

#------------------------------------------------------------------------------

def idx(C,clustercol=None):
    _typecheck(**locals())
    clustercol = _clusterfilter(clustercol)

    if not isinstance(C,int_types):
        raise TypeError("Can only pass a single cluster number to 'idx'")

    idxs = clustercol.index[clustercol==C]
    return list(idxs)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

"""
usage:

df['centrality'] = centrality(X)
montage(xcol='centrality',facetcol='cluster',shape='circle')
"""

def _centrality(centroid,pt):
    return np.linalg.norm(centroid-pt)

def centrality(X,clustercol=None):
    _typecheck(**locals())
    clustercol = _clusterfilter(clustercol)

    distcol = pd.Series(index=clustercol.index)
    clusternums = list(clustercol.value_counts().index)
    for clusternum in clusternums:
        idxs = clustercol.index[clustercol==clusternum]
        tmp = X.loc[idxs]
        centroid = np.array(tmp.apply(np.mean))
        for idx in idxs:
            row = np.array(tmp.loc[idx])
            dist = _centrality(centroid,row)
            distcol.loc[idx] = dist

    return distcol

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

"""
Note: it's possible that if the user passes an X whose index doesn't match
the pathcol they'll ultimately use to plot, you could have an indexing issue.
Not really any way to prevent that, but it's unlikely to happen because X will
often be obtained using extract(), which guards against it.
"""

def cluster(X,method='kmeans',k=4,centroids=None,**kwargs):
    _typecheck(**locals())

    if method=='kmeans':
        if centroids is not None:
            k = len(centroids)
            print("method:",method,"\nnumber of clusters:",str(k))
            return _cluster(X,
                            KMeans,
                            n_clusters=k,
                            init=X.loc[centroids],
                            **kwargs)
        elif centroids is None:
            print("method:",method,"\nnumber of clusters:",str(k))
            return _cluster(X,KMeans,n_clusters=k,**kwargs)

    elif method=='hierarchical':
        print("method:",method,"\nnumber of clusters:",str(k))
        return _cluster(X,
                        AgglomerativeClustering,
                        n_clusters=k,
                        **kwargs)

    elif method=='affinity':
        print("method:",method)
        return _cluster(X,
                        AffinityPropagation,
                        **kwargs)

    elif method=='birch':
        print("method:",method,"\nnumber of clusters:",str(k))
        return _cluster(X,
                        Birch,
                        n_clusters=k,
                        **kwargs)

    elif method=='dbscan':
        print("method:",method)
        return _cluster(X,
                        DBSCAN,
                        **kwargs)

    elif method=='minibatch':
        if centroids is not None:
            k = len(centroids)
            print("method:",method,"\nnumber of clusters:",str(k))
            return _cluster(X,
                            MiniBatchKMeans,
                            n_clusters=k,
                            init=X.loc[centroids],
                            **kwargs)
        elif centroids is None:
            print("method:",method,"\nnumber of clusters:",str(k))
            return _cluster(X,MiniBatchKMeans,n_clusters=k,**kwargs)

    elif method=='meanshift':
        print("method:",method)
        return _cluster(X,
                        MeanShift,
                        **kwargs)

    elif method=='spectral':
        print("method:",method,"\nnumber of clusters:",str(k))
        return _cluster(X,
                        SpectralClustering,
                        n_clusters=k,
                        **kwargs)

def _cluster(X,func,**kwargs):
    fitted = func(**kwargs).fit(X)
    return pd.Series(fitted.labels_,index=X.index)
