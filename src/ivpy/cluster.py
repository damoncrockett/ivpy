import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation,AgglomerativeClustering,Birch
from sklearn.cluster import DBSCAN,FeatureAgglomeration,KMeans,MiniBatchKMeans
from sklearn.cluster import MeanShift,SpectralClustering
from .data import _typecheck
from six import string_types

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

    if isinstance(clustercol, string_types):
        ATTACHED_CLUSTERCOL = ATTACHED_CLUSTERFRAME[clustercol]
    elif isinstance(clustercol, int):
        try:
            ATTACHED_CLUSTERCOL = ATTACHED_CLUSTERFRAME[clustercol]
        except:
            ATTACHED_CLUSTERCOL = ATTACHED_CLUSTERFRAME.iloc[:,clustercol]
    elif isinstance(clustercol, pd.Series):
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

def _reassign_i(item,dst,clustercol):
    affiliation = clustercol.loc[item]
    if affiliation is None:
        print("Item at index",str(item),"has no cluster assignment")
    elif affiliation is not None:
        clustercol.loc[i] = dst
        if dst is None:
            print("Removed",str(item),"from cluster",str(int(affiliation)))
        elif dst is not None:
            print("Moved",
                  str(item),
                  "from cluster",
                  str(int(affiliation)),
                  "to cluster",
                  str(dst))

def _reassign_C(clusternum,dst,clustercol):
    n = len(clustercol[clustercol==clusternum])
    if n==0:
        print("Cluster",str(clusternum),"is empty")
    elif n > 0:
        clustercol[clustercol==clusternum] = dst
        if dst is None:
            print("Removed all",str(n),"members of cluster",str(clusternum))
        elif dst is not None:
            print("Moved all",
                  str(n),
                  "members of cluster",
                  str(clusternum),
                  "to cluster",
                  str(dst))

#------------------------------------------------------------------------------

def cut(i=None,C=None,clustercol=None):
    _typecheck(**locals())
    clustercol = _clusterfilter(clustercol)

    if all([i is None, C is None]):
        raise ValueError("Must supply either 'i' or 'C' or both")

    if i is not None:
        if isinstance(i,(int.np.int64)):
            _reassign_i(i,None,clustercol)
        elif isinstance(i,(pd.Series,list,tuple,np.ndarray)):
            for item in i:
                _reassign_i(item,None,clustercol)

    if C is not None:
        if isinstance(C,(int,np.int64)):
            _reassign_C(C,None,clustercol)
        elif isinstance(C,(pd.Series,list,tuple,np.ndarray)):
            for clusternum in C:
                _reassign_C(clusternum,None,clustercol)

#------------------------------------------------------------------------------

def to(i,C,clustercol=None):
    _typecheck(**locals())
    clustercol = _clusterfilter(clustercol)

    if isinstance(C,(pd.Series,list,tuple,np.ndarray)):
        raise ValueError("Must choose a single destination cluster 'C'")
    elif isinstance(C,(int,np.int64)):
        if isinstance(i,(int,np.int64)):
            _reassign_i(i,C,clustercol)
        elif isinstance(i,(pd.Series,list,tuple,np.ndarray)):
            for item in i:
                _reassign_i(item,C,clustercol)

def merge(*args,clustercol=None):
    _typecheck(**locals())
    clustercol = _clusterfilter(clustercol)

    typelist = [isinstance(item,(int,np.int64)) for item in args]
    if not all(typelist):
        raise TypeError("Arguments passed to 'merge' must be integers")

    dst = args[-1]
    to_reassign = args[:-1]

    for arg in to_reassign:
        _reassign_C(arg,dst,clustercol)

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
