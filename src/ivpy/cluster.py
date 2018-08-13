import pandas as pd
from sklearn.cluster import KMeans,AgglomerativeClustering
from .data import _typecheck

#------------------------------------------------------------------------------

"""
Note: it's possible that if the user passes an X whose index doesn't match 
the pathcol they'll ultimately use to plot, you could have an indexing issue.
Not really any way to prevent that, but it's unlikely to happen because X will
often be obtained using extract(), which guards against it.
"""

def cluster(X,method=None,k=None,**kwargs):
    if method is None:
        raise ValueError("Must supply 'method'")
    if k is None:
        raise ValueError("Must supply number of clusters 'k'")

    _typecheck(**locals())

    if method=='kmeans':
        return _cluster(X,KMeans,n_clusters=k,**kwargs)
    elif method=='hierarchical':
        return _cluster(X,AgglomerativeClustering,n_clusters=k,**kwargs)

def _cluster(X,func,**kwargs):
    fitted = func(**kwargs).fit(X)
    return pd.Series(fitted.labels_,index=X.index)
