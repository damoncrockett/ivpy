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

def _cluster(X,func,**kwargs):
    fitted = func(**kwargs).fit(X)
    return pd.Series(fitted.labels_,index=X.index)
