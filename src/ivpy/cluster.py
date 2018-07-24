import pandas as pd
from sklearn.cluster import KMeans,AgglomerativeClustering

#------------------------------------------------------------------------------

def cluster(X,method=None,k=None,**kwargs):
    if not isinstance(X,pd.DataFrame):
        raise TypeError("Feature matrix must be a pandas DataFrame")
    if k is None:
        raise ValueError("Must supply number of clusters 'k'")
    if method=='kmeans':
        return _cluster(X,KMeans,n_clusters=k,**kwargs)
    elif method=='hierarchical':
        return _cluster(X,AgglomerativeClustering,n_clusters=k,**kwargs)
    else:
        raise ValueError("""Must supply clustering method: 'kmeans' or
                            'hierarchical'""")

def _cluster(X,func,**kwargs):
    fitted = func(**kwargs).fit(X)
    return pd.Series(fitted.labels_,index=X.index)
