import pandas as pd
from sklearn.manifold import TSNE
import umap as ump

#------------------------------------------------------------------------------

def tsne(X,**kwargs):
    if not isinstance(X,pd.DataFrame):
        raise TypeError("Feature matrix must be a pandas DataFrame")

    xy = TSNE(n_components=2,**kwargs).fit_transform(X)
    return pd.DataFrame(xy,index=X.index,columns=['x','y'])

def umap(X,**kwargs):
    if not isinstance(X,pd.DataFrame):
        raise TypeError("Feature matrix must be a pandas DataFrame")

    xy = ump.UMAP(**kwargs).fit_transform(X)
    return pd.DataFrame(xy,index=X.index,columns=['x','y'])
