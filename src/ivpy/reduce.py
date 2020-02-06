import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap as ump
from .data import _typecheck

#------------------------------------------------------------------------------

def pca(X,**kwargs):
    _typecheck(**locals())
    xy = PCA(**kwargs).fit_transform(X)
    return pd.DataFrame(xy,index=X.index)

def tsne(X,**kwargs):
    _typecheck(**locals())
    xy = TSNE(**kwargs).fit_transform(X)
    return pd.DataFrame(xy,index=X.index)

def umap(X,**kwargs):
    _typecheck(**locals())
    xy = ump.UMAP(**kwargs).fit_transform(X)
    return pd.DataFrame(xy,index=X.index)
