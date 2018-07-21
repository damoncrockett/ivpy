import pandas as pd
from sklearn.manifold import TSNE
import umap as ump

#------------------------------------------------------------------------------

def tsne(X,**kwargs):
    xy = TSNE(n_components=2,**kwargs).fit_transform(X)
    return pd.DataFrame(xy,columns=['x','y'])

def umap(X,**kwargs):
    xy = ump.UMAP(**kwargs).fit_transform(X)
    return pd.DataFrame(xy,columns=['x','y'])
