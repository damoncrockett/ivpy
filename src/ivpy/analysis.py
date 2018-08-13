from annoy import AnnoyIndex
from .data import _typecheck,_pathfilter
from .plot import show

"""
Currently this function will use show() to display k nearest neighbors of i,
in feature space X. It will print the indices as well. Future iterations could
return some of this data for use in other processes, but for now it's merely a
quick (and approximate) visual analysis tool.
"""

def nearest(pathcol=None,X=None,i=None,k=None):
    if isinstance(pathcol,int): # allowable for show(), blocked by _paste()
        raise TypeError("'pathcol' must be a pandas Series")
    if X is None:
        raise ValueError("Must supply feature matrix 'X'")
    if i is None:
        raise ValueError("Must supply query point 'i'")
    if k is None:
        raise ValueError("Must supply number of neighbors 'k'")

    _typecheck(**locals())
    pathcol = _pathfilter(pathcol)

    f = X.shape[1] # number of columns in X
    t = AnnoyIndex(f)  # Length of item vector that will be indexed
    for j in range(len(X)):
        t.add_item(j,list(X.loc[j]))

    t.build(10) # 10 trees
    nns = t.get_nns_by_item(i,k,include_distances=False) # could use dists
    print(nns)
    return show(pathcol=pathcol.loc[nns],idx=True)
