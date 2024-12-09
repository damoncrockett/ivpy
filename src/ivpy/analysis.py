from annoy import AnnoyIndex
from .data import _typecheck,_pathfilter,_featfilter,seq_types
from .plot import show,montage
import numpy as np

"""
Currently this function will use show() to display k nearest neighbors of i,
in feature space X. It will print the indices as well. Future iterations could
return some of this data for use in other processes, but for now it's merely a
quick (and approximate) visual analysis tool.
"""

def nearest(X=None,
            i=None,
            pathcol=None,
            k=4,
            notecol=None,
            thumb=False,
            bg='white',
            plot=True):

    if isinstance(pathcol,int): # allowable for show(), blocked by _paste()
        raise TypeError("'pathcol' must be a pandas Series")
    if X is None:
        raise ValueError("Must supply feature matrix 'X'")
    if i is None:
        i = np.random.choice(X.index)
    if isinstance(i,seq_types): # can be seq in cut()
        raise ValueError("Must choose a single 'i' as target")

    _typecheck(**locals())
    pathcol = _pathfilter(pathcol)
    notecol = _featfilter(pathcol,notecol)

    f = X.shape[1] # number of columns in X
    t = AnnoyIndex(f, metric='angular')  # Length of item vector that will be indexed

    for idx, item in enumerate(X.index):
        t.add_item(idx, list(X.loc[item]))


    idmap = dict(zip(X.index,list(range(len(X)))))
    idmapReverse = dict(zip(list(range(len(X))),X.index))

    t.build(10) # 10 trees
    nnsAnnoy = t.get_nns_by_item(idmap[i], k, include_distances=False) # dists?
    nnsNative = [idmapReverse[item] for item in nnsAnnoy]

    if plot in [True,'show']:
        print(nnsNative)
        if notecol is None:
            return show(pathcol=pathcol.loc[nnsNative],idx=True,thumb=thumb,bg=bg)
        elif notecol is not None:
            return show(pathcol=pathcol.loc[nnsNative],
                        idx=True,
                        notecol=notecol.loc[nnsNative],
                        thumb=thumb,bg=bg)
    elif plot=='montage':
        print(nnsNative)
        if notecol is None:
            return montage(pathcol=pathcol.loc[nnsNative],idx=True,thumb=thumb,bg=bg)
        elif notecol is not None:
            return montage(pathcol=pathcol.loc[nnsNative],
                        idx=True,
                        notecol=notecol.loc[nnsNative],
                        thumb=thumb,bg=bg)
    elif plot==False:
        return nnsNative
