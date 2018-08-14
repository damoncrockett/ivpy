import os
from .data import _colfilter,_typecheck

def resize(savedir=None,pathcol=None,thumb=256):
    """Creates thumbnails of images, saves to 'savedir'. Had considered default
       'savedir', but decided user should always supply 'savedir', putting
       the write responsibility on them. Could be dissuaded here."""

    if savedir==None:
        raise ValueError("Must supply 'savedir'")
    elif savedir is not None:
        if os.path.isdir(savedir)==True:
            pass
        else:
            try:
                os.mkdir(savedir)
            except:
                raise ValueError("'savedir' must be a valid filepath")

    _typecheck(**locals())
    pathcol,xcol,ycol,facetcol = _colfilter(pathcol)

    if isinstance(pathcol,string_types):
        return _resize(pathcol,savedir,thumb)

    elif isinstance(pathcol,pd.Series):
        pathcol_resized = pd.Series(index=pathcol.index)
        for i in pathcol.index:
            impath = pathcol.loc[i]
            pathcol_resized.loc[i] = _resize(impath,savedir,thumb)
        return pathcol_resized

def _resize(impath,savedir,thumb):
    im = Image.open(impath)
    im.thumbnail((thumb,thumb),Image.ANTIALIAS)
    basename = os.path.basename(impath)
    savestring = savedir + "/" + basename
    im.save(savestring)
    return savestring
