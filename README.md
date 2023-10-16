# ivpy: Iconographic Visualization in Python

Read the journal article [here](https://journals.ub.uni-heidelberg.de/index.php/dah/article/view/66401).

## Tutorial Dataset

A detailed guide to using ivpy is included here, in the [tutorial notebooks](src/). To avoid data access issues, I've written the tutorials using the publicly available [Oxford Flower 17 dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/). It contains 80 images each of 17 different flower types. I've included a data table, `oxfordflower.csv`, in the ivpy repo, but you'll need to download the images themselves [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz). The `filename` column in `oxfordflower.csv` corresponds to the filenames in the linked archive.

## Basic Install

I recommend cloning this repo into your home directory, creating and activating a Python virtual environment, and pip installing the dependencies:

```shell
$ git clone https://github.com/damoncrockett/ivpy
$ python3 -m venv ivpy_env
$ source ivpy_env/bin/activate
$ pip install numpy notebook pandas Pillow
```

### Custom CSS theme

If you want to use my custom jupyter theme, optimized for viewing image visualizations:

```shell
$ mkdir ~/.jupyter/custom
$ cp ~/ivpy/style/custom.css ~/.jupyter/custom
```

### Fonts

Some of the plots include text labels, and for that, I include a directory of fonts, because the defaults in PIL are pretty bad. You'll need to copy the font directory to your home directory:

```shell
$ cp -r ~/ivpy/fonts ~
```

## Basic Usage

```python
import pandas as pd
import sys,os
sys.path.append(os.path.expanduser("~") + "/ivpy/src")
from ivpy import *

df = pd.DataFrame(...)
imagecol = '...'

attach(df, imagecol)

# scrollable panel of images / glyphs
show()

# by default, a square montage of images
montage()

# image histogram
histogram(xcol='foo')

# image scatterplot
scatter(xcol='foo', ycol='bar')
```

## Additional functionality

There is a great deal more in ivpy, including [image feature extraction](src/ivpy/extract.py) (requires additional dependencies, like `scikit-image` and others); [glyph drawing](src/ivpy/glyph.py); [line plots](src/ivpy/plot.py); [clustering algorithms](src/ivpy/cluster.py) and [dimension reduction](src/ivpy/reduce.py) (with `scikit-learn` under the hood); [image resizing and slicing](src/ivpy/utils.py); [image signal processing](src/ivpy/utils.py); and even [nearest neighbor search](src/ivpy/analysis.py) (using `annoy`).
