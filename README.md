# ivpy
Iconographic Visualization in Python

### Tutorial Dataset

To avoid data access issues, I've written the tutorials using the publicly available [Oxford Flower 17 dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/). It contains 80 images each of 17 different flower types. I've included a data table, 'oxfordflower.csv', in the ivpy repo, but you'll need to download the images themselves [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz). The 'filename' column in 'oxfordflower.csv' corresponds to the filenames in the linked archive.

### A word about Python versions and virtual environments:

I officially recommend using Python 3. I recently was unable to install the dependencies using pip in Python 2.7, and the problem has to do with lack of support for new SSL/TSL protocols. See [this thread](https://github.com/pypa/get-pip/issues/26) for more information. I'm sure there are workarounds, but it seems not worth it, since Python 2 is nearing [the end of its life](https://legacy.python.org/dev/peps/pep-0373/).

Best thing to do, in my opinion, is to install Python 3 (remember to run Install Certificates.command after installing!), if you haven't already, and use `venv` to create a virtual environment (see below). It is not strictly necessary that you use a virtual environment, but it's the failsafe approach.

### Basic Install & Run

0. Clone this repo:

`$ git clone https://github.com/damoncrockett/ivpy`

1. Create Python 3 virtual environment using venv:

`$ python3 -m venv myEnv`

note: this will create a virtual environment directory called 'myEnv' inside whatever directory you are currently working in. If you want to put it somewhere else, you need to specify a full path. And you can of course name it whatever you want.

2. Activate virtual environment:

`$ source myEnv/bin/activate`

3. Install requirements:

`$ pip3 install numpy`
`$ pip3 install jupyter`
`$ pip3 install pandas`
`$ pip3 install Pillow`

4. Optional (if you want to use my custom jupyter theme): Create .jupyter/custom/ in your home folder, and copy ivpy/style/custom.css there

5. Run the jupyter notebook server in ivpy/src/:

`$ cd src`

`$ jupyter notebook`

note: The reason I recommend starting a server inside the ivpy/src is that the tutorial notebooks live there, and the way they import ivpy functions requires that they live there. Once the software is ready for beta, it will be pip-installable, and this won't be an issue (because the install will add the module to some directory in your Python path).

### Working on your own notebooks and updating ivpy

The above sequence will enable you to run the tutorial notebooks. If you start your own notebooks, it is easiest to simply keep them in ivpy/src. If you don't, you'll need the following Python code to import ivpy:

`import sys`

`sys.path.append("/Users/damoncrockett/ivpy/src/")` (You'll need to change this to reflect the path on your machine)

`from ivpy import attach,show,compose,montage,histogram,scatter` (or whichever functions you want)
`from ivpy import *` (more concise but less explicit, and perhaps a potential for namespace conflicts)


You will also need to copy the 'fonts' folder to the parent directory of your working directory (one level up from where your notebooks are).

### Pulling new changes to ivpy

I should also point out a potential danger with keeping notebooks inside ivpy/src. If they happen to have the same filename as one of the tutorial notebooks---if, for example, you started adding your own code cells to a tutorial notebook instead of opening a new notebook---then running `$ git pull` in the ivpy directory will re-write those files with the original tutorial notebooks. I want users to be able to easily pull any new changes to the software (and there are lots of those changes being made right now), but I don't want anyone to lose any work! _So make sure you give your notebooks new names_, and try to avoid doing any serious work inside the tutorial notebooks.

### Additional Dependencies

The basic functionality, including everything in the tutorial notebooks, will work with just numpy, pandas, jupyter, and Pillow. 

For feature extraction / dimension reduction, you'll need to install these additional dependencies (I recommend this install order as well):

scipy
scikit-image
scikit-learn
tensorflow
h5py
keras
umap-learn

For nearest neighbor search, you'll need:

nose
annoy

For slicing images (utils.shatter):

image_slicer

### Basic Functionality Working Setup

As of March 4, 2021, the following setup will successfully run the tutorial notebooks:

macOS Catalina 10.15.7
Python 3.6.5
pip 9.0.3
numpy 1.19.5
jupyter 1.0.0
pandas 1.1.5
Pillow 8.1.1

