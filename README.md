# ivpy
Iconographic Visualization in Python

### Tutorial Dataset

To avoid data access issues, I've written the tutorials using the publicly available [Oxford Flower 17 dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/). It contains 80 images each of 17 different flower types. I've included a data table, 'oxfordflower.csv', in the ivpy repo, but you'll need to download the images themselves [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz). The 'filename' column in 'oxfordflower.csv' corresponds to the filenames in the linked archive.

### A word about Python versions and virtual environments:

I officially recommend using Python 3. I recently was unable to install the dependencies using pip in Python 2.7, and the problem has to do with lack of support for new SSL/TSL protocols. See [this thread](https://github.com/pypa/get-pip/issues/26) for more information. 

Best thing to do, in my opinion, is to install Python 3, if you haven't already, and use `venv` to create a virtual environment. The dependencies can all be installed using the requirements.txt file, as described below. It is not strictly necessary that you use a virtual environment, but it's the failsafe approach.

### Dependencies 

pandas, numpy, Pillow, Shapely, jupyter (if using inside notebook)

### Dependencies for Feature Extraction

TensorFlow, scipy, scikit-image, scikit-learn, Keras (which itself may require a separate h5py install)

### Additional Dependencies

umap-learn (for umap embedding, found [here](https://github.com/lmcinnes/umap))

### Install & Run

0. Clone this repo:

`$ git clone https://github.com/damoncrockett/ivpy`

1. Create Python 3 virtual environment using venv:

`$ python3 -m venv myEnv`

note: this will create a virtual environment directory called 'myEnv' inside whatever directory you are currently working in. If you want to put it somewhere else, you need to specify a full path. And you can of course name it whatever you want.

2. Activate virtual environment:

`$ source myEnv/bin/activate`

3. Install requirements:

`$ cd ivpy`

`$ pip install -r requirements.txt`

4. Create .jupyter/custom/ in your home folder, and copy ivpy/style/custom.css there

5. Run the jupyter notebook server in ivpy/src/:

`$ cd src`

`$ jupyter notebook`

note: The reason I recommend starting a server inside the ivpy/src is that the tutorial notebooks live there, and the way they import ivpy functions requires that they live there. Once the software is ready for beta, it will be pip-installable, and this won't be an issue (because the install will add the module to some directory in your Python path).

### Working on your own notebooks and updating ivpy

The above sequence will enable you to run the tutorial notebooks. If you start your own notebooks, it is easiest to simply keep them in ivpy/src. If you don't, you'll need the following Python code to import ivpy:

`import sys`

`sys.path.append("/Users/damoncrockett/ivpy/src/")` (You'll need to change this to reflect the path on your machine)

`from ivpy import attach,show,compose,montage,histogram,scatter,extract` (or whichever functions you want)

I should also point out a potential danger with keeping notebooks inside ivpy/src. If they happen to have the same filename as one of the tutorial notebooks---if, for example, you started adding your own code cells to a tutorial notebook instead of opening a new notebook---then running `$ git pull` in the ivpy directory will re-write those files with the original tutorial notebooks. I want users to be able to easily pull any new changes to the software (and there are lots of those changes being made right now), but I don't want anyone to lose any work! So make sure you give your notebooks new names, and try to avoid doing any serious work inside the tutorial notebooks.
