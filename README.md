# ivpy
Iconographic Visualization in Python

### Tutorial Dataset

To avoid data access issues, I've written the tutorials using the publicly available [Oxford Flower 17 dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/). It contains 80 images each of 17 different flower types. I've included a data table, 'oxfordflower.csv', in the ivpy repo, but you'll need to download the images themselves [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz). The 'filename' column in 'oxfordflower.csv' corresponds to the filenames in the linked archive.

### A word about Python versions and virtual environments:

I officially recommend using Python 3. I recently was unable to install the dependencies using pip in Python 2.7, and the problem has to do with lack of support for new SSL/TSL protocols. See [this thread](https://github.com/pypa/get-pip/issues/26) for more information. 

Best thing to do, in my opinion, is to install Python 3, if you haven't already, and use `venv` to create a virtual environment. It is not strictly necessary that you use a virtual environment, but it's the failsafe approach. Originally, I had a requirements.txt file containing all dependencies, but updates to various packages make such a file nearly impossible to maintain over time. I recommend installing the dependencies one by one using pip3.

### My current working configuration

When you have---as we do here---a module that depends on lots of other modules, stuff breaks over time as things get updated. For example, tensorflow is not currently compatbile with Python 3.7 (as of Sep 21, 2018). So I will describe here my current configuration, which works. If it breaks, I'll fix it and update this part of the README.

iMac Retina 5K, 27-inch, 2017

macOS High Sierra 10.13.6

Python 3.6.5 (remember to run Install Certificates.command after installing)

pandas==0.23.4

Pillow==5.2.0

jupyter==1.0.0

tensorflow==1.3.0 (this is pretty old)

scipy==1.1.0

scikit-image==0.14.0

scikit-learn==0.19.2

Keras==2.1.0 (install tensorflow and h5py first)

umap-learn==0.3.2

annoy==1.13.0 (install nose first)


### Dependencies 

pandas, numpy, Pillow, jupyter (if using inside notebook)

### Dependencies for Feature Extraction

tensorflow (may need to specify version, e.g., tensorflow==1.3.0), scipy, scikit-image, scikit-learn, keras (install h5py first)

### Additional Dependencies

umap-learn (for umap embedding, found [here](https://github.com/lmcinnes/umap))

annoy (install nose first) (for nearest neighbor search)

### Install & Run

0. Clone this repo:

`$ git clone https://github.com/damoncrockett/ivpy`

1. Create Python 3 virtual environment using venv:

`$ python3 -m venv myEnv`

note: this will create a virtual environment directory called 'myEnv' inside whatever directory you are currently working in. If you want to put it somewhere else, you need to specify a full path. And you can of course name it whatever you want.

2. Activate virtual environment:

`$ source myEnv/bin/activate`

3. Install requirements:

`$ pip3 install [package name]`

4. Create .jupyter/custom/ in your home folder, and copy ivpy/style/custom.css there

5. Run the jupyter notebook server in ivpy/src/:

`$ cd src`

`$ jupyter notebook`

note: The reason I recommend starting a server inside the ivpy/src is that the tutorial notebooks live there, and the way they import ivpy functions requires that they live there. Once the software is ready for beta, it will be pip-installable, and this won't be an issue (because the install will add the module to some directory in your Python path).

### Working on your own notebooks and updating ivpy

The above sequence will enable you to run the tutorial notebooks. If you start your own notebooks, it is easiest to simply keep them in ivpy/src. If you don't, you'll need the following Python code to import ivpy:

`import sys`

`sys.path.append("/Users/damoncrockett/ivpy/src/")` (You'll need to change this to reflect the path on your machine)

`from ivpy import attach,show,compose,montage,histogram,scatter` (or whichever functions you want)

You will also need to move the 'fonts' folder into the parent directory of your working directory.

### Pulling new changes to ivpy

I should also point out a potential danger with keeping notebooks inside ivpy/src. If they happen to have the same filename as one of the tutorial notebooks---if, for example, you started adding your own code cells to a tutorial notebook instead of opening a new notebook---then running `$ git pull` in the ivpy directory will re-write those files with the original tutorial notebooks. I want users to be able to easily pull any new changes to the software (and there are lots of those changes being made right now), but I don't want anyone to lose any work! So make sure you give your notebooks new names, and try to avoid doing any serious work inside the tutorial notebooks.
