# ivpy
Iconographic Visualization in Python

### A word about Python versions and virtual environments:

I officially recommend using Python 3. I recently was unable to install the dependencies using pip in Python 2.7, and the problem has to do with lack of support for new SSL/TSL protocols. See [this thread](https://github.com/pypa/get-pip/issues/26) for more information. 

Best thing to do, in my opinion, is to install Python 3, if you haven't already, and use `venv` to create a virtual environment. The dependencies can all be installed using the requirements.txt file, as described below. It is not strictly necessary that you use a virtual environment, but it's the failsafe approach.

### Dependencies 

pandas, numpy, Pillow, Shapely, jupyter (if using inside notebook)

### Dependencies for Feature Extraction

TensorFlow, scipy, scikit-image, Keras (which itself may require a separate h5py install)

### Install & Run

0. Clone this repo:

`$ git clone https://github.com/damoncrockett/ivpy`

1. Create Python 3 virtual environment using venv:

`$ python3 -m venv myEnv`

note: this will create a virtual environment directory called 'myEnv' inside whatever directory you are currently working in. If you want to put it somewhere else, you need to specify a full path.

2. Activate virtual environment:

`$ source myEnv/bin/activate`

3. Install requirements:

`$ cd ivpy`

`$ pip install -r requirements.txt`

4. Create .jupyter/custom/ in your home folder, and copy ivpy/style/custom.css there

5. Run the jupyter notebook server in ivpy/src/:

`$ cd src`

`$ jupyter notebook`
