# ivpy
Iconographic Visualization in Python

### Dependencies 

pandas >= 0.22.0

numpy >= 1.9.0

Pillow >= 2.9.0

Shapely >= 1.5.7

jupyter >= 1.0.0 (if using inside notebook)

scikit-image >= 0.11.3 (if using feature extraction)

h5py >= 2.7.1

### Install

0. Either fork or clone this repo:

``$ git clone https://github.com/damoncrockett/ivpy``

1. Create Python virtual environment (not strictly necessary, but recommended) using virtualenv or, if you are using Python 3, venv:

``$ virtualenv myEnv``

2. Activate virtual environment:

``$ source myEnv/bin/activate``

3. Install requirements (you'll need to switch to the ivpy directory first):

``$ pip install -r requirements.txt``

4. Create .jupyter/custom/ in your home folder, and copy ivpy/style/custom.css there

5. Run the jupyter notebook server in ivpy/src/
