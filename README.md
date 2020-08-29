# imageSearch: Biometric identification using images 

imageSearch is a Python3 console application for image identification, designed for wildlife biologists. 

* Manage a register of identified images.
* Verify and authentify new entries against previously registered images.

## Contents of the repository

* `files` contains images used to illustrate the Jupyter tutorial. 
* `imageSearch` contains the modules:
* *  `database.py` with the Python class `DB`, which is the core of the application.
* *  `utils.py` with auxiliary functions. 
* `test.py` tests the application. The utrecht face database, (see download data section) will be used to build a new register and train a new model. The script returns similarity probability for a sample and plot a matched pair of image.
* `UserGuide.ipynb` is a Jupyter Notebook presenting the main functionalities of the application. 
* `requirements.txt` is a list of dependencies of the appication.
* `environment.yml` can be used to create the virtual environment for the application with conda (recommended; see below).
* `model.pkg` is the default scikit-learn machine learning model used in the application, saved as a pickled Python object. 

## Data
This application was specifically built for verification of toad images, although it can suit many other purposes. The app was successfully tested (98% accuracy, refer to User Guide for definition) on an unseen dataset containing 127 pairs of images of toads' upper body. This data, as well as the [*Utrecht Face dataset*](http://pics.stir.ac.uk/2D_face_sets.htm), used as a toy example in the User Guide, is to be found [here](https://drive.google.com/drive/folders/1r_1X1777maJ8mBIArpi4eLQaTuLOjmH-?usp=sharing). 

## Install

The application requires installing the packages listed in `requirements.txt` in the project folder. To avoid compatibility issues
we strongly recommend using a virtual environment for the project. As we will see, creating and activating a virtual environment with **Anaconda** is very simple, but you can also 
use the **virtualenv** Python package for this purpose. 

### 1. Download the repository

Download the repository and save it in a place of your choice.

### 2. Create a virtual environment and install dependencies
#### If you don't have Anaconda
##### On Windows
* Install **virualenv** using pip from the shell:
```
pip install virtualenv
```
* Make a directory where you want save the virtual environment in. For instance Spyder:
```
mkdir C:\Users\user\Documents\my_virt_envs
cd C:\Users\user\Documents\my_virt_envs
```
* Create a new environment. E.g. imageSearch. Then activate it:
```
python -m venv imageSearch
imageSearch\Scripts\activate
```
* Install the requirements into the virtual environment using the `requirements.txt` file from the project folder:
```
pip install -r project_folder\requirements.txt 
```
* If you prefer to work with and IDE, install your favorite IDE in the virtual environment. For instance:
```
pip install spyder
```

You can deactivate the environment at any time and come back to your default Python environment with:
```
deactivate
```

##### On Linux
* Install **virualenv** using pip from the shell:
```
pip install virtualenv
```
* Make a directory where you want save the virtual environment in. For instance:
```
mkdir ./my_virt_envs
cd ./my_virt_envs
```
* Create a new environment. E.g. Then activate it:
```
python3 -m venv imageSearch
source imageSearch/bin/activate
```
* Install the requirements into the virtual environment using the `requirements.txt` file from the project folder:
```
pip install -r project_folder/requirements.txt 
```
* If you prefer to work with and IDE, install your favorite IDE in the virtual environment. For instance:
```
pip install spyder
```

You can deactivate the environment at any time and come back to your default Python environment with:
```
deactivate
```
#### If you have Anaconda

* Create a virtual environment with the `environment.yml` file in the project file; it will be automatically named **imageSearch** and will 
install all dependencies  at once (you can modify the name by changing the first line of the `.yml` file).
Type in the Anaconda shell (pay attention to the path of `environment.yml`): 
```
conda env create -f path/to/environment.yml
```
### 3. Using Jupyter Notebook in the the virtual environment
If you want to interact with the application from Jupyter, you need to:

* Activate your virtual envrionment:
* * With Linux (mind the location of your virtual environment)
```
source path/to/imageSearch/bin/activate
```
* * With Windows (mind the location of your virtual environment):
```
path\to\imageSearch\Scripts\activate
```
* * or with conda
```
conda activate imageSearch
```
* Install Jupyter and ipykernel in your virtual environment,
* * either with `pip`: 
 ```
 pip install notebook
 pip install ipykernel
 ```
* * or with conda:
 ```
 conda install -c conda-forge notebook
 conda install ipykernel
 ```
* Create a kernel with the virtual environment for your Jupyter:
```
python -m ipykernel install --user --name imageSearch --display-name "Python(imageSearch)"
```

When launching Jupyter, you will be able to create a notebook with a kernel in the virtual environment.

## Getting started
imageSearch is a Python console-based application, so you need to launch Python in your shell 
and run your commands from there.

* First, activate your environment:

* * Linux: 
```
source path/to/imageSearch/bin/activate
```
* * Windows:
```
path\to\imageSearch\Scripts\activate
```
* *  Anaconda:
```
conda activate imageSearch
```
* Browse to your project folder:
```
cd path/to/my_project_folder
```
* Launch Python
```
python
```
or 
```
python3
```

You are now now ready to use the application by importing the module `database`:
```
>>> import database
>>> import utils
```
If you would like to use the application with your favorite IDE or Jupyter, launch them from from the activated
virtual environment, for exmaple
```
jupyter notebook
```

When you create a notebook with Jupyter, choose the imageSearch kernel you have set during the installation. 

Please refer to the User Guide (Jupyter Notebook) for an overview of the main functionalities of the application. 

## Author
Isaac Debache
