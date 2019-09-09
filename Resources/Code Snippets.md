A set of regularly used code snippets

## Packages: Installing, Importing, Searching, Updating

### Linux

**OS Updgrading**
```shell
sudo apt-get update && sudo apt-get upgrade
sudo apt-get update && sudo apt-get dist-upgrade
```

### Jupyter

**Configuration**

```python
%matplotlib inline
%pylab inline
```

### IPython

```shell
ipython kernelspec install-self # Install Python 3 into the kernelspec
```

### Python

**Updating via conda**
```shell
conda update python
```

**Importing Packages**

```python
import numpy as np
from numpy.random import random
from numpy.random import permutation
from numpy import cov
from numpy import median

import pandas as pd
from pandas import DataFrame as df

import sklearn
from sklearn import datasets
from sklearn.datasets import fetch_mldata

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import KFold

from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns

from bokeh.plotting import figure, output_file, show

from IPython.display import Image

from mpl_toolkits.basemap import Basemap

import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools.tools import rank, add_constant
from statsmodels.datasets import macrodata
import statsmodels.stats import diagnostic
import statsmodels.regression.linear_model as lm
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi

import scipy
from scipy import stats
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
from scipy.spatial import distance
from scipy import linspace
from scipy.stats import binom
from scipy.stats import linregress
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import norm
from scipy.stats.stats import pearsonr
from scipy.stats import trim_mean
from scipy.stats.mstats import mode, gmean, hmean

import nltk

from pyspark.sql import SQLContext

from bs4 import BeautifulSoup

# Python Core (Standard Library)
import math
import random
import statistics
import io
from io import StringIO
import requests
import csv
import json
import xml.etree.ElementTree
import xml.dom
import xml.sax
import os
import sys
from datetime import date
import shutil # Daily file and directory management tasks, higher level than os
import glob # Function for making file lists from directory wildcard searches
import re # Regular expressions
from urllib.request # HTTP URL requests/responses
from urllib.request import urlopen
import smtplib # # SMTP for emails
import poplib # # POP for emails
import zlib # Data compression
from timeit import Timer # Performance measurement
import doctest # Quality control
import unittest
import xmlrpc.client # RPC
import xmlrpc.server
import email # Managing email messages
import sqlite3
import gettext
import locale
import codecs

import re
import sqlite3
import psycopg2
import operator
from collections import Counter
```

### PIP

```shell
pip install -U pip # Update PIP
pip install <package_name>
```

### Conda/Anaconda

**Package Management**

```shell
conda list

conda search -f python # List available Python packages
conda search <package_name>

conda update conda 
conda update anaconda
conda update --all # E.g., conda update --all python=3.5 # Updates all packages in default environment to Python 3 versions
conda update -n <env_name> <package_name(s)>
conda update <package_name> # E.g., juptyer

conda install -n <env_name> <package_name> # NOTE: -n <env_name> is the environment to install into, if omitted is default
```

**REPL**

```
from __future__ import division # Converts division output to decimal format
```

### R

```shell
open https://cran.r-project.org/bin/macosx/
```

### Homebrew
```shell
brew list
brew update; brew upgrade
```

### Ruby/RBENV/RVM

```shell
rbenv install –l # Shows Ruby versions available to rbenv
rvm get stable
```

### Jekyll/Ruby Gems

```shell
gem list
gem list <package_name> # E.g., jekyll
gem update <package_name>
gem update --system # If issue (https://rubygems.org/pages/download): gem install rubygems-update; update_rubygems
```

### Node/NPM

```shell
open https://nodejs.org/en/download/
npm install npm@latest -g
npm update -g # Update all global packages
npm-check-updates -u

npm install <module> --save-dev
npm uninstall <package>
npm uninstall -g <package>
```

<!-- **Updating Packages**

**Listing Packages** -->

-----

## CLI and API

### Jupyter

**CLI**

```shell
jupyter notebook # Launch Jupyter
jupyter notebook --debug

jupyter notebook --generate-config # Creates commented config file at ~/.jupyter/jupyter_notebook_config.py
jupyter notebook --help # List config options
```

### IPython

**CLI**

```shell
jupyter kernelspec list
```

### Conda/Anaconda

**Environment Management**

```shell
conda info -e # List environments
ls -al ~/anaconda/envs/ # List environments in directory

# Create environment and install packages**
conda create -n <env_name> python=<python_version_number> <package_names> # E.g., name = py352, python version = 3.5.2
conda create -n <env_name> python=<python_version_number> matplotlib numpy scipy scikit-learn jupyter pandas statsmodels nltk seaborn # Sample starter environment

# Delete environment
conda remove -n <env_name> --all

# Activate and deactivate environment
source activate <env_name>
source deactivate <env_name>

# Export environment and load
conda env export > <path/filename.yml>
conda env create -f <path/filename.yml>
```

-----

## Versions and Locations

**Python**
```shell
which python
which python3
python --version
```

**Jupyter and IPython**
```shell
which jupyter
which ipython
```

**Conda/Anaconda**
```shell
conda info
which anaconda
anaconda --version
```

**PIP**
```shell
which pip
pip --version
```

**R**
```shell
R --version
```

**Node.js and NPM**
```shell
node -v
npm -v
```

**Ruby & RBENV**
```shell
which ruby
ruby -v
rbenv --version
```

**Jekyll & Ruby Gems**
```shell
jekyll --version
```

-----

## Important File Locations (Linux/Mac)
- Jupyter
- IPython
- Conda
    + ~/<conda_dist_directory>/envs
- Python
- General installation directory (mac): /usr/local/bin
- General framewords directory (mac): /Library/Frameworks



