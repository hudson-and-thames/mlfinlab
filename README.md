<div align="center">
  <img src="https://raw.githubusercontent.com/hudson-and-thames/mlfinlab/master/.github/logo/hudson_and_thames_logo.png" height="300"><br>
</div>

-----------------
# Machine Learning Financial Laboratory (mlfinlab)
[![Build Status](https://travis-ci.com/hudson-and-thames/mlfinlab.svg?branch=master)](https://travis-ci.com/hudson-and-thames/mlfinlab)
[![codecov](https://codecov.io/gh/hudson-and-thames/mlfinlab/branch/master/graph/badge.svg)](https://codecov.io/gh/hudson-and-thames/mlfinlab)
![pylint Score](https://mperlet.github.io/pybadge/badges/10.svg)
[![License: BSD3](https://img.shields.io/github/license/hudson-and-thames/mlfinlab.svg)](https://opensource.org/licenses/BSD-3-Clause)

[![Documentation Status](https://readthedocs.org/projects/mlfinlab/badge/?version=latest)](https://mlfinlab.readthedocs.io/en/latest/?badge=latest)
[![PyPi](https://img.shields.io/pypi/v/mlfinlab.svg)]((https://pypi.org/project/mlfinlab/))
[![Downloads](https://img.shields.io/pypi/dm/mlfinlab.svg)]((https://pypi.org/project/mlfinlab/))
[![Python](https://img.shields.io/pypi/pyversions/mlfinlab.svg)]((https://pypi.org/project/mlfinlab/))

MLFinLab is an open source package based on the research of Dr Marcos Lopez de Prado in his new book
Advances in Financial Machine Learning. This implementation started out as a spring board for a research project in the [Masters in Financial Engineering programme at WorldQuant University](https://wqu.org/) and has grown into a mini research group called [Hudson and Thames Quantitative Research](https://hudsonthames.org/) (not affiliated with the university).

The following is the online documentation for the package: [read-the-docs](https://mlfinlab.readthedocs.io/en/latest/#).

## Barriers to Entry
As most of you know, getting through the first 3 chapters of the book is challenging as it relies on HFT data to 
create the new financial data structures. Sourcing the HFT data is very difficult and thus we have resorted to purchasing the full history of S&P500 Emini futures tick data from [TickData LLC](https://www.tickdata.com/).

We are not affiliated with TickData in any way but would like to recommend others to make use of their service. The full history cost us about $750 and is worth every penny. They have really done a great job at cleaning the data and providing it in a user friendly manner. 

### Sample Data
TickData does offer about 20 days worth of raw tick data which can be sourced from their website [link](https://s3-us-west-2.amazonaws.com/tick-data-s3/downloads/ES_Sample.zip).

For those of you interested in working with a two years of sample tick, volume, and dollar bars, it is provided for in the [research repo.](https://github.com/hudson-and-thames/research/tree/master/Sample-Data).

You should be able to work on a few implementations of the code with this set. 


<div align="center">
  <img src="https://raw.githubusercontent.com/hudson-and-thames/research/master/Chapter3/readme_image.png" height="350"><br>
</div>

---

## Progress

**Part 4: Useful Financial Features**
* Working on Chapter 19: Microstructural Features (Maksim)

**Part 3: Backtesting**
* Done Chapter 16: Asset Allocation
* Done Chapter 10: Bet Sizing

**Part 2: Modelling**
* Done Chapter 8: Feature Importance
* Done Chapter 7: Cross-Validation
* Done Chapter 6: Ensemble Methods (Sequential Bootstrap Ensemble)

**Part 1: Data Analysis**
* Done Chapter 5: Fractional Differentiation
* Done Chapter 4: Sample Weights
* Done Chapter 3: Labeling
* Done Chapter 2: Data Structures
* Purchased high quality raw tick data.
* Email us if you would like a sample of the standard bars.

---

## Getting Started

Recommended versions:
* Anaconda 3
* Python 3.6

### Installation for users
The package can be installed from the PyPi index via the console:
 1. Launch the terminal and run: ```pip install mlfinlab```

### Installation for developers
Clone the [package repo](https://github.com/hudson-and-thames/mlfinlab) to your local machine then follow the steps below.

#### Installation on Mac OS X and Ubuntu Linux
1. Make sure you install the latest version of the Anaconda 3 distribution. To do this you can follow the install and update instructions found on this link: https://www.anaconda.com/download/#mac
2. Launch a terminal
3. Create a New Conda Environment. From terminal: ```conda create -n <env name> python=3.6 anaconda``` accept all the requests to install.
4. Now activate the environment with ```source activate <env name>```.
5. From Terminal: go to the directory where you have saved the file, example: cd Desktop/mlfinlab/.
6. Install Python requirements, by running the command: ```pip install -r requirements.txt```

#### Installation on Windows
1. Download and install the latest version of [Anaconda 3](https://www.anaconda.com/distribution/#download-section)
2. Launch Anaconda Navigator
3. Click Environments, choose an environment name, select Python 3.6, and click Create
4. Click Home, browse to your new environment, and click Install under Jupyter Notebook
5. Launch Anaconda Prompt and activate the environment: ```conda activate <env name>```
6. From Anaconda Prompt: go to the directory where you have saved the file, example: cd Desktop/mlfinlab/.
7. Install Python requirements, by running the command: ```pip install -r requirements.txt```

### How To Run Checks Locally
On your local machine open the terminal and cd into the working dir. 
1. Code style checks: ```./pylint```
2. Unit tests: ```python -m unittest discover```
3. Code coverage: ```bash coverage```

## Built With
* [Github](https://github.com/hudson-and-thames/mlfinlab) - Development platform and repo
* [Travis-CI](https://www.travis-ci.com) - Continuous integration, test and deploy

## Authors

* **Ashutosh Singh** - [LinkedIn](https://www.linkedin.com/in/ashusinghpenn/)
* **Jacques Joubert** - [LinkedIn](https://www.linkedin.com/in/jacquesjoubert/)
* **Oleksandr Proskurin** - [LinkedIn](https://www.linkedin.com/in/proskurinolexandr/)


## Additional Research Repo
BlackArbsCEO has a great repo based on de Prado's research. It covers many of the questions at the back of every chapter and was the first source on Github to do so. It has also been a good source of inspiration for our research. 

* [Adv Fin ML Exercises](https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises)

## Contact us
At the moment the project is still rather small and thus I would recommend getting in touch with us over email so that we can further discuss the areas of contribution that interest you the most. We have a slack channel where we all communicate.

For now you can get hold us at: research@hudsonthames.org

Looking forward to hearing from you!

## License

This project is licensed under the 3-Clause BSD License - see the [LICENSE.txt](https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt) file for details.
