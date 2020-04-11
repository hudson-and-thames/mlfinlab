
============
Installation
============

Recommended Versions
####################

* Anaconda 3
* Python 3.6

Installation for Users
######################

The package can be installed from the PyPi index via the console:
Launch the terminal and run.

.. code-block::

   pip install mlfinlab

-----------------------------

Installation for Developers
###########################

Clone the `package repo`_ to your local machine then follow the steps below.

Mac OS X and Ubuntu Linux
*************************

1. Make sure you install the latest version of the Anaconda 3 distribution. To do this you can follow the install and update instructions found on this `link`_
2. Launch a terminal
3. Create a New Conda Environment. From terminal.

.. code-block::

   conda create -n <env name> python=3.6 anaconda

Accept all the requests to install.

4. Now activate the environment with:

.. code-block::

   source activate <env name>

5. From Terminal: go to the directory where you have saved the file, example:

.. code-block::

   cd Desktop/mlfinlab

6. Install Python requirements, by running the command:

.. code-block::

   pip install -r requirements.txt

Windows
*******

1. Download and install the latest version of `Anaconda 3`_
2. Launch Anaconda Navigator
3. Click Environments, choose an environment name, select Python 3.6, and click Create
4. Click Home, browse to your new environment, and click Install under Jupyter Notebook
5. Launch Anaconda Prompt and activate the environment:

.. code-block::

   conda activate <env name>

6. From Anaconda Prompt: go to the directory where you have saved the file, example:

.. code-block::

   cd Desktop/mlfinlab

7. Install Python requirements, by running the command:

.. code-block::

   pip install -r requirements.txt

.. _package repo: (https://github.com/hudson-and-thames/mlfinlab)
.. _link: (https://www.anaconda.com/download/#mac)
.. _Anaconda 3: (https://www.anaconda.com/distribution/#download-section)
