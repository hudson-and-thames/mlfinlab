.. _getting_started-installation:

============
Installation
============

Recommended Versions
####################

* Anaconda
* Python 3.8.

Installation
############

Mac OS X and Ubuntu Linux
*************************

1. Make sure you install the latest version of the Anaconda 3 distribution. To do this you can follow the install and update instructions found on this `link <https://www.anaconda.com/download/#mac>`_
2. Launch a terminal
3. Create a New Conda Environment. From terminal.

   .. code-block::

      conda create -n <env name> python=3.8 anaconda

   Accept all the requests to install.

4. Now activate the environment with:

   .. code-block::

      source activate <env name>

5. From Terminal: install the MlFinLab package:

   .. code-block::

      pip install mlfinlab

6. (Optional) **Only if you want to use the CorrGAN from the Data Generation Module**, install the TensorFlow package.
   Note that you should have pip version "pip==20.1.1" to do this. Supported TensorFlow version is "tensorflow==2.2.1".

   To change the pip version:

   .. code-block::

      pip install --user "pip==20.1.1"

   To install TensorFlow:

   .. code-block::

      pip install "tensorflow==2.2.1"

   .. warning::

      You may be encountering the following error during the installation:

      ``ERROR: tensorflow 2.2.1 has requirement numpy<1.19.0,>=1.16.0,``
      ``but you'll have numpy 1.20.1 which is incompatible.``

      You can ignore this message. It appears due to the updated dependency versions in the MlFinLab package.

      All the MlFinLab functionality still works as expected.


Windows
*******

.. warning::

    Before installing MlFinLab on Windows machines you should download and install
    `Visual Studio build tools for Python3 <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_.
    You can use this `installation guide <https://drive.google.com/file/d/0B4GsMXCRaSSIOWpYQkstajlYZ0tPVkNQSElmTWh1dXFaYkJr/view?usp=sharing>`_.

**Option A (with user interface)**

1. Download and install the latest version of `Anaconda 3 <https://www.anaconda.com/distribution/#download-section>`__
2. Launch Anaconda Navigator
3. Click Environments, choose an environment name, select Python 3.8, and click Create
4. Click Home, browse to your new environment, and click Install under Jupyter Notebook
5. Launch Anaconda Prompt and activate the environment:

   .. code-block::

      conda activate <env name>

6. From Anaconda Prompt: install the MlFinLab package:

   .. code-block::

      pip install mlfinlab

**Option B (with command prompt)**

1. Download and install the latest version of `Anaconda 3 <https://www.anaconda.com/distribution/#download-section>`__
2. Launch Anacoda Prompt
3. Create new environment (replace <env name> with a name, for example ``mlfinlab``):

   .. code-block::

      conda create -n <env name> python=3.8

4. Activate the new environment:

   .. code-block::

      conda activate <env name>

5. Install scs library (try one of the below options):

   .. code-block::

      pip install scs

   .. code-block::

      conda install -c conda-forge scs

   .. code-block::

      conda install -c anaconda ecos

6. Install MlFinLab:

   .. code-block::

      pip install mlfinlab

   .. Note::

       If you have problems with installation related to Numba and llvmlight, `this solution <https://github.com/hudson-and-thames/mlfinlab/issues/448>`_ might help.

7. (Optional) **Only if you want to use the CorrGAN from the Data Generation Module**, install the TensorFlow package.
   Note that you should have pip version "pip==20.1.1" to do this. Supported TensorFlow version is "tensorflow==2.2.1".

   To change the pip version:

   .. code-block::

      pip install --user "pip==20.1.1"

   To install TensorFlow:

   .. code-block::

      pip install "tensorflow==2.2.1"

   .. warning::

      You may be encountering the following error during the installation:

      ``ERROR: tensorflow 2.2.1 has requirement numpy<1.19.0,>=1.16.0,``
      ``but you'll have numpy 1.20.1 which is incompatible.``

      You can ignore this message. It appears due to the updated dependency versions in the MlFinLab package.

      All the MlFinLab functionality still works as expected.
