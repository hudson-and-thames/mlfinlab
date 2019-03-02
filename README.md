# Machine Learning Financial Laboratory (mlfinlab)
Package based on the work of Dr Marcos Lopez de Prado, regarding his research with respect to Advances in Financial Machine Learning.

## Notes
* We just finished implementing the standard bar types (tick, volume, dollar). 
* Works on BIG csv files 25Gigs and up.
* Purchased high quality raw tick data.
* Email us if you would like a sample of the standard bars.
* Next we are busy adding the code for the labeling. See [research repo](https://github.com/hudson-and-thames/research/tree/ch3/chapter3) for the Q&A work in progress for chapter 3.

---

## Getting Started

### Installation on Mac OS X and Ubuntu Linux
Make sure you install the latest version of the Anaconda 3 distribution. To do this you can follow the install and update instructions found on this link: https://www.anaconda.com/download/#mac

From Terminal: go to the directory where you have saved the file, example: cd Desktop/mlfinlab/.

Run the command: ```pip install -r pip_requirements.txt```

### How To Run Checks Locally
On your local machine open the terminal and cd into the working dir. 
1. Code style checks: ```./pylint```
2. Unit tests: ```python -m unittest discover```
3. Code coverage: ```bash coverage```

### Installation on Windows
We still have to write this section but Ashutosh uses Windows and it runs on his machine. 

## Built With
* [Github](https://github.com/hudson-and-thames/mlfinlab) - Development platform and repo
* [Travis-CI](https://www.travis-ci.com) - Continuous integration, test and deploy

## Authors

* **Ashutosh Singh** - [LinkedIn](https://www.linkedin.com/in/ashusinghpenn/)
* **Jacques Joubert** - [LinkedIn](https://www.linkedin.com/in/jacquesjoubert/)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
