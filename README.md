# Machine Learning Financial Laboratory (mlfinlab)
MLFinLab is an open source package based on the research of Dr Marcos Lopez de Prado in his new book
Advances in Financial Machine Learning. This implementation started out as a platform for which Ashutosh and
Jacques could base their research project on for their [Masters in Financial Engineering at WorldQuant University](https://wqu.org/).

As we were working through the book we saw the opportunity to code up the implementations as well answer the 
questions at the back of every chapter. 

## Barriers to Entry
As most of you know, getting through the first 3 chapters of the book is challenging as it relies on HFT data to 
create the new financial data structures. Sourcing the HFT data is very difficult and thus we have resorted to purchasing the
full history of S&P500 Emini futures tick data from [TickData LLC](https://www.tickdata.com/).

We are not affiliated with TickData in any way but would like to recommend others to make use of their service. The full history 
cost us about $750 and is worth every penny. They have really done a great job at cleaning the data and providing it in 
a user friendly manner. 

### Sample Data
TickData does offer about 20 days worth of raw tick data which can be sourced from their website [link](https://s3-us-west-2.amazonaws.com/tick-data-s3/downloads/ES_Sample.zip).

For those of you interested in working with a two years of sample tick, volume, and dollar bars, it is provided for in the [research repo.](https://github.com/hudson-and-thames/research/tree/master/Sample-Data).

You should be able to work on a few implementations of the code with this set. 

---

## Notes
* We just finished implementing the standard bar types (tick, volume, dollar). 
* Works on BIG csv files 25Gigs and up.
* Purchased high quality raw tick data.
* Email us if you would like a sample of the standard bars.
* Next we are busy adding the code for the labeling. See [research repo](https://github.com/hudson-and-thames/research) for the Q&A work in progress for chapter 3.

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

## Contact us
At the moment the project is still rather small and thus I would recommend getting in touch with us over email so that we can further discuss the areas of contribution that interest you the most. As soon as we get to more than 4 maintainers we will switch over to a slack channel.

For now you can get hold us at: hudsonthames19@gmail.com

Looking forward to hearing from you!

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
