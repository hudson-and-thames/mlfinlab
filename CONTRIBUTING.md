# Contributing to mlfinlab
First off we wanted to thank you for taking the time to contribute to the project. 

mlfinlab is an open source package based on the research of Dr Marcos Lopez de Prado in his new book
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

For those of you interested in working with a one year sample of tick, volume, and dollar bars, it is provided for in this repo.

You should be able to work on a few implementations of the code with this set. 

## Areas of Contribution
Currently we have a [live project board](https://github.com/orgs/hudson-and-thames/projects) that follows the principles of Agile Project Management. This board is available to the public
and lets everyone know what the maintainers are currently working on. The board has an ice bucket filled with new features and documentation 
that have priority. 

Needless to say, if you are interested in working on something not on the project board, just drop us an email. We would be very excited to 
discuss this further.

The typical contributions are:
* Answering the questions at the back of a chapter in a Jupyter Notebook. [Research repo](https://github.com/hudson-and-thames/research)
* Adding new features and core functionality to the mlfinlab package. 

## Templates
We have created [templates](https://github.com/hudson-and-thames/mlfinlab/issues/new/choose) to help aid in creating issues and PRs:
* Bug report
* Feature request
* Custom issue template
* Pull Request Template

---

## Contact us
At the moment the project is still rather small and thus I would recommend getting in touch with us over email so that we can further 
discuss the areas of contribution that interest you the most. As soon as we get to more than 4 maintainers we will switch over to a 
slack channel.

For now you can get hold us at: hudsonthames19@gmail.com

Looking forward to hearing from you!

