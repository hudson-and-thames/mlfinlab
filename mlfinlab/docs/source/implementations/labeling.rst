.. _implementations-labeling:

========
Labeling
========

The primary labeling method used in financial academia is the fixed-time horizon method. While ubiquitous, this method has many faults which are remedied by the triple-barrier method discussed below. The triple-barrier method can be extended to incorporate meta-labeling which will also be demonstrated and discussed below.

Triple-Barrier Method
=====================

The idea behind the triple-barrier method is that we have three barriers: an upper barrier, a lower barrier, and a vertical barrier. The upper barrier represents the threshold an observation's return needs to reach in order to be considered a buying opportunty (a label of 1), the lower barrier represents the threshold an observation's return needs to reach in order to be considered a selling opportunity (a label of -1), and the vertical barrier represents the amount of time an observation has to reach its given return in either direction before it is given a label of 0. This concept can be better understood visually and is shown in the figure below taken from Advances in Financial Machine Learning (`reference`_):

.. image:: labeling_images/triple_barrier.png
   :scale: 100 %
   :align: center

One of the major faults with the fixed-time horizon method is that observations are given a label with respect to a certain threshold after a fixed interval regardless of their respective volatilities. In other words, the expected returns of every observation are treated equally regardless of the associated risk. The triple-barrier method tackles this issue by dynamically setting the upper and lower barriers for each observation based on their given volatilities.

.. _reference: https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086

Meta-Labeling
=============

Advances in Financial Machine Learning, Chapter 3, page 50. Reads:

"Suppose that you have a model for setting the side of the bet (long or short). You just need to learn the size of that bet, which includes the possibility of no bet at all (zero size). This is a situation that practitioners face regularly. We often know whether we want to buy or sell a product, and the only remaining question is how much money we should risk in such a bet. We do not want the ML algorithm to learn the side, just to tell us what is the appropriate size. At this point, it probably does not surprise you to hear that no book or paper has so far discussed this common problem. Thankfully, that misery ends here.""

I call this problem meta-labeling because we want to build a secondary ML model that learns how to use a primary exogenous model.

The ML algorithm will be trained to decide whether to take the bet or pass, a purely binary prediction. When the predicted label is 1, we can use the probability of this secondary prediction to derive the size of the bet, where the side (sign) of the position has been set by the primary model.

How to use Meta-Labeling
~~~~~~~~~~~~~~~~~~~~~~~~

Binary classification problems present a trade-off between type-I errors (false positives) and type-II errors (false negatives). In general, increasing the true positive rate of a binary classifier will tend to increase its false positive rate. The receiver operating characteristic (ROC) curve of a binary classifier measures the cost of increasing the true positive rate, in terms of accepting higher false positive rates.

.. image:: labeling_images/confusion_matrix.png
   :scale: 40 %
   :align: center


The image illustrates the so-called “confusion matrix.” On a set of observations, there are items that exhibit a condition (positives, left rectangle), and items that do not exhibit a condition (negative, right rectangle). A binary classifier predicts that some items exhibit the condition (ellipse), where the TP area contains the true positives and the TN area contains the true negatives. This leads to two kinds of errors: false positives (FP) and false negatives (FN). “Precision” is the ratio between the TP area and the area in the ellipse. “Recall” is the ratio between the TP area and the area in the left rectangle. This notion of recall (aka true positive rate) is in the context of classification problems, the analogous to “power” in the context of hypothesis testing. “Accuracy” is the sum of the TP and TN areas divided by the overall set of items (square). In general, decreasing the FP area comes at a cost of increasing the FN area, because higher precision typically means fewer calls, hence lower recall. Still, there is some combination of precision and recall that maximizes the overall efficiency of the classifier. The F1-score measures the efficiency of a classifier as the harmonic average between precision and recall.

**Meta-labeling is particularly helpful when you want to achieve higher F1-scores**. First, we build a model that achieves high recall, even if the precision is not particularly high. Second, we correct for the low precision by applying meta-labeling to the positives predicted by the primary model.

Meta-labeling will increase your F1-score by filtering out the false positives, where the majority of positives have already been identified by the primary model. Stated differently, the role of the secondary ML algorithm is to determine whether a positive from the primary (exogenous) model is true or false. It is not its purpose to come up with a betting opportunity. Its purpose is to determine whether we should act or pass on the opportunity that has been presented.

Meta-labeling is a very powerful tool to have in your arsenal, for four additional reasons. **First**, ML algorithms are often criticized as black boxes. Meta-labeling allows you to build an ML system on top of a white box (like a fundamental model founded on economic theory). This ability to transform a fundamental model into an ML model should make meta-labeling particularly useful to “quantamental” firms. **Second**, the effects of overfitting are limited when you apply metalabeling, because ML will not decide the side of your bet, only the size. **Third**, by decoupling the side prediction from the size prediction, meta-labeling enables sophisticated strategy structures. For instance, consider that the features driving a rally may differ from the features driving a sell-off. In that case, you may want to develop an ML strategy exclusively for long positions, based on the buy recommendations of a primary model, and an ML strategy exclusively for short positions, based on the sell recommendations of an entirely different primary model. **Fourth**, achieving high accuracy on small bets and low accuracy on large bets will ruin you. As important as identifying good opportunities is to size them properly, so it makes sense to develop an ML algorithm solely focused on getting that critical decision (sizing) right. We will retake this fourth point in Chapter 10. In my experience, meta-labeling ML models can deliver more robust and reliable outcomes than standard labeling models.

Model Architecture
~~~~~~~~~~~~~~~~~~

The following image explains the model architecture. The **first** step is to train a primary model (binary classification). **Second** a threshold level is determined at which the primary model has a high recall, in the coded example you will find that 0.30 is a good threshold, ROC curves could be used to help determine a good level. **Third** the features from the first model are concatenated with the predictions from the first model, into a new feature set for the secondary model. Meta Labels are used as the target variable in the second model. Now fit the second model. **Fourth** the prediction from the secondary model is combined with the prediction from the primary model and only where both are true, is your final prediction true. I.e. if your primary model predicts a 3 and your secondary model says you have a high probability of the primary model being correct, is your final prediction a 3, else not 3.

.. image:: labeling_images/meta_labeling_architecture.png
   :scale: 70 %
   :align: center


Implementation
==============

The following functions are used for the triple-barrier method which works in tandem with meta-labeling.


.. function:: get_daily_vol(close, lookback=100)

    Snippet 3.1 computes the daily volatility
    at intraday estimation points, applying a span of lookback days to an exponentially weighted moving
    standard deviation.

    See the pandas documentation for details on the pandas.Series.ewm function.

    Note: This function is used to compute dynamic thresholds for profit taking and stop loss limits.

    :param close: Closing prices
    :param lookback: lookback period to compute volatility
    :return: series of daily volatility value


.. function:: add_vertical_barrier(t_events, close, num_days=0, num_hours=0, num_minutes=0, num_seconds=0)

    Snippet 3.4 page 49, Adding a Vertical Barrier

    For each index in t_events, it finds the timestamp of the next price bar at or immediately after
    a number of days num_days. This vertical barrier can be passed as an optional argument t1 in get_events.

    This function creates a series that has all the timestamps of when the vertical barrier would be reached.

    :param t_events: (series) series of events (symmetric CUSUM filter)
    :param close: (series) close prices
    :param num_days: (int) number of days to add for vertical barrier
    :param num_hours: (int) number of hours to add for vertical barrier
    :param num_minutes: (int) number of minutes to add for vertical barrier
    :param num_seconds: (int) number of seconds to add for vertical barrier
    :return: (series) timestamps of vertical barriers


.. function:: get_events(close, t_events, pt_sl, target, min_ret, num_threads, vertical_barrier_times=False, side_prediction=None)

    Snippet 3.6 page 50, Getting the Time of the First Touch, with Meta Labels

    This function is orchestrator to meta-label the data, in conjunction with the Triple Barrier Method.

    :param close: (series) Close prices

    :param t_events: (series) of t_events. These are timestamps that will seed every triple barrier. 
    	These are the timestamps selected by the sampling procedures discussed in Chapter 2, Section 2.5. Eg: CUSUM Filter

    :param pt_sl: (2 element array) element 0, indicates the profit taking level; element 1 is stop loss level. A non-negative float that sets the width of the two barriers. 
    	A 0 value means that the respective horizontal barrier (profit taking and/or stop loss) will be disabled.

    :param target: (series) of values that are used (in conjunction with pt_sl) to determine the width of the barrier. In this program this is daily volatility series.

    :param min_ret: (float) The minimum target return required for running a triple barrier search.

    :param num_threads: (int) The number of threads concurrently used by the function.

    :param vertical_barrier_times: (series) A pandas series with the timestamps of the vertical barriers. We pass a False when we want to disable vertical barriers.

    :param side_prediction: (series) Side of the bet (long/short) as decided by the primary model


    :return: (data frame) of events
            events.index is event's starttime

            events['t1'] is event's endtime

            events['trgt'] is event's target

            events['side'] (optional) implies the algo's position side


.. function:: get_bins(triple_barrier_events, close)

    Snippet 3.7, page 51, Labeling for Side & Size with Meta Labels

    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:

    Now the possible values for labels in out['bin'] are {0,1}, as opposed to whether to take the bet or pass,
    a purely binary prediction. When the predicted label the previous feasible values {−1,0,1}.
    The ML algorithm will be trained to decide is 1, we can use the probability of this secondary prediction
    to derive the size of the bet, where the side (sign) of the position has been set by the primary model.

    :param triple_barrier_events: (data frame)

        events.index is event's starttime

	events['t1'] is event's endtime

	events['trgt'] is event's target

	events['side'] (optional) implies the algo's position side

	Case 1: ('side' not in events): bin in (-1,1) <-label by price action

	Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)

    :param close: (series) close prices
    :return: (data frame) of meta-labeled events

.. function:: drop_labels(events, min_pct=.05)

    Snippet 3.8 page 54
    This function recursively eliminates rare observations.

    :param events: (data frame) events
    :param min_pct: (float) a fraction used to decide if the observation occurs less than that fraction
    :return: (data frame) of event

Example
=======

Suppose we use a mean reverting strategy as our primary model, giving each observation a label of 1 or -1.
We can then use meta-labeling to act as a filter for the bets of our primary model.

::

	import mlfinlab as ml
	import numpy as np
	import pandas as pd

::

	# Read in data
	data = pd.read_csv('FILE_PATH')

Assuming we have a pandas series with the timestamps of our observations and their respective labels given by the primary model, the process to generate meta-labels goes as follows.

::

	# Compute daily volatility
	daily_vol = ml.util.get_daily_vol(close=data['close'], lookback=50)

	# Apply Symmetric CUSUM Filter and get timestamps for events
	# Note: Only the CUSUM filter needs a point estimate for volatility
	cusum_events = ml.filters.cusum_filter(data['close'],
        threshold=daily_vol['2011-09-01':'2018-01-01'].mean()*0.5)

	# Compute vertical barrier
	vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events,
        close=data['close'], num_days=1)

Once we have computed our daily volatility along with our vertical time barriers and have downsampled our series using the CUSUM filter, we can use the triple-barrier method to compute our meta-labels by passing in the side predicted by the primary model.

::

	pt_sl = [1, 2]
	min_ret = 0.005
	triple_barrier_events = ml.labeling.get_events(close=data['close'],
                                               t_events=cusum_events,
                                               pt_sl=pt_sl,
                                               target=daily_vol,
                                               min_ret=min_ret,
                                               num_threads=3,
                                               vertical_barrier_times=vertical_barriers,
                                               side_prediction=data['side'])

As can be seen above, we have scaled our lower barrier and set our minimum return to 0.005.

Meta-labels can then be computed using the time that each observation touched its respective barrier

::

	meta_labels = ml.labeling.get_bins(triple_barrier_events, data['close'])

This example ends with creating the meta-labels. To see a further explanation of using these labels in a secondary model to help filter out false positives, see the research notebooks below.

Research Notebooks
==================

The following research notebooks can be used to better understand the triple-barrier method and meta-labeling

Triple-Barrier Method
~~~~~~~~~~~~~~~~~~~~~

* `Trend Follow Question`_
* `Bollinger band Question`_

.. _Trend Follow Question: https://github.com/hudson-and-thames/research/blob/master/Chapter3/2019-03-06_JJ_Trend-Follow-Question.ipynb
.. _Bollinger Band Question: https://github.com/hudson-and-thames/research/blob/master/Chapter3/2019-03-09_AS_BBand-Question.ipynb

Meta-Labeling
~~~~~~~~~~~~~~

* `Meta Labeling MNIST`_

.. _Meta Labeling MNIST: https://github.com/hudson-and-thames/research/blob/master/Chapter3/2019-03-06_JJ_Meta-Labels-MNIST.ipynb









