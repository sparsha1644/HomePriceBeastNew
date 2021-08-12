# Home Price Beast
### Important Data Announcement
 - Please download the home-inventory and pricing data from [Redfin Data Center](https://redfin-public-data.s3-us-west-2.amazonaws.com/redfin_covid19/weekly_housing_market_data_most_recent.tsv)
 - Replace ```data_folder``` with the right one on your machine.
 
 ### Running the ARIMA models
The time-series ARIMA models are fit and analyzed in the ``priceArimaModel`` notebook. For a small tutorial on fitting models with pmdarima, see instead ``arimaModelsTutorial``.

### Running Stepwise Regression
This loops through our housing market dataset and -- for each county -- fits a model based on backward selection.  The resulting coefficients are then used as the basis for the counterfactual analysis we presented in this section of our slide deck.  The script is labeled ``stepwise_regresssion.py``.

Note to Dr. Z (from Brian Levine): As someone with zero Python experience -- and as someone who is far from a data scientist -- I struggled implementing this portion in Python.  In the spirit of transparency, I used R, which I am much more comfortable with.  Accordingly, when trying to rewrite the R code in Python (for submission purposes), I admittedly did not *quite* capture the entire scope or breadth of what I presented in class -- instead, I tried to provide the general framework for how we approached this section.  Most importantly: Michael helped immensely here, as he wrote two critical functions to assist in my attempted rewrite.  These functions are embedded in the ``backward_selection.py`` script.  The first runs the algorithm, and the second harnesses the resulting coefficients to make a prediction (for our counterfactual).  

That said, I want to be clear that any issues with the ``stepwise_regression.py`` code are my own, and I am happy to chat separately if you wish to inquire about my R code.

### Running LSTM Models. 

All LSTM models first start with data preparation to transform the weekly data into lag and lead variables and then transform into tensors. 
``DataTransformationTimeSeries.ipynb`` does this transformation agnostic of the models we will build. 

Currently we have two models, one to build predict future inventory using static variables and historical inventory. 

Any data transformation specific to this model is done in the file : ``Inventory_model_data_prep.ipynb``
Followed by actual training of this model which is done in (this includes prediction of test and post covid time frame results): ``LSTM_EncoderDecoder.ipynb``

Second we use the historical sale price, future inventory and static features to fit a model to predict future sale price. 
Any data transformation specific to this model is done in the file : ``mediansaleprice_model_data_prep.ipynb``
Followed by actual training of this model which is done in : ``LSTM_EncoderDecoder-Sale.ipynb``
Preparation of post covid prediction data : ``CounterfactualAnalysisSalePrice.ipynb``
Followed by predicting the results to test and post covid time period which is done in file : ``CounterfactualPlots.ipynb``

### Other files. 

Rest of the python notebooks where primarily used for the data analysis tasks. We also have checked in a few model versions and intermediary data files that were small.

