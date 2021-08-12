# Home Price Beast
### Important Data Announcement
 - Please download the home-inventory and pricing data from [Redfin Data Center](https://redfin-public-data.s3-us-west-2.amazonaws.com/redfin_covid19/weekly_housing_market_data_most_recent.tsv)
 - Replace ```data_folder``` with the right one on your machine.
 
 ### Running the ARIMA models
The time-series ARIMA models are fit and analyzed in the ``priceArimaModel`` notebook. For a small tutorial on fitting models with pmdarima, see instead ``arimaModelsTutorial``.

### Running Stepwise Regression
This loops through our housing market dataset and -- for each county -- fits a model based on backward selection.  The resulting coefficients are then used as the basis for the counterfactual analysis we presented in this section of our slide deck.  The script is labeled stepwise_regresssion.py.

Note to Dr. Z (from Brian Levine): As someone with zero Python experience -- and as someone who is far from a data scientist -- I struggled implementing this portion in Python.  In the spirit of transparency, I used R, which I am much more comfortable with.  Accordingly, when trying to rewrite the R code in Python (for submission purposes), I admittedly did not *quite* capture the entire scope or breadth of what I presented in class -- instead, I tried to provide the general framework for how we approached this section.  Most importantly: Michael helped immensely here, as he wrote two critical functions to assist in my attempted rewrite.  These functions are embedded in the backward_selection.py script.  The first runs the algorithm, and the second harnesses the resulting coefficients to make a prediction (for our counterfactual).  That said, I want to be clear that any issues with the stepwise_regression.py code are my own, and I am happy to chat separately if you wish to inquire about my R code.
