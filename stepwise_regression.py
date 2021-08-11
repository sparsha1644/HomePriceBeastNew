import pandas as pd
import statsmodels.formula.api as smf
import os
#import numpy as np
#import itertools
#import statsmodels.api as sm

#############################################################################
# Preliminaries
#############################################################################

os.chdir("C:\\Users\\blevi\\Documents\\python_files")
data_folder = "C:\\Users\\blevi\\Documents\\python_files"

# Import
raw_home_data = pd.read_csv("projectcombined_home_data_v2.csv")

# Training Data
data_train = raw_home_data[raw_home_data['period_begin_num'] < 43800]  # split at 12/1/2019


#############################################################################
# Define Variables
#############################################################################

data_train = data_train.rename(columns={'region_id': 'id', 'median_sale_price_yoy': 'y',
                                      'total_homes_sold': 'x1',
                                      'median_days_to_close_yoy': 'x2',
                                      'total_new_listings_yoy': 'x3',
                                      'inventory_yoy': 'x4',
                                      'active_listings_yoy': 'x5',
                                      'age_of_inventory_yoy': 'x6',
                                      'average_sale_to_list_ratio_yoy': 'x7',
                                      'median_days_on_market_yoy': 'x8',
                                      'months_of_supply_yoy': 'x9',
                                      'average_pending_sales_listing_updates_yoy': 'x10',
                                      'percent_total_price_drops_of_inventory_yoy': 'x11',
                                      'percent_homes_sold_above_list_yoy': 'x12',
                                      'unrate_new': 'x13',
                                      'fixed_mortgage_30yr': 'x14'})


X = data_train[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
                            "x10", "x11", "x12", "x13", "x14"]]

y = data_train[["y"]]

data_reg = pd.concat([X, y, data_train[["id"]]], axis=1)  

# Get Unique IDs
id_list = data_train[["id"]]
id_unique = id_list.drop_duplicates(subset='id', keep="last")


#############################################################################
# Define Function
#############################################################################

# Author: Trevor Smith @ https://planspace.org/20150423-forward_selection_with_statsmodels/

def forward_selected(data, response):
    
    """Linear model designed by forward selection.
   
    ---------------------------------------------------
    Parameters:
   
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data
    
    ---------------------------------------------------
    Returns:
   
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    
    remaining = set(data.columns)
    
    remaining.remove(response)
    
    selected = []
   
    current_score, best_new_score = 0.0, 0.0
   
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
   
    model = smf.o


#############################################################################
# Forward Selection
#############################################################################

for x in id_unique:
    
    temp = data_reg[data_reg['id'] == id_unique.iloc[x]]  # Code is not recognizing 
                                                          # the id_unique.iloc[x] portion
    
    x_temp = temp[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
                            "x10", "x11", "x12", "x13", "x14"]]
    
    y_temp = temp["y"]
    
    store = forward_selected(x_temp, y_temp)  # run function based on current ID
    
    #########################################################################
    
    # Ideally, insert code to store coefficients of each iteration here
    
    #########################################################################


