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
# Import Packages
#############################################################################

import pandas as pd

# Install Necessary Functions
#pip install mlxtend
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression


#############################################################################
# Backward Elimination
#############################################################################

for i in id_unique:
    
    temp = data_reg[data_reg['id'] == id_unique.iloc[i]]  # NOTE: syntax is not recognizing the id_unique.iloc[i] portion
    
    x_temp = temp[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
                            "x10", "x11", "x12", "x13", "x14"]]
    
    y_temp = temp["y"]
    
    # Initialize Model
    lreg = LinearRegression()
    
    # Specify Parameters (NOTE: this pre-specifies how many features to select (k=4), which is not what the R code does...)
    sfs1 = sfs(lreg, k_features=4, forward=False, verbose=1, scoring='neg_mean_squared_error')
    
    # Fit Model
    sfs1 = sfs1.fit(x_temp, y_temp)
    
    # Store Selected Features (NOTE: need help storing these for *each* iteration...)
    feat_names = list(sfs1.k_feature_names_)



