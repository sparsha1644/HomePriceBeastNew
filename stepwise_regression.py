import pandas as pd
import statsmodels.formula.api as smf
import os

#############################################################################
# Preliminaries
#############################################################################

os.chdir("C:\\Users\\blevi\\Documents\\python_files")
data_folder = "C:\\Users\\blevi\\Documents\\python_files"

# Import
raw_home_data = pd.read_csv("projectcombined_home_data_v2.csv")

# Training Data
data_train = raw_home_data[raw_home_data['period_begin_num'] < 43800]  # split at 12/1/2019

# Test Data
data_test = raw_home_data[raw_home_data['period_begin_num'] >= 43800]  # split at 12/1/2019


#############################################################################
# Define Variables
#############################################################################

# Training Data

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



# Test Data

data_test = data_test.rename(columns={'region_id': 'id', 'median_sale_price_yoy': 'y',
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


X_test = data_test[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
                            "x10", "x11", "x12", "x13", "x14"]]

y_test = data_test[["y"]]

data_reg_test = pd.concat([X_test, y_test, data_test[["id"]]], axis=1)  



# Get Unique IDs

id_list = data_train[["id"]]
id_unique = id_list.drop_duplicates(subset='id', keep="last")


#############################################################################
# Import Packages & Functions
#############################################################################

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from backward_selection import *


#############################################################################
# Perform Backward Selection
#############################################################################

for i in id_unique:
    

    # Declare Temporary Data For County 'i'

    temp = data_reg[data_reg['id'] == i] 
    temp_test = data_reg_test[data_reg_test['id'] == i] 
    
    
  
    # Handle Missing Data
  
    temp = temp.dropna(axis=1)
    temp_test = temp_test.dropna(axis=1)
    
    
   
    # Partition Variables

    x_temp = temp[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
                            "x10", "x11", "x12", "x13", "x14"]]
    y_temp = temp["y"]
    
    x_test = temp_test[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
                            "x10", "x11", "x12", "x13", "x14"]]
    y_test = temp_test["y"]
    
    
  
    # Run Backward Selection Algorithm
   
    out = backward_selection(x_temp, y_temp, criterion='bic', threshold=0, add_intercept=True, keep_intercept=True)
    
    

    # Store Variable Names
 
    var_names = out.params



    # Fit Test Data  
   
    reg_predict(out, x_test)






