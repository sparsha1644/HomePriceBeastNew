"""
    Utility for doing backward selection on an OLS model.
"""

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def backward_selection(X, y, criterion='bic', threshold=0, add_intercept=True, keep_intercept=True):
    """ Train a multivariate regression model with variable selection via
        backward selection. We are assuming that X and y are DataFrames (or series)
    """
    
    # If desired, add an intercept
    if add_intercept:
        X =  X.assign(Intercept=1)
        min_variables = 1
    else:
        min_variables = 0
        
    # Fit the base model with all variables
    model = OLS(y,X).fit()
    fitness = getattr(model, criterion)
    dFitness = np.inf
    
    # Iteratively try dropping variables until we see no improvement, or run out
    while (dFitness > threshold) and (X.shape[1]>min_variables):
        # Pullthe list of columns which we *could* drop
        if keep_intercept:
            all_vars = [c for c in X.columns if c!='Intercept']
        else:
            all_vars = [c for x in X.columns]
            
        # Fit a reduced-form model
        sub_models = [OLS(y,X.drop(c,axis=1)).fit() for c in all_vars]
        sub_scores = [getattr(m,criterion) for m in sub_models]
        best_model = np.argmin(sub_scores)

        # If we can improve the fit criterion by dropping a variable, iterate        
        if sub_scores[best_model] < fitness:
            dFitness = fitness - sub_scores[best_model]
            fitness = sub_scores[best_model]
            model = sub_models[best_model]
            
            X = X.drop(all_vars[best_model], axis=1)
            
        # Otherwise stop: the current model is good enough!
        else:
            return model
    
    return model