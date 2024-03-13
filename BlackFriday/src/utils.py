import os
import sys

import pandas as pd
import numpy as np
import dill

from sklearn.metrics import (
    r2_score,
    # mean_squared_error,
    # mean_absolute_error
)


from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    
    
    except Exception as e:
        raise CustomException(e, sys)
    
    
    
# def evaluate_models(X_train, y_train, X_validate, y_validate, models):
#     try:
#         report = {}
        
#         for model_name, model in models.items():
#         # for i in range(len(list(models))):
            
#         #     model = list(models.values())[i]
            
#             model.fit(X_train, y_train)
            
#             y_train_pred = model.predict(X_train)
#             y_validate_pred = model.predict(X_validate)
            
            
#             train_model_score = r2_score(y_train, y_train_pred)
#             validation_model_score = r2_score(y_validate, y_validate_pred)
            
#             # report[model_name] = {
#             #     'train_score': train_model_score,
#             #     'validation_score': validation_model_score
#             # }
            
#             report[f'{model_name}_train_score'] = train_model_score
#             report[f'{model_name}_validation_score'] = validation_model_score
            
#             # report[list(models.keys())[1]] = validation_model_score

#         return report
           
#     except Exception as e:
#         raise CustomException(e, sys) 



"""
This function is used to evalute hyper-parameter model.
"""   
    
def evaluate_models(X_train, y_train, X_validate, y_validate, models, param):
    report = {}
    best_model_name = None
    best_model_score = -np.inf
    try:
        for model_name, model in models.items():
            logging.info(f"Processing model: {model_name}")
            if model_name in param:  # Check if there are hyperparameters defined for the model
                # Perform hyperparameter tuning
                random_search = RandomizedSearchCV(model, param[model_name], cv=3, n_iter=8,
                                                   scoring='r2', verbose=1, n_jobs=-1, random_state=42)
                random_search.fit(X_train, y_train)
                best_model = random_search.best_estimator_
                best_params = random_search.best_params_
            else:
                # Train the model with default parameters
                model.fit(X_train, y_train)
                best_model = model
                best_params = 'Default parameters'
                # Logging success
                logging.error(f"Exception occurred while processing model {model_name}: {str(e)}")

            # Evaluate the best model (tuned or default) on the validation data
            y_validate_pred = best_model.predict(X_validate)
            validation_model_score = r2_score(y_validate, y_validate_pred)
            
            # Update the report with the validation score and best parameters
            report[model_name] = {
                'validation_score': validation_model_score,
                'best_params': best_params
            }

            if validation_model_score > best_model_score:
                best_model_score = validation_model_score
                best_model_name = model_name

        
        if best_model_name:
            # Append the best model information to the report if we found a best model
            report['best_model'] = {
                'name': best_model_name,
                'score': best_model_score,
                'parameters': report[best_model_name]['best_params']
            }
        else:
            logging.error("No best model found. Please check the model configurations and data.")
        return report
           
    except Exception as e:
        raise CustomException(e, sys)