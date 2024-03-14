import os  # Importing the os module for operating system related functions
import sys  # Importing the sys module for system-specific parameters and functions

import pandas as pd  # Importing the pandas library as pd for data manipulation and analysis
import numpy as np  # Importing the NumPy library as np for numerical computing
import dill  # Importing the dill module for object serialization

from sklearn.metrics import r2_score  # Importing r2_score for evaluating regression models
from sklearn.feature_selection import mutual_info_regression  # Importing mutual_info_regression for feature selection

# Importing custom modules
from src.exception import CustomException  # Importing CustomException class from custom modules
from src.logger import logging  # Importing the logging module from custom modules


def save_object(file_path, obj):
    """
    Save an object to a file using dill serialization.

    Args:
        file_path (str): Path to save the object.
        obj: Object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)  # Extracting the directory path
        
        os.makedirs(dir_path, exist_ok=True)  # Creating the directory if it doesn't exist
        
        with open(file_path, "wb") as file_obj:  # Opening the file in binary write mode
            dill.dump(obj, file_obj)  # Serializing and saving the object to the file
    
    except Exception as e:  # Handling any exception that might occur
        raise CustomException(e, sys)  # Raising a custom exception with the caught exception and sys module
    
    
def evaluate_models(X_train, y_train, X_validate, y_validate, models):
    """
    Evaluate machine learning models on training and validation data.

    Args:
        X_train: Features of the training data.
        y_train: Target variable of the training data.
        X_validate: Features of the validation data.
        y_validate: Target variable of the validation data.
        models (dict): Dictionary containing machine learning models.

    Returns:
        dict: Dictionary containing evaluation scores for each model.
    """
    try:
        report = {}  # Initializing an empty dictionary to store evaluation scores
        
        for model_name, model in models.items():  # Iterating over each model in the dictionary
            model.fit(X_train, y_train)  # Fitting the model on the training data
            
            y_train_pred = model.predict(X_train)  # Predicting on the training data
            y_validate_pred = model.predict(X_validate)  # Predicting on the validation data
            
            # Calculating R^2 scores for training and validation data
            train_model_score = r2_score(y_train, y_train_pred)
            validation_model_score = r2_score(y_validate, y_validate_pred)
            
            # Storing the scores in the report dictionary
            report[f'{model_name}_train_score'] = train_model_score
            report[f'{model_name}_validation_score'] = validation_model_score
            
        return report  # Returning the report containing evaluation scores
           
    except Exception as e:  # Handling any exception that might occur
        raise CustomException(e, sys)  # Raising a custom exception with the caught exception and sys module 
    

def select_features_by_mutual_info(X_train, y_train, n_features=10):
    """
    Select top features based on mutual information between features and target.

    Args:
        X_train: Features of the training data.
        y_train: Target variable of the training data.
        n_features (int): Number of top features to select.

    Returns:
        np.ndarray: Indices of the selected top features.
    """
    try:
        mi_scores = mutual_info_regression(X_train, y_train)  # Computing mutual information scores
        
        # Selecting top features based on mutual information scores
        top_features_indices = np.argsort(mi_scores)[-n_features:]
        
        return top_features_indices  # Returning the indices of the selected top features
    
    except Exception as e:  # Handling any exception that might occur
        raise CustomException(e, sys)  # Raising a custom exception with the caught exception and sys module





"""
This function is used to evalute hyper-parameter model.
"""   
    
# def evaluate_models(X_train, y_train, X_validate, y_validate, models, param):
#     report = {}
#     best_model_name = None
#     best_model_score = -np.inf
#     try:
#         for model_name, model in models.items():
#             logging.info(f"Processing model: {model_name}")
#             if model_name in param:  # Check if there are hyperparameters defined for the model
#                 # Perform hyperparameter tuning
#                 random_search = RandomizedSearchCV(model, param[model_name], cv=3, n_iter=8,
#                                                    scoring='r2', verbose=1, n_jobs=-1, random_state=42)
#                 random_search.fit(X_train, y_train)
#                 best_model = random_search.best_estimator_
#                 best_params = random_search.best_params_
#             else:
#                 # Train the model with default parameters
#                 model.fit(X_train, y_train)
#                 best_model = model
#                 best_params = 'Default parameters'
#                 # Logging success
#                 logging.error(f"Exception occurred while processing model {model_name}: {str(e)}")

#             # Evaluate the best model (tuned or default) on the validation data
#             y_validate_pred = best_model.predict(X_validate)
#             validation_model_score = r2_score(y_validate, y_validate_pred)
            
#             # Update the report with the validation score and best parameters
#             report[model_name] = {
#                 'validation_score': validation_model_score,
#                 'best_params': best_params
#             }

#             if validation_model_score > best_model_score:
#                 best_model_score = validation_model_score
#                 best_model_name = model_name

        
#         if best_model_name:
#             # Append the best model information to the report if we found a best model
#             report['best_model'] = {
#                 'name': best_model_name,
#                 'score': best_model_score,
#                 'parameters': report[best_model_name]['best_params']
#             }
#         else:
#             logging.error("No best model found. Please check the model configurations and data.")
#         return report
           
#     except Exception as e:
#         raise CustomException(e, sys)


"""
This function is use to evaluate before hyper-parameter and after hyper-parameter models.
It get the best model score from both model. It uses the highest model score.
"""

# def evaluate_models(X_train, y_train, X_validate, y_validate, models, params={}):
#     report = {"pre_tuning": {}, "post_tuning": {}}
#     best_scores = {
#         "pre_tuning": {"score": -np.inf, "name": None, "is_tuned": False},
#         "post_tuning": {"score": -np.inf, "name": None, "is_tuned": True}
#         }

#     try:
#         for model_name, model in models.items():
#             # Initial evaluation with default parameters
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_validate)
#             score = r2_score(y_validate, y_pred)
#             report["pre_tuning"][model_name] = {"score": score, "parameters": model.get_params()}
#             # report["pre_tuning"][model_name] = score
            
#             if score > best_scores["pre_tuning"]["score"]:
#                 best_scores["pre_tuning"].update({"score": score, "name": model_name})


#             # Hyperparameter tuning if parameters provided
#             if model_name in params and params[model_name]:
#                 tuner = RandomizedSearchCV(model, params[model_name], n_iter=8, 
#                                            scoring='r2', cv=3, random_state=42)
#                 tuner.fit(X_train, y_train)
#                 tuned_model = tuner.best_estimator_
#                 tuned_score = r2_score(y_validate, tuned_model.predict(X_validate))
#                 report["post_tuning"][model_name] = tuned_score
                
#                 if tuned_score > best_scores["post_tuning"]["score"]:
#                     best_scores["post_tuning"].update({"score": tuned_score, "name": model_name})

#     except Exception as e:
#         raise CustomException(e, sys)

#     return report, best_scores