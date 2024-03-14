import os  # Importing the os module for operating system related functions
import sys  # Importing the sys module for system-specific parameters and functions
import pandas as pd  # Importing the pandas library as pd for data manipulation and analysis
import numpy as np  # Importing the NumPy library as np for numerical computing
from dataclasses import dataclass  # Importing the dataclass decorator from the dataclasses module

# Importing custom modules
from src.utils import save_object, select_features_by_mutual_info  # Importing specific functions from custom modules
from src.logger import logging  # Importing the logging module from custom modules
from src.exception import CustomException  # Importing the CustomException class from custom modules

@dataclass  # Decorating the class FeatureSelection with the @dataclass decorator
class FeatureSelectionConfig:  # Defining a dataclass for configuration settings related to feature selection
    selected_features_path: str = os.path.join('artifacts', "selected_features.pkl")  # Default path for saving selected features

class FeatureSelection:  # Defining a class FeatureSelection for feature selection functionality
    def __init__(self, config: FeatureSelectionConfig):  # Initializing the FeatureSelection class with a configuration object
        self.config = config  # Storing the configuration object as an attribute
        
    def initiate_feature_selection(self, X_train, y_train):  # Method to initiate feature selection
        try:  # Beginning of a try block to handle exceptions
            logging.info("Starting feature selection based on mutual information")  # Logging a message to indicate the start of feature selection

            # Selecting top features based on mutual information
            top_features_indices = select_features_by_mutual_info(X_train, y_train, n_features=10)
            
            logging.info(f"Selected top 10 feature indices: {top_features_indices}")  # Logging the indices of the selected top features
            print(f"Selected top 10 feature indices: {top_features_indices}")  # Printing the indices of the selected top features

            # Optionally, saving the indices of the selected features
            save_object(self.config.selected_features_path, top_features_indices)
            
            return top_features_indices  # Returning the indices of the selected top features
        
        except Exception as e:  # Handling any exception that might occur
            raise CustomException(e, sys)  # Raising a custom exception with the caught exception and sys module
