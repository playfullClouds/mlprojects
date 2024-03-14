import os  # Importing the OS module for interacting with the operating system
import sys  # Importing the sys module 
from src.exception import CustomException  # Importing CustomException from the exception module stored in the src dir.
from src.logger import logging  # Importing the logging object from the logger module for logging
import pandas as pd  # Importing the pandas library as pd for data manipulation and analysis
import numpy as np  # Importing the NumPy library as np for numerical computing

from sklearn.model_selection import train_test_split  # Importing train_test_split function for splitting data arrays
from dataclasses import dataclass  # Importing dataclass decorator for creating data classes

# Importing components from custom modules
from src.components.data_transformation import DataTransformation, DataTransformationConfig  # Importing DataTransformation and DataTransformationConfig classes
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer  # Importing ModelTrainerConfig and ModelTrainer classes
from src.components.feature_selection import FeatureSelection, FeatureSelectionConfig  # Importing FeatureSelection and FeatureSelectionConfig classes

# Decorator to automatically generate special methods for the class below
@dataclass
class DataIngestionConfig:  # Defining a data class for data ingestion configuration
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path to the training data CSV file
    test_data_path: str = os.path.join('artifacts', "test.csv")  # Path to the test data CSV file
    val_data_path: str = os.path.join('artifacts', "val.csv")  # Path to the validation data CSV file
    raw_data_path: str = os.path.join('artifacts', "raw.csv")  # Path to the raw data CSV file

class DataIngestion:  # Defining the DataIngestion class
    def __init__(self):  # Constructor method to initialize the DataIngestion object
        self.ingestion_config = DataIngestionConfig()  # Initialize the data ingestion configuration
        
    def initiate_data_ingestion(self):  # Method to start the data ingestion process
        logging.info("Entered the data ingestion method")  # Log entry
        try:  # Try block to catch exceptions
            df = pd.read_csv(r'C:\mlprojects\BlackFriday\data\cleaned_black_friday_data.csv')  # Read the dataset into a pandas DataFrame
            logging.info('Read the dataset as dataframe')  # Log entry
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save the DataFrame to the raw CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train, test, and validate split initiated")  # Log entry
            # Split the data into training and temporary sets
            train_set, temp_set = train_test_split(df, test_size=0.4, random_state=42)
            # Split the temporary set into validation and test sets
            val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)
            
            # Save the training set to CSV file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # Save the test set to CSV file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            # Save the validation set to CSV file
            val_set.to_csv(self.ingestion_config.val_data_path, index=False, header=True)

            logging.info("Data ingestion completed")  # Log entry
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path, 
                self.ingestion_config.val_data_path
            )  # Return paths to the training, test, and validation data files
        
        except Exception as e:  # Catch exceptions
            raise CustomException(e, sys)  # Raise CustomException with the caught exception and sys module


if __name__ == "__main__":  # If the script is run as the main program
    try:  # Try block to catch exceptions
        # Data Ingestion
        data_ingestion_obj = DataIngestion()  # Create an instance of DataIngestion class
        train_data_path, test_data_path, val_data_path = data_ingestion_obj.initiate_data_ingestion()  # Start data ingestion and get paths to data files

        # Data Transformation
        data_transformation_obj = DataTransformation()  # Create an instance of DataTransformation class
        train_arr, test_arr, val_arr, preprocessor_obj_file_path = data_transformation_obj.initiate_data_transformation(train_data_path, test_data_path, val_data_path)  # Start data transformation and get transformed data and path to preprocessor object file

        # Feature Selection
        feature_selection_config = FeatureSelectionConfig()  # Create an instance of FeatureSelectionConfig class
        feature_selection_obj = FeatureSelection(feature_selection_config)  # Create an instance of FeatureSelection class
        
        # Prepare data for feature selection
        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]  # Separate features and target from training data
        top_features_indices = feature_selection_obj.initiate_feature_selection(X_train, y_train)  # Start feature selection and get top feature indices

        # Apply the selected features mask to datasets
        train_arr_selected = train_arr[:, np.append(top_features_indices, -1)]  # Include target column
        val_arr_selected = val_arr[:, np.append(top_features_indices, -1)]  # Include target column
        test_arr_selected = test_arr[:, np.append(top_features_indices, -1)]  # Include target column

        # Model Training with Selected Features
        model_trainer_obj = ModelTrainer()  # Create an instance of ModelTrainer class
        best_score, best_model_name = model_trainer_obj.initiate_model_trainer(train_arr_selected, val_arr_selected)  # Start model training with selected features and get best score and model name
        
        print(f"Best model: {best_model_name} with score: {best_score}")  # Print the best model name and score
    
    except Exception as e:  # Catch exceptions
        raise CustomException(e, sys)  # Raise CustomException with the caught exception and sys module
