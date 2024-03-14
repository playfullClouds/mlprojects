import sys  # Importing the sys module for system-specific parameters and functions
import os  # Importing the os module for operating system related functions
from dataclasses import dataclass  # Importing the dataclass decorator from the dataclasses module

import numpy as np  # Importing the NumPy library as np for numerical computing
import pandas as pd  # Importing the pandas library as pd for data manipulation and analysis
from sklearn.compose import ColumnTransformer  # Importing ColumnTransformer for feature transformation
from sklearn.impute import SimpleImputer  # Importing SimpleImputer for imputing missing values
from sklearn.pipeline import Pipeline  # Importing Pipeline for creating a pipeline of transformers
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler  # Importing encoders and scalers

# Importing custom modules
from src.exception import CustomException  # Importing CustomException class from custom modules
from src.logger import logging  # Importing the logging module from custom modules
from src.utils import save_object  # Importing save_object function from custom modules

@dataclass  # Decorating the class DataTransformationConfig with the @dataclass decorator
class DataTransformationConfig:  # Defining a dataclass for configuration settings related to data transformation
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")  # Default path for saving the preprocessor object

class DataTransformation:  # Defining a class DataTransformation for data transformation functionality
    def __init__(self):  # Initializing the DataTransformation class
        self.data_transformation_config = DataTransformationConfig()  # Storing the configuration object as an attribute
        
    def get_data_transformer_object(self):  # Method to get the data transformer object
        """
        This function is responsible for data transformation base on the type of data
        """
        try:  # Beginning of a try block to handle exceptions
            # Defining nominal and ordinal categorical features
            nominal_categorical_features = ['Gender', 'Occupation', 'City_Category']
            
            """
            Typical 'Product_Category_1', 'Product_Category_2', 'Product_Category_3
            should under nominal variable/data but due to lack of domen knowledge
            and not wanting to expand my features, i included it under the ordinal
            feature.
            """
            ordinal_categorical_features = ['Age', 'Stay_In_Current_City_Years', 
                                            'Marital_Status', 'Product_Category_1',
                                            'Product_Category_2', 'Product_Category_3']
            
              # # used for numerical data
            # num_pipeline = Pipeline(
            #     steps=[
            #         ("imputer", SimpleImputer(strategy="median"))
            #         ("scaler", StandardScaler())
            #     ]
            # )
            
            # Defining pipelines for nominal and ordinal categorical features
            nom_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder()),  # One-hot encoding
                    ("scaler", StandardScaler(with_mean=False))  # Standard scaling
                ]
            )
            ord_pipeline = Pipeline(
                steps=[
                    ("Ordinal_encoder", OrdinalEncoder()),  # Ordinal encoding
                    ("scaler", StandardScaler(with_mean=False))  # Standard scaling
                ]
            )
            
            # Logging categorical features
            logging.info(f"Nominal categorical: {nominal_categorical_features}")
            logging.info(f"Ordinal categorical: {ordinal_categorical_features}")
            
            # Creating a ColumnTransformer to apply different transformations to different columns
            preprocessor = ColumnTransformer(
                [
                    ("nom_pipeline", nom_pipeline, nominal_categorical_features),  # Applying nominal pipeline to nominal features
                    ("ord_pipeline", ord_pipeline, ordinal_categorical_features)  # Applying ordinal pipeline to ordinal features
                ]
            )
            
            return preprocessor  # Returning the preprocessor object
            
        except Exception as e:  # Handling any exception that might occur
            raise CustomException(e, sys)  # Raising a custom exception with the caught exception and sys module
        
  
        
    def initiate_data_transformation(self, train_path, test_path, val_path):  # Method to initiate data transformation
        try:  # Beginning of a try block to handle exceptions
            # Reading train, test, and validation data into DataFrames
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            val_df = pd.read_csv(val_path)
            
            # Logging completion of reading data
            logging.info("Read train, test and val data completed")
            
            # Storing feature names (excluding the target column name)
            self.feature_names = list(train_df.columns[:-1])
            
            # Logging obtaining preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()  # Getting the data transformer object
            
            target_column_name = "Purchase"  # Target column name
            
            # Extracting input and target features from train, test, and validation DataFrames
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            input_feature_val_df = val_df.drop(columns=[target_column_name], axis=1)
            target_feature_val_df = val_df[target_column_name]
            
            # Logging applying preprocessing object on dataframes
            logging.info("Applying preprocessing object on training, testing, and validating dataframe.")
            
            # Transforming input features using the preprocessing object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df).toarray()
            input_feature_val_arr = preprocessing_obj.transform(input_feature_val_df).toarray()
            
            # Logging shapes before concatenation
            logging.info(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
            logging.info(f"Shape of target_feature_train_df: {target_feature_train_df.shape}")
            logging.info(f"Shape of input_feature_test_arr: {input_feature_test_arr.shape}")
            logging.info(f"Shape of target_feature_test_df: {target_feature_test_df.shape}")
            logging.info(f"Shape of input_feature_val_arr: {input_feature_val_arr.shape}")
            logging.info(f"Shape of target_feature_val_df: {target_feature_val_df.shape}")
            
            # Concatenating input and target features
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            val_arr = np.c_[input_feature_val_arr, np.array(target_feature_val_df)]
            
            # Logging saving preprocessing object
            logging.info("Saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                val_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )  # Returning the transformed data and the path to the saved preprocessing object
            
        except Exception as e:  # Handling any exception that might occur
            raise CustomException(e, sys)  # Raising a custom exception with the caught exception and sys module

