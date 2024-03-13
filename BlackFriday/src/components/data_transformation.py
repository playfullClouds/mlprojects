import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation base on the type of data
        """
        try:
            # nominal variables as those without an intrinsic order
            nominal_categorical_features = ['Gender', 'Occupation', 'City_Category']
            
            #non-nominal (ordinal) variables as those with an inherent order
            """
            Typical 'Product_Category_1', 'Product_Category_2', 'Product_Category_3
            should under nominal variable/data but due to lack of domen knowledge
            and not wanting to expand my features, i included it under the ordinal
            feature.
            """
            ordinal_categorical_features = [ 'Age', 'Stay_In_Current_City_Years', 
                                                'Marital_Status', 'Product_Category_1', 
                                                'Product_Category_2', 'Product_Category_3']
            # # used for numerical data
            # num_pipeline = Pipeline(
            #     steps=[
            #         ("imputer", SimpleImputer(strategy="median"))
            #         ("scaler", StandardScaler())
            #     ]
            # )
            
            nom_pipeline = Pipeline(
                steps=[
                    # ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            ord_pipeline = Pipeline(
                steps=[
                    # ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("label_encoder", OrdinalEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Nominal categorical {nominal_categorical_features}")
            logging.info(f"Ordinal categorical: {ordinal_categorical_features}")
            
            preprocessor = ColumnTransformer(
                [
                    # ("num_pipeline", num_pipeline, numerical_columns), 
                    ("nom_pipeline", nom_pipeline, nominal_categorical_features),
                    ("ord_pipeline", ord_pipeline, ordinal_categorical_features)

                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def initiate_data_transformation(self, train_path, test_path, val_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            val_df = pd.read_csv(val_path)
            
            logging.info("Read train, test and val data completed")
            
            logging.info("obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "Purchase"
            # numerical_columns = []
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            input_feature_val_df = val_df.drop(columns=[target_column_name], axis=1)
            target_feature_val_df = val_df[target_column_name]
            
            logging.info(
                f"Applying preprocessing object on training, testing, and validating dataframe."
            )
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df).toarray()
            input_feature_val_arr = preprocessing_obj.transform(input_feature_val_df).toarray()
            
            
            # logging shapes before concatenation
            logging.info(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
            logging.info(f"Shape of target_feature_train_df: {target_feature_train_df.shape}")

            logging.info(f"Shape of input_feature_test_arr: {input_feature_test_arr.shape}")
            logging.info(f"Shape of target_feature_test_df: {target_feature_test_df.shape}")

            logging.info(f"Shape of input_feature_val_arr: {input_feature_val_arr.shape}")
            logging.info(f"Shape of target_feature_val_df: {target_feature_val_df.shape}")
            
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            val_arr = np.c_[input_feature_val_arr, np.array(target_feature_val_df)]
            
            
            logging.info(f"Saved preprocessing object")
            
            
            save_object(
                
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                val_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
            
        except Exception as e:
            raise CustomException(e, sys)