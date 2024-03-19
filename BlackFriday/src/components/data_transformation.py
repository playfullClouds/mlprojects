# import sys
# import os
# from dataclasses import dataclass

# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# from src.exception import CustomException
# from src.logger import logging
# from src.utils import save_object


# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    
# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()
        
#     def get_data_transformer_object(self):
#         """
#         This function is responsible for data transformation base on the type of data
#         """
#         try:
#             # nominal variables as those without an intrinsic order
#             nominal_categorical_features = ['Gender', 'Occupation', 'City_Category']
            
#             #non-nominal (ordinal) variables as those with an inherent order
#             """
#             Typical 'Product_Category_1', 'Product_Category_2', 'Product_Category_3
#             should under nominal variable/data but due to lack of domen knowledge
#             and not wanting to expand my features, i included it under the ordinal
#             feature.
#             """
#             ordinal_categorical_features = [ 'Age', 'Stay_In_Current_City_Years', 
#                                                 'Marital_Status', 'Product_Category_1', 
#                                                 'Product_Category_2', 'Product_Category_3']
#             # # used for numerical data
#             # num_pipeline = Pipeline(
#             #     steps=[
#             #         ("imputer", SimpleImputer(strategy="median"))
#             #         ("scaler", StandardScaler())
#             #     ]
#             # )
            
#             nom_pipeline = Pipeline(
#                 steps=[
#                     # ("imputer", SimpleImputer(strategy="most_frequent")),
#                     ("one_hot_encoder", OneHotEncoder()),
#                     ("scaler", StandardScaler(with_mean=False))
#                 ]
#             )
            
#             ord_pipeline = Pipeline(
#                 steps=[
#                     # ("imputer", SimpleImputer(strategy="most_frequent")),
#                     ("Ordinal_encoder", OrdinalEncoder()),
#                     ("scaler", StandardScaler(with_mean=False))
#                 ]
#             )
            
#             logging.info(f"Nominal categorical {nominal_categorical_features}")
#             logging.info(f"Ordinal categorical: {ordinal_categorical_features}")
            
#             preprocessor = ColumnTransformer(
#                 [
#                     # ("num_pipeline", num_pipeline, numerical_columns), 
#                     ("nom_pipeline", nom_pipeline, nominal_categorical_features),
#                     ("ord_pipeline", ord_pipeline, ordinal_categorical_features)

#                 ]
#             )
            
#             return preprocessor
            
#         except Exception as e:
#             raise CustomException(e, sys)
        
        
#     def initiate_data_transformation(self, train_path, test_path, val_path):
        
#         try:
#             train_df = pd.read_csv(train_path)
#             test_df = pd.read_csv(test_path)
#             val_df = pd.read_csv(val_path)
            
#             logging.info("Read train, test and val data completed")
            
#             logging.info("obtaining preprocessing object")
            
#             preprocessing_obj = self.get_data_transformer_object()
            
#             target_column_name = "Purchase"
#             # numerical_columns = []
            
#             input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
#             target_feature_train_df = train_df[target_column_name]
            
#             input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
#             target_feature_test_df = test_df[target_column_name]
            
#             input_feature_val_df = val_df.drop(columns=[target_column_name], axis=1)
#             target_feature_val_df = val_df[target_column_name]
            
#             logging.info(
#                 f"Applying preprocessing object on training, testing, and validating dataframe."
#             )
            
#             input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df).toarray()
#             input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df).toarray()
#             input_feature_val_arr = preprocessing_obj.transform(input_feature_val_df).toarray()
            
            
#             # logging shapes before concatenation
#             logging.info(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
#             logging.info(f"Shape of target_feature_train_df: {target_feature_train_df.shape}")

#             logging.info(f"Shape of input_feature_test_arr: {input_feature_test_arr.shape}")
#             logging.info(f"Shape of target_feature_test_df: {target_feature_test_df.shape}")

#             logging.info(f"Shape of input_feature_val_arr: {input_feature_val_arr.shape}")
#             logging.info(f"Shape of target_feature_val_df: {target_feature_val_df.shape}")
            
            
#             train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
#             test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
#             val_arr = np.c_[input_feature_val_arr, np.array(target_feature_val_df)]
            
            
#             logging.info(f"Saved preprocessing object")
            
            
#             save_object(
                
#                 file_path = self.data_transformation_config.preprocessor_obj_file_path,
#                 obj = preprocessing_obj
#             )
            
#             return (
#                 train_arr,
#                 test_arr,
#                 val_arr,
#                 self.data_transformation_config.preprocessor_obj_file_path
#             )
           
            
#         except Exception as e:
#             raise CustomException(e, sys)





# import sys
# import os
# from dataclasses import dataclass

# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# from src.exception import CustomException
# from src.logger import logging
# from src.utils import save_object

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()

#     def get_data_transformer_object(self):
#         """
#         This function is responsible for data transformation based on the type of data.
#         """
#         try:
#             # Combine all categorical features under one list for ordinal encoding
#             categorical_features = ['Gender', 'Occupation', 'City_Category', 'Age', 'Stay_In_Current_City_Years', 
#                                     'Marital_Status', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
            
#             # Creating a single pipeline for both nominal and ordinal features since
#             # we're treating all as ordinal for simplicity and to avoid feature expansion.
#             cat_pipeline = Pipeline(
#                 steps=[
#                     ("imputer", SimpleImputer(strategy="most_frequent")),
#                     ("ordinal_encoder", OrdinalEncoder()), # No more OneHotEncoder
#                     # ("scaler", StandardScaler(with_mean=False))
#                 ]
#             )
            
#             logging.info(f"Categorical features: {categorical_features}")
            
#             # No longer distinguishing between nominal and ordinal features in preprocessing
#             preprocessor = ColumnTransformer(
#                 transformers=[
#                     ("cat_pipeline", cat_pipeline, categorical_features)
#                 ]
#             )
            
#             return preprocessor
            
#         except Exception as e:
#             raise CustomException(e, sys)
        

#     def initiate_data_transformation(self, train_path, test_path, val_path):
#         try:
#             train_df = pd.read_csv(train_path)
#             test_df = pd.read_csv(test_path)
#             val_df = pd.read_csv(val_path)

#             logging.info("Read train, test, and val data completed")
#             logging.info("Obtaining preprocessing object")
            
#             preprocessing_obj = self.get_data_transformer_object()
            
#             target_column_name = "Purchase"
            
#             input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
#             target_feature_train_df = train_df[target_column_name]
            
#             input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
#             target_feature_test_df = test_df[target_column_name]
            
#             input_feature_val_df = val_df.drop(columns=[target_column_name], axis=1)
#             target_feature_val_df = val_df[target_column_name]
            
#             logging.info("Applying preprocessing object on training, testing, and validating dataframe.")
            
#             input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
#             input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
#             input_feature_val_arr = preprocessing_obj.transform(input_feature_val_df)

#             logging.info(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
#             logging.info(f"Shape of target_feature_train_df: {target_feature_train_df.shape}")
#             logging.info(f"Shape of input_feature_test_arr: {input_feature_test_arr.shape}")
#             logging.info(f"Shape of target_feature_test_df: {target_feature_test_df.shape}")
#             logging.info(f"Shape of input_feature_val_arr: {input_feature_val_arr.shape}")
#             logging.info(f"Shape of target_feature_val_df: {target_feature_val_df.shape}")
            
#             logging.info("")
            
#             # Example for logging the output type directly (overall type, not per-column)
#             logging.info(f"Data type of the transformed training features: {input_feature_train_arr.dtype}")

#             # If you wish to infer and log what the data types might be after transformation
#             # This is more of a hack and should be used carefully
#             dummy_columns = ['feature_' + str(i) for i in range(input_feature_train_arr.shape[1])]
#             dummy_df = pd.DataFrame(input_feature_train_arr, columns=dummy_columns)
#             logging.info(f"Implied data types of transformed training features (for logging purposes):\n{dummy_df.dtypes}")

            
#             train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
#             test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
#             val_arr = np.c_[input_feature_val_arr, np.array(target_feature_val_df)]
            
#             logging.info("Saved preprocessing object")
            
#             save_object(
#                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
#                 obj=preprocessing_obj
#             )
            
#             return (
#                 train_arr,
#                 test_arr,
#                 val_arr,
#                 self.data_transformation_config.preprocessor_obj_file_path
#             )
            
#         except Exception as e:
#             raise CustomException(e, sys)







# import sys
# import os
# from dataclasses import dataclass

# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.base import BaseEstimator, TransformerMixin

# from src.exception import CustomException
# from src.logger import logging
# from src.utils import save_object

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

# # Custom transformer that ensures data remains as integers
# class EnsureIntTransformer(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self  # Nothing to fit

#     def transform(self, X):
#         # Ensure the output is integer, useful after ordinal encoding
#         return X.astype(int)

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()

#     def get_data_transformer_object(self):
#         try:
#             # Assuming these are all the features and are categorical
#             categorical_features = ['Gender', 'Occupation', 'City_Category', 'Age', 
#                                     'Stay_In_Current_City_Years', 'Marital_Status', 
#                                     'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
            
#             # Pipeline for categorical features
#             cat_pipeline = Pipeline([
#                 ("imputer", SimpleImputer(strategy="most_frequent")),
#                 ("ordinal_encoder", OrdinalEncoder()),
#                 ("ensure_int", EnsureIntTransformer())  # Ensures output remains as integer
#             ])
            
#             logging.info(f"Categorical features: {categorical_features}")
            
#             preprocessor = ColumnTransformer(transformers=[
#                 ("cat_pipeline", cat_pipeline, categorical_features)
#             ], remainder='passthrough')  # This allows for other feature types to be added easily
            
#             return preprocessor
            
#         except Exception as e:
#             raise CustomException(e, sys)

    # def initiate_data_transformation(self, train_path, test_path, val_path):
    #     try:
    #         train_df = pd.read_csv(train_path)
    #         test_df = pd.read_csv(test_path)
    #         val_df = pd.read_csv(val_path)

    #         logging.info("Read train, test, and val data completed")
            
    #         preprocessing_obj = self.get_data_transformer_object()
            
    #         target_column_name = "Purchase"
            
    #         input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
    #         target_feature_train_df = train_df[target_column_name]
            
    #         input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
    #         target_feature_test_df = test_df[target_column_name]
            
    #         input_feature_val_df = val_df.drop(columns=[target_column_name], axis=1)
    #         target_feature_val_df = val_df[target_column_name]
            
    #         logging.info("Applying preprocessing object on training, testing, and validating dataframe.")
            
    #         input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
    #         input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
    #         input_feature_val_arr = preprocessing_obj.transform(input_feature_val_df)

    #         logging.info(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
    #         logging.info(f"Shape of target_feature_train_df: {target_feature_train_df.shape}")
    #         logging.info(f"Shape of input_feature_test_arr: {input_feature_test_arr.shape}")
    #         logging.info(f"Shape of target_feature_test_df: {target_feature_test_df.shape}")
    #         logging.info(f"Shape of input_feature_val_arr: {input_feature_val_arr.shape}")
    #         logging.info(f"Shape of target_feature_val_df: {target_feature_val_df.shape}")
            
    #         logging.info("")
            
    #         # Example for logging the output type directly (overall type, not per-column)
    #         logging.info(f"Data type of the transformed training features: {input_feature_train_arr.dtype}")

    #         # If you wish to infer and log what the data types might be after transformation
    #         # This is more of a hack and should be used carefully
    #         dummy_columns = ['feature_' + str(i) for i in range(input_feature_train_arr.shape[1])]
    #         dummy_df = pd.DataFrame(input_feature_train_arr, columns=dummy_columns)
    #         logging.info(f"Implied data types of transformed training features (for logging purposes):\n{dummy_df.dtypes}")

            
    #         # Save the preprocessing object for later use
    #         save_object(
    #             file_path=self.data_transformation_config.preprocessor_obj_file_path,
    #             obj=preprocessing_obj
    #         )
            
    #         return (
    #             np.c_[input_feature_train_arr, np.array(target_feature_train_df)],
    #             np.c_[input_feature_test_arr, np.array(target_feature_test_df)],
    #             np.c_[input_feature_val_arr, np.array(target_feature_val_df)],
    #             self.data_transformation_config.preprocessor_obj_file_path
    #         )
            
    #     except Exception as e:
    #         raise CustomException(e, sys)


 
 
import sys
import os
from dataclasses import dataclass 
 
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")


class CustomOrdinalEncoder(OrdinalEncoder):
    """
    This encoder extends the functionality of OrdinalEncoder to handle unknown categories
    by assigning them a specific unknown value during transformation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handle_unknown = 'use_encoded_value'
        self.unknown_value = -1  # You can choose any value that doesn't clash with your known categories

    def fit(self, X, y=None):
        return super().fit(X, y)
    
    def transform(self, X):
        try:
            return super().transform(X)
        except ValueError:
            X_tr = np.copy(X)
            for col in range(X.shape[1]):
                unique_values = self.categories_[col]
                unknown_indices = ~np.isin(X[:, col], unique_values)
                X_tr[unknown_indices, col] = self.unknown_value
            return super().transform(X_tr)

# Custom transformer that ensures data remains as integers
class EnsureIntTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        # Ensure the output is integer, useful after ordinal encoding
        return X.astype(int)

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_features = ['Gender', 'Occupation', 'City_Category', 'Age', 
                                    'Stay_In_Current_City_Years', 'Marital_Status', 
                                    'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
            
            # Pipeline for categorical features
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("custom_ordinal_encoder", CustomOrdinalEncoder()),
                ("ensure_int", EnsureIntTransformer())  # Ensures output remains as integer
            ])
            
            logging.info(f"Categorical features: {categorical_features}")
            
            preprocessor = ColumnTransformer(transformers=[
                ("cat_pipeline", cat_pipeline, categorical_features)
            ], remainder='passthrough')  # This allows for other feature types to be added easily
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)

    # The rest of your class implementation remains unchanged

 
 
 
    def initiate_data_transformation(self, train_path, test_path, val_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            val_df = pd.read_csv(val_path)

            logging.info("Read train, test, and val data completed")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "Purchase"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            input_feature_val_df = val_df.drop(columns=[target_column_name], axis=1)
            target_feature_val_df = val_df[target_column_name]
            
            logging.info("Applying preprocessing object on training, testing, and validating dataframe.")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            input_feature_val_arr = preprocessing_obj.transform(input_feature_val_df)

            logging.info(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
            logging.info(f"Shape of target_feature_train_df: {target_feature_train_df.shape}")
            logging.info(f"Shape of input_feature_test_arr: {input_feature_test_arr.shape}")
            logging.info(f"Shape of target_feature_test_df: {target_feature_test_df.shape}")
            logging.info(f"Shape of input_feature_val_arr: {input_feature_val_arr.shape}")
            logging.info(f"Shape of target_feature_val_df: {target_feature_val_df.shape}")
            
            logging.info("")
            
            # Example for logging the output type directly (overall type, not per-column)
            logging.info(f"Data type of the transformed training features: {input_feature_train_arr.dtype}")

            # If you wish to infer and log what the data types might be after transformation
            # This is more of a hack and should be used carefully
            dummy_columns = ['feature_' + str(i) for i in range(input_feature_train_arr.shape[1])]
            dummy_df = pd.DataFrame(input_feature_train_arr, columns=dummy_columns)
            logging.info(f"Implied data types of transformed training features (for logging purposes):\n{dummy_df.dtypes}")

            
            # Save the preprocessing object for later use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                np.c_[input_feature_train_arr, np.array(target_feature_train_df)],
                np.c_[input_feature_test_arr, np.array(target_feature_test_df)],
                np.c_[input_feature_val_arr, np.array(target_feature_val_df)],
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
    