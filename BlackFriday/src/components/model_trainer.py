import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.metrics import (
    r2_score,
    # mean_squared_error,
    # mean_absolute_error
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
   
# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()
        
#     def get_best_model_from_report(self, model_report):
#         # Extract only validation scores and model names
#         validation_scores = {key.replace('_validation_score', ''): score for key, score in model_report.items() if '_validation_score' in key}
       
#         # Find the model with the highest validation score
#         best_model_name = max(validation_scores, key=validation_scores.get)
#         best_model_score = validation_scores[best_model_name]
        
#         return best_model_name, best_model_score
    
#     def initiate_model_trainer(self, train_array, validate_array):
        
#         try:
#             logging.info("Split training, validation, and test input data")
            
#             X_train, y_train = train_array[:, :-1], train_array[:, -1]
#             X_validate, y_validate = validate_array[:, :-1], validate_array[:, -1]

#             models = {
#                 "Random Forest": RandomForestRegressor(),
#                 "Gradient Boosting": GradientBoostingRegressor(),
#                 "XGBRegressor": XGBRegressor(),
#                 "CatBoosting Regressor": CatBoostRegressor(verbose=False)
#             }
            
#             model_report = evaluate_models(X_train=X_train, y_train=y_train, X_validate=X_validate, 
#                                            y_validate=y_validate,models=models)
            
            
#             # Use the utility function to get the best model based on validation scores
#             best_model_name, best_model_score = self.get_best_model_from_report(model_report)
            
            
#             if best_model_score < 0.6:
#                 raise CustomException("No best model found with sufficient score")
            
#             logging.info(f"Best model: {best_model_name} with R^2 score: {best_model_score}")

#             # Access the best model using its name
#             best_model = models[best_model_name]
#             best_model.fit(X_train, y_train)  # Ensure the best model is retrained on the full training set
            
#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )
            
#             predicted = best_model.predict(X_validate)
#             r2_square = r2_score(y_validate, predicted)
            
#             return r2_square, best_model_name
            
        
#         except Exception as e:
#             raise CustomException(e, sys)


"""
The function is for Hyper-Parameter. The reason why i am not using it, is because
the highest model score is lower than the highest model score without Hyper-parameter.     
"""

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()   
        
        
#     def initiate_model_trainer(self, train_array, validate_array):
        
#         try:
#             logging.info("Split training, validation, and test input data")
            
#             X_train, y_train = train_array[:, :-1], train_array[:, -1]
#             X_validate, y_validate = validate_array[:, :-1], validate_array[:, -1]
#             # X_test, y_test = test_array[:, :-1], test_array[:, -1]

#             models = {
#                 "Random Forest": RandomForestRegressor(),
#                 "Gradient Boosting": GradientBoostingRegressor(),
#                 "XGBRegressor": XGBRegressor(),
#                 "CatBoosting Regressor": CatBoostRegressor(verbose=False)
#             }
            
#             params = {
#             "Random Forest": {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5]},
#             "Gradient Boosting": {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200], 'max_depth': [3, 5]},
#             "XGBRegressor": {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200], 'max_depth': [3, 5]},
#             "CatBoosting Regressor": {'learning_rate': [0.01, 0.1], 'depth': [4, 6], 'iterations': [100, 200]}
#             }
            
#             model_report = evaluate_models(X_train=X_train, y_train=y_train, X_validate=X_validate, 
#                                            y_validate=y_validate,models=models,param=params)
            
            
#             # Check if best_model information is present in the report
#             if 'best_model' in model_report and model_report['best_model']['score'] is not None:
#                 best_model_name = model_report['best_model']['name']
#                 best_model_score = model_report['best_model']['score']
                
#                 if best_model_score < 0.6:
#                     raise CustomException("No best model found with sufficient score")
                
#                 logging.info(f"Best model: {best_model_name} with R^2 score: {best_model_score}")
                
#                 # Access and retrain the best model using its name
#                 best_model = models[best_model_name].fit(X_train, y_train)
                
#                 save_object(
#                     file_path=self.model_trainer_config.trained_model_file_path,
#                     obj=best_model
#                 )
                
#                 predicted = best_model.predict(X_validate)
#                 r2_square = r2_score(y_validate, predicted)
                
#                 return r2_square, best_model_name
#             else:
#                 raise CustomException("Failed to identify the best model from evaluation.")
            
        
#         except Exception as e:
#             raise CustomException(e, sys)


"""
This function is to compare the highest model score between before Hyper-Parameter and
after Hyper-Parameter. It takes the highest model between both and use the model.
It also print out the best model score name and best model score.
 """   

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, validate_array):
        
        X_train, y_train = train_array[:, :-1], train_array[:, -1]
        X_validate, y_validate = validate_array[:, :-1], validate_array[:, -1]
            
        models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False)
            }
            
        params = {
            "Random Forest": {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5]},
            "Gradient Boosting": {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200], 'max_depth': [3, 5]},
            "XGBRegressor": {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200], 'max_depth': [3, 5]},
            "CatBoosting Regressor": {'learning_rate': [0.01, 0.1], 'depth': [4, 6], 'iterations': [100, 200]}
            }
        
        
        try:
            logging.info("Initiating model training and evaluation...")
            
            # Evaluate models and determine the best model before and after hyperparameter tuning
            report, best_scores = evaluate_models(X_train, y_train, X_validate, 
                                                  y_validate, models, params)
            
            # Determine the highest performing model overall
            best_model_key = max(["pre_tuning", "post_tuning"], key=lambda x: best_scores[x]["score"])
            best_model_info = best_scores[best_model_key]
            best_model_name = best_model_info["name"]
            best_model_score = best_model_info["score"]

            logging.info(f"Overall best model: {best_model_name} with R^2 score: {best_model_score}")

            # If the best model comes from post-tuning phase, it needs special handling to retrain
            best_model = models[best_model_name]
            if best_model_info['is_tuned']:
                # Assuming best parameters are stored in the report for the post-tuning phase
                best_params = report["post_tuning"][best_model_name]['parameters']
                best_model.set_params(**best_params)
            
            
            # Check if the best model's score is below the threshold
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient score")
            
            # Retrain the best model on the full training set
            best_model.fit(X_train, y_train)
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj=best_model
                )
            logging.info(f"Best model saved successfully at {self.model_trainer_config.trained_model_file_path}")
            
            # Return the name and score of the best model, along with a message indicating success
            return {
                "message": "Model training and evaluation completed successfully.",
                "best_model_name": best_model_name,
                "best_model_score": best_model_score,
                # "model_saved_at": self.model_trainer_config.trained_model_file_path
            }

            
        except Exception as e:
            raise CustomException(e, sys)
   