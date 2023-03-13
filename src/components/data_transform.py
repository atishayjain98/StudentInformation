# Importing Libraries 
import os
import sys
sys.path.append(".")

import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
           
    def get_data_transformer_obj(self):
        '''
        This function is responsible for creating Data Transformation object
        '''
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]
            
            logging.info("Categorical and Numerical Columns seperated.")
            logging.info(f"Categorical Feature : {categorical_columns}")
            logging.info(f"Numerical Feature : {numerical_columns}")
            
            num_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy = 'median')), # Handling missing value
                    ("Scaler", StandardScaler(with_mean=False)) # Standard scaling on numerical data
                ]
            )
            
            logging.info("Numerical Columns scaling completed.")
            
            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy='most_frequent')),
                    ("OH_Encoding", OneHotEncoder()),
                    ("Scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Categorical Columns encoding completed.")
            
            preprocessor = ColumnTransformer(
                [
                    ("Numerical_Pipeline", num_pipeline, numerical_columns),
                    ("Categorical_Pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    # Start the transformation
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading of train and test data completed.")
            logging.info("Obtaining preprocessing object")
            
            preprocessor_obj = self.get_data_transformer_obj()
            
            target_col_name = "math score"
            
            input_feature_train_df = train_df.drop([target_col_name], axis = 1)
            target_feature_train_df = train_df[target_col_name]
            
            input_feature_test_df = test_df.drop([target_col_name], axis = 1)
            target_feature_test_df = test_df[target_col_name]
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            input_feature_training_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_testing_arr = preprocessor_obj.fit_transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_training_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_testing_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            
            return (
                train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        