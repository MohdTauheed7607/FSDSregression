import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from src.Exception import CustomException
from src.logger import logging
from src.utils import save_obj

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('Artifact','DataTransformation','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation initiated')
            # defining categorical and numerical columns

            categorical_cols=['cut','color','clarity']
            numerical_cols=['carat','depth','table','x','y','z']

            # defining cat_categories ranking

            logging.info('defining categories ranking')

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info('numerical pipeline initiated')

            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
            )
            
            logging.info('categorical pipeline initiated')

            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('encoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())

                ]
            )

            logging.info('combining both the pipeline')

            preprocessor=ColumnTransformer([
                ('numpipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
                ])
            
            logging.info('pipeline completed')
            logging.info('returning the preprocessor object')
        
            return preprocessor

        except Exception as e:
            logging.info('Exception occured in get data transformation object method',e)
            raise CustomException(e,sys)
        

    

    def Initiate_data_transformation(self,train_path,test_path):
        try:
            preprocessor_obj=self.get_data_transformation_object()

            # Reading train and test data

            logging.info('reading train and test dataframe')
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info('Dividing training data into dependent and independent feature')
            X_train=train_df.drop(columns=['id','price'])
            y_train=train_df['price']

            logging.info('Dividing testing data into dependent and independent feature')
            X_test=test_df.drop(columns=['id','price'])
            y_test=test_df['price']

            # Transforming using preprocessing object

            logging.info('Data transformation has been started using preprocessor obj')
            X_train_arr=preprocessor_obj.fit_transform(X_train)
            X_test_arr=preprocessor_obj.transform(X_test)

            logging.info('Data transformation has done')
            save_obj(
            obj=preprocessor_obj,
            file_path=self.data_transformation_config.preprocessor_obj_file_path
            )

            logging.info('Preprocessor obj has been saved into preprocessor.pkl file')

            return (
                X_train_arr,
                y_train,
                X_test_arr,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info('Error occured in Initiate_Data_Transformation function',e)
            raise CustomException(e,sys)
        