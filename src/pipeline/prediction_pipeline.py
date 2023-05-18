import os
import sys

from src.Exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig


class PredictionPipeline:
    def __init__(self):
        self.datatransformationconfig=DataTransformationConfig()
        self.modeltrainerconfig=ModelTrainerConfig()

    def Predict(self,features):
        try:
            preprocessor_file_path=self.datatransformationconfig.preprocessor_obj_file_path
            model_file_path=self.modeltrainerconfig.Trained_model_file_path
            
            logging.info('loading preprocessor object')
            preprocessor_obj=load_object(file_path=preprocessor_file_path)

            logging.info('loading trained model')
            model=load_object(file_path=model_file_path)

            logging.info('transforming data by using preprocessor')
            transformed_data=preprocessor_obj.transform(features)

            logging.info('predicting about data')
            prediction=model.predict(transformed_data)

            return prediction
        
        except Exception as e:
            logging.info('error occured in predict method in prediction pipeline',e)
            raise CustomException(e,sys)