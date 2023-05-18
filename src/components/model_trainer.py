import os
import sys
from dataclasses import dataclass

from src.Exception import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

from src.utils import save_obj
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    Trained_model_file_path=os.path.join('Artifact','ModelTrainer','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.trainingconfig=ModelTrainerConfig()

    def Initiate_model_training(self,X_train_array,y_train,X_test_array,y_test):
        try:
            logging.info('creating dict of models')
            models={
                'LinearRegression':LinearRegression(),
                'RidgeRegression':Ridge(),
                'LassoRegression':Lasso(),
                'ElasticNet':ElasticNet()
                }
            
            logging.info('evaluating model and saving the report')
            Model_report:dict=evaluate_model(X_train_array,y_train,X_test_array,y_test,models)
            logging.info(f'Model report:{Model_report}')

            best_model_score=max(Model_report.values())
            logging.info(f'finding the best model score \n best model score:{best_model_score}')

            best_model_name=list(Model_report.keys())[list(Model_report.values()).index(best_model_score)]
            logging.info(f'finding the best model name \n best model name:{best_model_name}')

            best_model=models[best_model_name]
            logging.info(f'Best model : {best_model}')

            os.makedirs(os.path.dirname(
                self.trainingconfig.Trained_model_file_path),
                exist_ok=True)

            logging.info('saving the model object')
            save_obj(obj=best_model,
                    file_path=self.trainingconfig.Trained_model_file_path
                    )
            

        except Exception as e:
            logging.info('error occured in Initiate_model_training function',e)
            raise CustomException(e,sys)



