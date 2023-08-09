import os,sys
import pickle
from src.Exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
import pandas as pd

def save_obj(obj,file_path):
    try:    
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        logging.info('error occured in save object function in utils',e)
        raise CustomException(e,sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models:dict):
    try:
        
        report={}

        for i in range(len(models)):
            # creating model object
            model=list(models.values())[i]
            # training the model
            model.fit(X_train,y_train)
            # predicting 
            y_pred=model.predict(X_test)

            # evaluating the model
            R_square=r2_score(y_test,y_pred)

            report[list(models.keys())[i]]=R_square

            return report
    
    except Exception as e:
        logging.info('error occured in evaluate model function in utils',e)
        raise CustomException(e,sys)



def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info('exception occured in load_object function in utils',e)
        raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat=carat
        self.depth=depth
        self.table=table
        self.z=z
        self.cut=cut
        self.color=color
        self.clarity=clarity

    def get_data_as_dataframe(self):
            try:
                custom_input_data_dic={
                    'carat':[self.carat],
                    'depth':[self.depth],
                    'table':[self.table],
                    'z':[self.z],
                    'cut':[self.cut],
                    'color':[self.color],
                    'clarity':[self.clarity]
                }
                df=pd.DataFrame(custom_input_data_dic)
                logging.info('dataframe created')
                return df


            except Exception as e:
                logging.info('error occured in get_data_as_dataframe method in custom data class')
                raise CustomException(e,sys)