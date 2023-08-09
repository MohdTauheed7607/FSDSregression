import os, sys
import pandas as pd
from src.logger import logging
from src.Exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


## Initializing the data ingestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('Artifact','DataIngestion','train.csv')
    test_data_path:str=os.path.join('Artifact','DataIngestion','test.csv')
    raw_data_path:str=os.path.join('Artifact','DataIngestion','raw.csv')


## creating a class for data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def InitiateDataIngestion(self):
        try:
            logging.info('Data ingestion has been started')

            df=pd.read_csv("https://raw.githubusercontent.com/krishnaik06/FSDSRegression/main/notebooks/data/gemstone.csv")

            logging.info('Data has been ingested successfully')
            logging.info('Creating the raw data directory')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            logging.info('Raw data directory has been created')
            logging.info('Storing raw data into raw data directory')

            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info('Spliting the data into train and test')

            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            logging.info('Data has been splitted')

            logging.info('dropping x and y columns from train and test sets')
            train_set=train_set.drop(columns=['x','y'],axis=1)

            test_set=test_set.drop(columns=['x','y'],axis=1)

            logging.info('Storing the train and test data into train and test directory respctively')

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Error occured in Initiate data ingestion function',e)
            raise CustomException(e,sys)
        




