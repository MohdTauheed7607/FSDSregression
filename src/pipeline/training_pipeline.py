from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=='__main__':
    ingestion_obj=DataIngestion()
    transformation_obj=DataTransformation()
    trainer_obj=ModelTrainer()

    train_path,test_path=ingestion_obj.InitiateDataIngestion()
    X_train_arr,y_train,X_test_arr,y_test,_=transformation_obj.Initiate_data_transformation(train_path,test_path)
    trainer_obj.Initiate_model_training(X_train_arr,y_train,X_test_arr,y_test)

