from dataclasses import dataclass
import os,sys
import pandas as pd
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.components.data_transformation import DataTranformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import Model_trainer

 

@dataclass
class DataIngestionConfig(): 
    train_data_path :str=os.path.join('artifact','train.csv')
    test_data_path :str=os.path.join('artifact','test.csv')
    raw_data_path :str=os.path.join('artifact','data.csv')
    #train_data_path, test_data_path, raw_data_path: These define where the raw, training, and test data files will be saved.
    #Files will be stored in a directory named artifact.

class DataIngestion():
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        #Initializes an instance of DataIngestionConfig as self.ingestion_config, storing the configuration settings for the data paths.
        logging.info("entered the data ingestion methods or components")

    def initiate_data_ingestion(self):
        logging.info("data ingestion initiated")
        try:
            df=pd.read_csv('heart.csv')
            logging.info("read the dataset")
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            #Create Directories: Ensures that the directory artifact exists to save data files, creating it if it doesn’t already exist.
            
            logging.info("Train test Split started")
            train_data,test_data=train_test_split(df,test_size=0.2,random_state=24)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            #save the entire dataset (df) to data.csv for a raw data backup.
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            #save train_data to train.csv.
            #test_data to test.csv. 
            #Each file is saved with headers and without an index.
            logging.info("ingestion of the data completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    model_train=Model_trainer()
    print(model_train.initiate_model_trainer(train_arr,test_arr))
