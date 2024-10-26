from dataclasses import dataclass
import os,sys
import pandas as pd
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.exception import CustomException

@dataclass
class DataIngestionConfig():
    train_data_path :str=os.path.join('artifact','train.csv')
    test_data_path :str=os.path.join('artifact','test.csv')
    raw_data_path :str=os.path.join('artifact','data.csv')

class DataIngestion():
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        logging.info("entered the data ingestion methods or components")

    def initiate_data_ingestion(self):
        logging.info("data ingestion initiated")
        try:
            df=pd.read_csv('heart.csv')
            logging.info("read the dataset")
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)

            logging.info("Train test Split started")
            train_data,test_data=train_test_split(df,test_size=0.2,random_state=24)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("ingestion of the data completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()
