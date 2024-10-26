import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd 
from src.utils import save_obj
from src.logger import logging
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import os

@dataclass
class DataTranformationConfig():
    preprocessor_obj_path=os.path.join('artifact','preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.Data_Tranformationconfig=DataTranformationConfig()
    
    def get_dataTransformer_obj(self):#responsible for data transformation
        try:
            logging.info("data trasformation started")

            num_features=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']
            #cat_feature=[]



            #Creating numerical Pipeline and categorical pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='mean')),#if any missing values then replace it with mean if no outliers if not then replace it with median
                    ('scaler',StandardScaler())
                    ],

            #cat_pipeline=Pipeline(
                #steps=[
                    #('imputer',SimpleImputer(strategy='most_frequent')),
                    #('encoding',OneHotEncoder()),
                    #('scaler',StandardScaler())])
                    # here there are no categorical features so not using it
            )
            logging.info("scaling num column through pipeline completed")




            # combining the Categoricatl and numerical pipeline using columnTransformer
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_features)
                    #('cat_pipeline',cat_pipeline,cat_features)

                ]
            )
            logging.info("combining pipeline complete completed")


            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("initiating data transformation")

            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            preprocessing_obj=self.get_dataTransformer_obj()

            target_col='target'
            numerical_col=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']

            input_feature_train_df=train_df.drop(columns=[target_col],axis=1)
            target_feature_train_df=train_df[target_col]
            
            input_feature_test_df=test_df.drop(columns=[target_col],axis=1)
            target_feature_test_df=test_df[target_col]

            input_feature_train_array=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_array,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_array,np.array(target_feature_test_df)
            ]

            save_obj(
                file_path=self.Data_Tranformationconfig.preprocessor_obj_path,
                obj=preprocessing_obj
            )
            logging.info("saved obj transformation done")
            logging.info("returned data")



            return(
                train_arr,
                test_arr,
                self.Data_Tranformationconfig.preprocessor_obj_path,
            )
        except Exception as e:
            raise CustomException(e,sys)