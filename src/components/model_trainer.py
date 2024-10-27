from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import os,sys
from src.logger import logging
from src.utils import save_obj
from dataclasses import dataclass
from src.utils import evaluate_model


@dataclass
class ModelTrainerConfig():
    trained_model_file_path=os.path.join('artifact','model.pkl')

class Model_trainer():
    def __init__(self):
        self.Model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("spliting traing and testinput data")
            Xtrain,ytrain,Xtest,ytest=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                    'Logistic Regression':LogisticRegression(),
                    'Naive Bayes':GaussianNB(),
                    'AdaBoost':AdaBoostClassifier(),
                    'Gradient Boosting':GradientBoostingClassifier(),
                    'KNN':KNeighborsClassifier(),
                    'Random Forest Classfier':RandomForestClassifier(n_estimators=20, random_state=12,max_depth=5),
                    'XG Boost':XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
                                    reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5),
                    'K Nearest Neighbors':KNeighborsClassifier(n_neighbors=10),
                    'Decision Tree':DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6),
                    'Support Vector Machine':SVC(kernel='rbf', C=2)
                    }
            
            model_report :dict=evaluate_model(Xtrain=Xtrain,ytrain=ytrain,Xtest=Xtest,ytest=ytest,models=models)
            logging.info("evaluate model called and returned the model report")


            best_model_acc=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_acc)]

            logging.info("best model with accuracy found")

            save_obj(
                file_path=self.Model_trainer_config.trained_model_file_path,
                obj=best_model_name
            )

            return(
                best_model_name,
                best_model_acc
            )


            
            
        except  Exception as e:
            raise CustomException(e,sys)


