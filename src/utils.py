import os,sys,dill
import pandas as pd
import numpy as np
from src.exception import CustomException
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix


def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(Xtrain,ytrain,Xtest,ytest,models):
    try:
        acc_dict={}
        model_list = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        for model_name, model in models.items():

            model.fit(Xtrain, ytrain)
            y_pred_train = model.predict(Xtrain)
            y_pred = model.predict(Xtest)

            test_accuracy = accuracy_score(ytest, y_pred)
            precision = precision_score(ytest, y_pred)
            recall = recall_score(ytest, y_pred)
            f1 = f1_score(ytest, y_pred)
            confusion_mat = confusion_matrix(ytest, y_pred)

            '''print(f"Model: {model_name}")
            print("Testing Accuracy: ", test_accuracy)
            print("Precision: ",precision)
            print("Recall: ",recall)
            print("F1 Score: ",f1)
            print("Confusion Matrix:\n ",confusion_mat)'''


            model_list.append(model_name)
            accuracy_list.append(test_accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            
            

            #print("=" * 35)
            #print('\n')

            
            for i in range(len(model_list)):
                acc_dict.update({model_list[i]:accuracy_list[i]}) 

        return acc_dict
    except:
        pass