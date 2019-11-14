
#Diabetes Prediction Using Random Forest

import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#For training
def train():
    dataset = pd.read_csv('pima.csv')
    X = dataset[['Body_Mass_Index','Triceps_Skin_Fold_Thickness','Serum_Insulin','Plasma_Glucose_Concentration','Diastolic_Blood_Pressure']]
    Y = dataset[['Output']]
    
    #train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    nb_classifier = GaussianNB()
    nb = nb_classifier.fit(X_train, Y_train.values.ravel())
    
    #Save Model As Pickle File
    with open('nb.pkl','wb') as m:
        pickle.dump(nb,m)
    test(X_test,Y_test)

#Test accuracy of the model
def test(X_test,Y_test):
    with open('nb.pkl','rb') as mod:
        p=pickle.load(mod)
    
    pre=p.predict(X_test)
    print (confusion_matrix(Y_test,pre))
    print(classification_report(Y_test, pre))

def find_data_file(filename):
    if getattr(sys, "frozen", False):
        # The application is frozen.
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen.
        datadir = os.path.dirname(__file__)

    return os.path.join(datadir, filename)


def check_input(data) ->int :
    df=pd.DataFrame(data=data,index=[0])
    with open(find_data_file('rf.pkl'),'rb') as model:
        p=pickle.load(model)
    op=p.predict(df)
    return op[0]


if __name__=='__main__':
    train()    
