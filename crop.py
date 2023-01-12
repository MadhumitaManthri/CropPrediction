from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('crop_recommendation.csv')
features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

def decision_tree_model(N,P,K,temp, humidity, ph, rainfall):
    DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)
    DecisionTree.fit(Xtrain,Ytrain)
    data = np.array([[N,P,K,temp, humidity, ph, rainfall]])
    prediction = DecisionTree.predict(data)

    return prediction[0]

def logistic_regression_model(N,P,K,temp, humidity, ph, rainfall):
    LogReg = LogisticRegression(random_state=2)
    LogReg.fit(Xtrain,Ytrain)
    data = np.array([[N,P,K,temp, humidity, ph, rainfall]])
    prediction = LogReg.predict(data)

    return prediction[0]



N = int(input("Enter N: "))
P = int(input("Enter P: "))
K = int(input("Enter K: "))
temp = int(input("Enter temperature: "))
humidity = int(input("Enter humidity: "))
ph = int(input("Enter ph: "))
rainfall = int(input("Enter rainfall in mm: "))

logistic_regression_model_predicted_crop = logistic_regression_model(N,P,K,temp, humidity, ph, rainfall)
decision_tree_model_predicted_crop = decision_tree_model(N,P,K,temp, humidity, ph, rainfall)

if logistic_regression_model_predicted_crop == decision_tree_model_predicted_crop:
    final_crop = logistic_regression_model_predicted_crop
else:
    final_crop = logistic_regression_model_predicted_crop + " or " + decision_tree_model_predicted_crop

print("Under the given soil conditions and weather, " + final_crop + " would probably be the ideal crop(s).")
