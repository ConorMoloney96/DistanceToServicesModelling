# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 21:40:33 2020

@author: User
"""

import pandas as pd
import chardet
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import linear_model
from sklearn.preprocessing import Binarizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import chardet
#from sknn.mlp import Regressor, Layer
#import sknn.mlp

#Performs data cleaning i.e. removes columns with too many missing values and fills in other missing values
def clean_data(df):
    print(df.isnull().sum())
    #Drop commas from numbers with thousands so they can be converted to ints
    df = df.replace(',','', regex=True)
    #Fill NaN values with mean from the column
    df['Travel time (public transport) to a GP premises \n(minutes)'].fillna((df['Travel time (public transport) to a GP premises \n(minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to a dentist \n(minutes)'].fillna((df['Travel time (public transport) to a dentist \n(minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to a pharmacist \n(minutes)'].fillna((df['Travel time (public transport) to a pharmacist \n(minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to an optician \n(minutes)'].fillna((df['Travel time (public transport) to an optician \n(minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to a Job Centre or Jobs and Benefits Office (minutes)'].fillna((df['Travel time (public transport) to a Job Centre or Jobs and Benefits Office (minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to a Post Office \n(minutes)'].fillna((df['Travel time (public transport) to a Post Office \n(minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to a supermarket / food store \n(minutes)'].fillna((df['Travel time (public transport) to a supermarket / food store \n(minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to a large service centre (minutes)'].fillna((df['Travel time (public transport) to a large service centre (minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to a library \n(minutes)'].fillna((df['Travel time (public transport) to a library \n(minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to a Council Leisure Centre or sports facilities \n(minutes)'].fillna((df['Travel time (public transport) to a Council Leisure Centre or sports facilities \n(minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to financial services (minutes)'].fillna((df['Travel time (public transport) to financial services (minutes)'].mean()), inplace=True)
    df.drop(columns= ['Travel time (public transport) to a day nursery or crŠche (minutes)', 'Travel time (public transport) to a restaurant  \n(minutes)', 'Travel time (public transport) to a \nfastfood outlet\n(minutes)', 'Travel time (public transport) to a pub (minutes)', 'Travel time (public transport) to a health & beauty\nestablishment (minutes)'], axis = 1, inplace=True)
    #Convert strings representing categories to ints
    df['SA'] = pd.factorize(df['SA'])[0].astype(np.uint16)
    df['SA Code'] = pd.factorize(df['SA Code'])[0].astype(np.uint16)
    return df
                                                        
                                                                                        
                                                           
#Evaluate the classification problems using accuracy, precision and recall metrics
def test_classification(model,X_train, X_test, y_train, y_test):
    print(X_train.shape)
    print(y_train.shape)
    print("Test Model accuracy: ")
    print(model.score(X_test, y_test))
    test_predictions = model.predict(X_test)
    #TP/TP+FN
    #Answers question: Of all cases of churn how many did we identify
    micro_recall = metrics.recall_score(y_test, test_predictions)
    print("Micro averaged recall score: {0:0.4f}".format(micro_recall) )
    #Precision = True positives/All positives
    #High precision indicates a low false positive rate
    micro_precision = metrics.precision_score(y_test, test_predictions)
    print("Micro averaged precision score: {0:0.4f}".format(micro_precision) )

#Evaluates how effective a model is at predicting the outcome
#Uses MAE, MSE and R^2 metrics
def test_model(regressor, X_train, X_test, y_train, y_test):
    #print(y_train.describe())
# =============================================================================
#     acc_score = regressor.score(X_test, y_test)
#     print("R^2 score")
#     print(acc_score)
#     y_pred = regressor.predict(X_test)
#     print("Mean absolute Error: ")
#     print(mean_absolute_error(y_pred, y_test))
#     print("Mean Square Error: ")
#     print(mean_squared_error(y_pred, y_test))
#     print(" ")
# =============================================================================
    y_train_pred = regressor.predict(X_train)
    print("Train Mean absolute Error: ")
    print(mean_absolute_error(y_train, y_train_pred))
    print("Train Mean Square Error: ")
    print(mean_squared_error(y_train, y_train_pred))
    acc_score = regressor.score(X_test, y_test)
    print("R^2 score")
    print(acc_score)
    y_pred = regressor.predict(X_test)
    print("Mean absolute Error: ")
    print(mean_absolute_error(y_test, y_pred))
    print("Mean Square Error: ")
    print(mean_squared_error(y_test, y_pred))
    print(" ")
    
#Converts problem from regression to classification so that different svm regularization techniques can be used
def convert_to_classification(X_train, X_test, y_train, y_test, threshold, num_features):
    #Discretize y values i.e. convert the target variable from a continuous value into categorical
    #Median target variable value used as threshold
    transformer = Binarizer(threshold)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    X_test = X_test.reshape(-1, num_features)
    X_train = X_train.reshape(-1, num_features)
    y_train_discretized = transformer.fit_transform(y_train)
    y_test_discretized = transformer.fit_transform(y_test)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train_discretized.shape)
    print(y_test_discretized.shape)
    return X_train, X_test, y_train_discretized, y_test_discretized

#Creates and fits an SVM model
def support_vector_machine(X_train, y_train):
    regressor = SVR(kernel='sigmoid', C=5, max_iter = 5000, verbose=True)
    regressor.fit(X_train, y_train)
    print("SVR Coefficients")
    #print(regressor.coef_)
    scores = cross_val_score(regressor, X_train, y_train, cv = 10)
    print("Cross validation score: %0.4f" % scores.mean())
    
    return regressor
 
#Creates and fits a linear regression model
def lin_regression(X_train, y_train):
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    print("LR Coefficients")
    print(lr.coef_)
    scores = cross_val_score(lr, X_train, y_train, cv = 10)
    print("Cross validation score: %0.4f" % scores.mean())
    return lr

#L1 regularization
def ridge_regression(X_train, y_train):
    ridge = linear_model.Ridge()
    ridge.fit(X_train, y_train)
    print("Ridge Coefficients")
    print(ridge.coef_)
    scores = cross_val_score(ridge, X_train, y_train, cv = 10)
    print("Cross validation score: %0.4f" % scores.mean())
    return ridge

#L2 Regularization
def lasso_regression(X_train, y_train):
    lasso = linear_model.Lasso()
    lasso.fit(X_train, y_train)
    print("Lasso Coefficients")
    print(lasso.coef_)
    scores = cross_val_score(lasso, X_train, y_train, cv = 10)
    print("Cross validation score: %0.4f" % scores.mean())
    return lasso

#Creates and fits an elastic net model which combines L1 and L2 regularization
def elastic_net_regression(X_train, y_train):
    en = linear_model.ElasticNet()
    en.fit(X_train, y_train)
    print("Elastic Net Coefficients")
    print(en.coef_)
    scores = cross_val_score(en, X_train, y_train, cv = 10)
    print("Cross validation score: %0.4f" % scores.mean())
    return en

#Creates and fits a Support Vector Classifier using L1 regularization
def svc_l1(X_train, y_train):
    lin_svc = LinearSVC(penalty = 'l1', dual=False, max_iter=7000)
    lin_svc.fit(X_train, y_train)
    return lin_svc

#Creates and fits a Support Vector Classifier using L2 regularization
def svc_l2(X_train, y_train):
    lin_svc = LinearSVC(penalty = 'l2', max_iter=7000)
    lin_svc.fit(X_train, y_train)
    return lin_svc

#Creates and fits a Multi Layer Perceptron for regression
def mlp_reg(X_train, y_train):
    mlp = MLPRegressor(random_state=1, max_iter=750, alpha = 0.1)
    mlp.fit(X_train, y_train)
    scores = cross_val_score(mlp, X_train, y_train, cv = 10)
    print("Cross validation score: %0.4f" % scores.mean())
    return mlp


#Performs predictive analysis on the distance to services dataset
def distance_to_services_analysis():
    #The following values will be converted to NaN when read in
    missing_values = ["n/a", "na", "--", r"\\N", r"\N", r"N"]
    #Obtains the encoding for the csv file being read in
    with open("DistanceToServices.csv", "rb") as f:
        result = chardet.detect(f.read())
    df = pd.read_csv("DistanceToServices.csv", encoding=result['encoding'], na_values=missing_values)
    df = clean_data(df)
    #sets the target variable y
    y = df['Travel time (public transport) to a GP premises \n(minutes)']
    X = df.drop(columns = ['Travel time (public transport) to a GP premises \n(minutes)'], axis=1)
    X_train, X_test, y_train, y_test =  train_test_split(X,y, test_size=0.2)
     
    #Begins modelling
    print("SVR")
    regressor = support_vector_machine(X_train, y_train)
    test_model(regressor,X_train, X_test, y_train, y_test )
    print("Linear Regression")
    lr = lin_regression(X_train, y_train)
    test_model(lr,X_train, X_test, y_train, y_test )
    print("Ridge Regression")
    ridge = ridge_regression(X_train, y_train)
    test_model(ridge,X_train, X_test, y_train, y_test )
    print("Lasso Regression")
    lasso = lasso_regression(X_train, y_train)
    test_model(lasso,X_train, X_test, y_train, y_test )
    print("Elastic Net Regression")
    en = elastic_net_regression(X_train, y_train)
    test_model(en,X_train, X_test, y_train, y_test )
    print("MLP")
    mlp = mlp_reg(X_train, y_train)
    test_model(mlp,X_train, X_test, y_train, y_test )

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.values
    y_test = y_test.values
    print((X_train.shape))
    
    
# =============================================================================
#     #Convert the regression problem to a classification problem so that SVM can be done with l1 and L2 regularization
#     X_train, X_test, y_train_classification, y_test_classification = convert_to_classification(X_train, X_test, y_train, y_test, 10, 51)
#     
#     #Create and test a Support Vector Classifier with L1 regularization
#     svc = svc_l1(X_train, y_train_classification)
#     test_classification(svc,X_train, X_test, y_train_classification, y_test_classification )
#     
#     #Create and test a Support Vector Classifier with L1 regularization
#     svc_2 = svc_l2(X_train, y_train_classification)
#     test_classification(svc_2,X_train, X_test, y_train_classification, y_test_classification )
# =============================================================================
    

#Driver class 
#Can choose to run analysis on either (or both) the boston housing price and services distances datasets
def main():
    distance_to_services_analysis()
    
    
if __name__ == "__main__":
    main()