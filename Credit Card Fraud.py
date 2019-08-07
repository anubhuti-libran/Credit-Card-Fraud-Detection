# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:42:47 2019

@author: Anubhuti Singh
"""
#importing packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#reading the csv file
bnk_prt_df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

#displaying first ten columns of the csv file
bnk_prt_df.head(10)

#looking into the details of our dataset
bnk_prt_df.describe()

#analysis of dataset for null values using heatmap
sns.heatmap(bnk_prt_df.isnull())
# Clearly; Heatmap of isnull indicates that this dataset does not have any null values

#analysing the features of our dataset
#analysing of 'isFraud' True or False
sns.countplot(x='isFraud',data=bnk_prt_df)
print(bnk_prt_df['isFraud'].value_counts())
#positive fraud data is very minimal in this dataset

#analysis of 'type' vs 'isFraud' in the dataset
sns.countplot(x='type',data=bnk_prt_df[bnk_prt_df['isFraud']==1],hue='isFraud')
print(bnk_prt_df['type'].value_counts())
print(bnk_prt_df[bnk_prt_df['isFraud']==1]['type'].unique())
#analyzing only the positive fraud dataset, it seems like they belong only to the datatype - Transfer, Cash_out and other types aren't contributing much to positive fraud data

#analysis of 'isFlaggedFraud' Vs 'isFraud' in the dataset
sns.countplot(x='isFlaggedFraud',data=bnk_prt_df,hue='isFraud')

bnk_prt_df[bnk_prt_df['isFlaggedFraud']==1]['isFraud'].value_counts()
#16 data which are identified as true isFlaggedFraud are found as Fraud

bnk_prt_df[bnk_prt_df['isFlaggedFraud']==0]['isFraud'].value_counts()
#8197 data out of 63.62 lakhs data which are identified with true isFlaggedFraud are found to be Fraud

bnk_prt_df[np.logical_and(bnk_prt_df['isFlaggedFraud']==0, bnk_prt_df['isFraud']==1)]['type'].value_counts()
#among 8197 data, there are almost equal splits for Transfer and Cashout types

#we can also perform data analysis to find the type of transactions occuring the most
var = bnk_prt_df.groupby('type').amount.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
var.plot(kind='bar')
ax1.set_title("Total amount per transaction type")
ax1.set_xlabel('Type of Transaction')
ax1.set_ylabel('Amount');
#hence; 'CASH OUT' and 'TRANSFER' are two most used mode of transaction and we can see that TRANSFER and CASH_OUT are also the only way in which fraud happen

#data analysis
fraud = bnk_prt_df.loc[bnk_prt_df.isFraud == 1]
nonfraud = bnk_prt_df.loc[bnk_prt_df.isFraud == 0]
piedata = fraud.groupby(['isFlaggedFraud']).sum()
f, axes = plt.subplots(1,1, figsize=(6,6))
axes.set_title("% of fraud transaction detected")
piedata.plot(kind='pie',y='isFraud',ax=axes, fontsize=14,shadow=False,autopct='%1.1f%%');
axes.set_ylabel('');
plt.legend(loc='upper left',labels=['Not Detected','Detected'])
plt.show()

fig = plt.figure()
axes = fig.add_subplot(1,1,1)
axes.set_title("Fraud transaction which are Flagged Correctly")
axes.scatter(nonfraud['amount'],nonfraud['isFlaggedFraud'],c='g')
axes.scatter(fraud['amount'],fraud['isFlaggedFraud'],c='r')
plt.legend(loc='upper right',labels=['Not Flagged','Flagged'])
plt.show()
#analysis of 'step' vs 'isFraud' in the dataset
sns.countplot(x='step',data=bnk_prt_df)
bnk_prt_df['step'].value_counts()
#The plot above clearly shows the need for a system which can be fast and reliable to mark the transaction which is fraud
#he current system is letting fraud transaction able to pass through a system which is not labeling them as a fraud

#analysis of 'amount' vs 'isFraud'
sns.jointplot('amount','isFraud',data=bnk_prt_df)
#Fraudulent transactions took place when the amount were less

#analysis of 'oldbalanceOrg' vs 'isFraud'
sns.jointplot(x='oldbalanceOrg',y='isFraud',data=bnk_prt_df)
#fraudulent transactions are more when oldBalanceOrg higher than when it is less

#analysis of 'newBalanceOrig' vs 'isFraud'
sns.jointplot(x='newbalanceOrig',y='isFraud',data=bnk_prt_df)
#fraudulent transactions are more when newBalanceOrg is higher than when it is less

#analysis of 'oldbalanceDest'/'newbalanceDest'  Vs 'isFraud' in the dataset
sns.jointplot(x='oldbalanceDest',y='isFraud',data=bnk_prt_df)
sns.jointplot(x='newbalanceDest',y='isFraud',data=bnk_prt_df)
#fraudulent transactions are more when oldbalanceDest/newbalanceDest is higher than when it is less

#performing some data exploration to see the relations between the features
#1
fraud = bnk_prt_df.loc[bnk_prt_df.isFraud == 1]
nonfraud = bnk_prt_df.loc[bnk_prt_df.isFraud == 0]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(nonfraud['oldbalanceOrg'],nonfraud['amount'],c='g')
ax.scatter(fraud['oldbalanceOrg'],fraud['amount'],c='r')
plt.show()

#2
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['oldbalanceOrg'])
ax.scatter(fraud['step'],fraud['oldbalanceDest'])
plt.show()

#3
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['newbalanceOrig'])
ax.scatter(fraud['step'],fraud['newbalanceDest'])
plt.show()

#4
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['newbalanceDest'])
ax.scatter(fraud['step'],fraud['oldbalanceDest'])
plt.show()

#5
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['oldbalanceOrg'])
ax.scatter(fraud['step'],fraud['newbalanceOrg'])
plt.show()

#multivariate analysis
bnk_prt_df.corr()
#from correlation matrix values,it can be concluded that 'isFraud' is not related to any variables here

#building the model
#creating dummy variables
bnk_prt_df['CASH_IN'] = pd.get_dummies(bnk_prt_df['type'])['CASH_IN']
bnk_prt_df['CASH_OUT'] = pd.get_dummies(bnk_prt_df['type'])['CASH_OUT']
bnk_prt_df['DEBIT'] = pd.get_dummies(bnk_prt_df['type'])['DEBIT']
bnk_prt_df['PAYMENT'] = pd.get_dummies(bnk_prt_df['type'])['PAYMENT']
bnk_prt_df['TRANSFER'] = pd.get_dummies(bnk_prt_df['type'])['TRANSFER']
bnk_prt_df.drop('CASH_IN',axis=True,inplace=True)
bnk_prt_df.info()

#putting independent and dependent features
X = bnk_prt_df[['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFlaggedFraud','CASH_OUT','DEBIT','PAYMENT','TRANSFER']]
Y = bnk_prt_df['isFraud']

#importing packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#building test train split model
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=1)

lr = LogisticRegression()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()

#building a function for various models
def trainFitTest(model):
   
    score = []
    model.fit(X_train,Y_train)
    print(' score - ',model.score(X_test,Y_test))
    score.append(model.score(X_test,Y_test))
    Y_rfc_pred = model.predict(X_test)
    print(classification_report(Y_test,Y_rfc_pred))
    print(confusion_matrix(Y_test,Y_rfc_pred))

#fitting different models

#logistic regression
trainFitTest(lr)
    
#Decision Tree Classifier
trainFitTest(dtc)

#Random Forest Classifier
trainFitTest(rfc)

#conclusion: 
##1.Machine learning can be used for the detection of fraud transaction.
##2.Predictive models produce good precision score and are capable of detection of fraud transaction.





