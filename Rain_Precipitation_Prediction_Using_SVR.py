# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import re
import os
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt
#import pmdarima as pmd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None

"""# Data Pre-Processing"""

dataframe = pd.read_excel('/content/drive/MyDrive/AdvanceML/dataset/04-10 February 2021.xlsx',sheet_name="Weekly ET",usecols='A:L')
#dataframe.columns = ['Location','Crop_Coefficient','Day_1','Day_2','Day_3','Day_4','Day_5','Day_6','Day_7','Total_ETo','Precipitation','Irrigation_Required']

dataframe

#clean the string
def clean(s):
  s = s.replace("\n","")
  s = re.sub(r"[!@#$%^&*()_\-\+=\[\]{};'\"/?.\\.,><:~`]|[0-9]","",s)
  s.lower()
  s=s.strip(' ')
  return s

#use to clean crop_name
def clean_2(s):
  s = s.replace(" ","")
  s=s.strip(' ')
  return s

def Pre_Processor(dataframe):
  dataframe.columns = ['Location','Crop_Coefficient','Day_1','Day_2','Day_3','Day_4','Day_5','Day_6','Day_7','Total_ETo','Precipitation','Irrigation_Required']
  #filling the missing values
  dataframe['Location']=dataframe['Location'].replace(to_replace =np.NaN,value ="x_x")
  temp=""
  index=0
  for row in dataframe['Location']:
    if row == "x_x":
      dataframe['Location'][index]=dataframe['Location'][index].replace("x_x",temp)
    else:
      temp=dataframe['Location'][index]
    index = index+1

  #creating new column
  dataframe['Crop_Name']='crop'

  #settingthe crop_Coefficient
  index=0
  for row in dataframe['Crop_Coefficient']:
    list_ = row.split('=')
    list_[0] #send to crop name
    dataframe['Crop_Name'][index]=dataframe['Crop_Name'][index].replace("crop",list_[0])
    list_[1] #send to Crop_Coefficient
    dataframe['Crop_Coefficient'][index]=dataframe['Crop_Coefficient'][index].replace(row,list_[1])
    
    list_=""#clear the list
    index=index+1
    
  #type_casting
  dataframe=dataframe.astype({'Crop_Coefficient': 'float64'})

  #cleaning location
  index=0
  for row in dataframe['Location']:
    clean_string = clean(row)
    dataframe['Location'][index]=dataframe['Location'][index].replace(row,clean_string)
    index=index+1

  #cleaning crop_name with clean
  index=0
  for row in dataframe['Crop_Name']:
    clean_string = clean(row)
    dataframe['Crop_Name'][index]=dataframe['Crop_Name'][index].replace(row,clean_string)
    index=index+1


  #cleaning crop_name with clean_2
  index=0
  for row in dataframe['Crop_Name']:
    clean_string = clean_2(row)
    dataframe['Crop_Name'][index]=dataframe['Crop_Name'][index].replace(row,clean_string)
    index=index+1

  #replacing value
  dataframe['Crop_Name']=dataframe['Crop_Name'].replace(to_replace ="MaizeHybrid",value ="Maize")
  dataframe['Location']=dataframe['Location'].replace(to_replace ="Islamabd",value ="Islamabad")

  return dataframe

#main aggregator
def Aggregator_(path): #Function to create single document of all files
  flag = True
  for subdir, dirs, files in os.walk(path):
      for filename in files:
          filepath=""
          filepath = subdir + os.sep + filename
          #print(filepath)
          if flag == True:
            dataframe = pd.read_excel(filepath,sheet_name="Weekly ET",usecols='A:L')
            mainData= Pre_Processor(dataframe)
            flag = False
          else:
            dataframe = pd.read_excel(filepath,sheet_name="Weekly ET",usecols='A:L')
            dataframe_cleaned = Pre_Processor(dataframe)
            mainData = pd.concat([mainData,dataframe_cleaned], axis=0)
  return mainData

mainData = Aggregator_('/content/drive/MyDrive/AdvanceML/dataset')

df = mainData.sort_values(by = 'Location')

df = mainData

df[103:120]

#converting zero values to NaN
df['Crop_Coefficient']=df['Crop_Coefficient'].replace(to_replace =0,value =np.NaN)
df['Day_1']=df['Day_1'].replace(to_replace =0,value =np.NaN)
df['Day_2']=df['Day_2'].replace(to_replace =0,value =np.NaN)
df['Day_3']=df['Day_3'].replace(to_replace =0,value =np.NaN)
df['Day_4']=df['Day_4'].replace(to_replace =0,value =np.NaN)
df['Day_5']=df['Day_5'].replace(to_replace =0,value =np.NaN)
df['Day_6']=df['Day_6'].replace(to_replace =0,value =np.NaN)
df['Day_7']=df['Day_7'].replace(to_replace =0,value =np.NaN)
df['Total_ETo']=df['Total_ETo'].replace(to_replace =0,value =np.NaN)
df['Precipitation']=df['Precipitation'].replace(to_replace =0,value =np.NaN)
df['Irrigation_Required']=df['Irrigation_Required'].replace(to_replace =0,value =np.NaN)

df.groupby(['Location']).sum()

df = df[df.Location !=  'Attock' ]
df = df[df.Location !=  'Chakwal' ]
df = df[df.Location !=  'Islamabad' ]
df = df[df.Location !=  'Jaffarabad' ]
df = df[df.Location !=  'Naseerabad' ]

df.to_csv('/content/drive/MyDrive/AdvanceML/main_cleaned.csv', index = False, header=True)

df = pd.read_csv('/content/drive/MyDrive/AdvanceML/main_cleaned.csv',sep = ',')

df.isnull().sum()



df

"""#Missing Values Imputation"""



df.isnull().sum()

df2 = df[['Location', 'Crop_Name']].copy()
df2

df = df.drop(columns=['Location', 'Crop_Name'])
df

imputer = KNNImputer(n_neighbors=5)
# fit on the dataset
imputer.fit(df)
# transform the dataset
trans = imputer.transform(df)

df = pd.DataFrame(trans)
df.columns = ['Crop_Coefficient','Day_1','Day_2','Day_3','Day_4','Day_5','Day_6','Day_7','Total_ETo','Precipitation','Irrigation_Required']

df

df2

df = pd.concat([df2,df], axis=1)
df

df.to_csv('/content/drive/MyDrive/AdvanceML/final_data.csv', index = False, header=True)

df.isnull().sum()

"""#SVR

Splitting the data for training and testing (80% 20%)
"""

df = pd.read_csv('/content/drive/MyDrive/AdvanceML/final_data.csv',sep = ',')

trainSamples=df.sample(frac=0.8,random_state=200) #random state is a seed value
testsamples=df.drop(trainSamples.index)

trainSamples

testsamples

df

X = trainSamples.iloc[:, 2:12].values
y = trainSamples.iloc[:, 12].values
#y=y.reshape(-1,1)

print(X)
type(X)

print(y)
type(y)

"""#Training"""

sc_X = StandardScaler()
sc_y = StandardScaler()
#X = sc_X.fit_transform(X)
#y = sc_y.fit_transform(y)
#y = y.ravel()

regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

t = [[0.31,	1.178000,	1.240000	,1.157333,	1.405333,	1.116000,	1.074667	,1.095333	,8.266667,	1.125		 ]]

y_pred = regressor.predict(t)

y_pred

"""***Testing***"""

t = testsamples.iloc[:, 2:12].values
y_pred = regressor.predict(t)

"""***Evaluation***

Correlation Cofficient
"""

y_true = testsamples.iloc[:, 12].values
math.sqrt(r2_score(y_true, y_pred))

"""MAPE"""

MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
MAPE

#plot the predictions for validation set
plt.plot(y_true, label='Valid')
plt.show()

plt.plot(y_true, label='Valid')
plt.plot(y_pred, label='Prediction')
plt.show()

"""# Auto-ARIMA

Splitting the data for training and testing (80% 20%)
"""

df = pd.read_csv('/content/drive/MyDrive/AdvanceML/final_data.csv',sep = ',')

trainSamples=df.sample(frac=0.8,random_state=200) #random state is a seed value
testsamples=df.drop(trainSamples.index)

"""There are two assumptions of Auto Arima
1.   The series should be univariate
2.   The series should be stationary 
"""

print(trainSamples.shape)
print(testsamples.shape)

"""Satisfying first assumtion: Making the series univariate"""

X = trainSamples['Irrigation_Required']
y = testsamples['Irrigation_Required']

print("Training Shape: ",X.shape)
print("Testing Shape: ",y.shape)

"""Satisfying the second assumption: Testing if the series is stationary or not"""

#We are performing the test that wether the series is stationary or not
#The null hypothesis of the test is that the time series is not stationary, while 
#the alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary.
print("p-value:", adfuller(X)[1])

"""The p-value is less than significance interval i-e 0.05, so we tested that the series is stationary

***Training Auto Arima***
"""

autoarima_model = pmd.auto_arima(X, start_p=1, start_q=1,test="adf",trace=True)

"""***Testing Auto Arima***"""

testsamples['ARIMA_prediction'] = autoarima_model.predict(len(testsamples.Irrigation_Required))

y_true, y_pred = np.array(testsamples.Irrigation_Required), np.array(testsamples.ARIMA_prediction)

"""Mean Absolute Percentage Error (MAPE)"""

MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
MAPE

"""Corelation Cofficient"""

y_pred

r2_score(y_true, y_pred)

"""Visualizing the results"""

#Plotting actual datapoints
plt.plot(y_true, label='Valid')
plt.show()

#plotting predicted datapoints
plt.plot(y_true, label='Valid')
plt.plot(y_pred, label='Prediction')
plt.show()