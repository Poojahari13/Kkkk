import numpy as np

import pandas as pd

dataFrame=pd.read_csv('/content/Heart.csv')

dataFrame.shape # shape

dataFrame.info() 

dataFrame.head() 

dataFrame.dtypes # datatype 

dataFrame.isnull() # missing values

dataFrame.isnull().sum() # missing values : count

dataFrame.Age.mean() #mean of age

count = (dataFrame['Age'] == 0).sum() # counting zeros

count

# selected columns

var=dataFrame.loc[:,['Age','Sex','ChestPain','RestBP','Chol']] 

var

# Splitting the dataset into train and test sets: 75-25 split

from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(var, test_size = 0.25, random_state = 42)

X_train.shape, X_test.shape
tp=45

fp=55

fn=05

tn=395

acc=(tp+tn)/(tp+fp+fn+tn)

pre=tp/(tp+fp)

rec=tp/(tp+fn)

print("Accuracy is : {}".format(acc))

print("Precision is : {}".format(pre))

print("Recall is : {}".format(rec))

print("F1-Score is : {}".format((2*pre*rec)/(pre+rec)))
