from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output
from six.moves import urllib
import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc


#Some basic matplot functions
""" x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]
plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.xlabel('Years')
plt.ylabel('Price')
plt.show()
 """

dftrain = pd.read_csv('/Users/youssefhemimy/Development /Python /TF-CoreAlgos/train.csv') # training data
dfeval = pd.read_csv('/Users/youssefhemimy/Development /Python /TF-CoreAlgos/eval.csv') # testing data
print(dftrain.head())
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

dftrain.describe() #statistic details about the data 

print("data shape is", dftrain.shape)  # shows the shape(rows, columns)

#dftrain.age.hist(bins=20) # plot a histogram 

#dftrain.sex.value_counts().plot(kind = 'barh') 

print("evaldata shape is" , dfeval.shape)

#Create categorical columns and numerical columns

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 
                        'embark_town', 'alone']

NUMERICAL_COLUMNS = ['age','fare']

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    #gets a list of all unique values from give feature column 
    vocalbulary = dftrain[feature_name].unique() 
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocalbulary))

for feature_name in NUMERICAL_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

#https://www.tensorflow.org/tutorials/estimator/linear  DOC
#create input function to convert our current pandas dataframe into tf.data.Dataset object
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

#Creating the model 
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn) #train
#result = linear_est.evaluate(eval_input_fn) #get model metrics/stats by testing on testing data
result = list(linear_est.predict(eval_input_fn)) 
clear_output()
#print("Accuracy is", result[0]) # the result variable is stats about the model

