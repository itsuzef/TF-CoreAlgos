from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output
from six.moves import urllib
import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc


#Some basic matplot functions
x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]
plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.xlabel('Years')
plt.ylabel('Price')
plt.show()


dftrain = pd.read_csv('/Users/youssefhemimy/Development /Python /TF-CoreAlgos/train.csv') # training data
dfeval = pd.read_csv('/Users/youssefhemimy/Development /Python /TF-CoreAlgos/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

dftrain.head()
