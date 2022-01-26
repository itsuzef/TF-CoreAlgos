"""""
Clustering is a Machine Learning technique that involves the grouping of data points. 
In theory, data points that are in the same group should have similar properties and/or features, while data points in different groups should have highly dissimilar properties and/or features. 
Unfortunalty there are issues with the current version of TensorFlow and the implementation for KMeans. This means we cannot use KMeans without writing the algorithm from scratch. 
Basic Algorithm for K-Means:
Step 1: Randomly pick K points to place K centroids
Step 2: Assign all the data points to the centroids by distance. The closest centroid to a point is the one it is assigned to.
Step 3: Average all the points belonging to each centroid to find the middle of those clusters (center of mass). Place the corresponding centroids into that position.
Step 4: Reassign every point once again to the closest centroid.
Step 5: Repeat steps 3-4 until no point changes which centroid it belongs to.
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_probability as tfp 
import pandas as pd


tfd = tfp.distributions  # making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])  # Refer to point 2 above
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                 [0.2, 0.8]])  # refer to points 3 and 4 above
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # refer to point 5 above

# the loc argument represents the mean and the scale is the standard devitation
