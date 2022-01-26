"""""
a convolutional neural network(CNN, or ConvNet) is a class of artificial neural network, 
most commonly applied to analyze visual imagery.
They are also known as shift invariant or space invariant artificial neural networks (SIANN), 
based on the shared-weight architecture of the convolution kernels or filters that slide along input features and 
provide translation equivariant responses known as feature maps.Counter-intuitively, most convolutional neural networks are only equivariant, 
as opposed to invariant, to translation.
They have applications in image and video recognition, recommender systems, image classification, image segmentation, 
medical image analysis, natural language processing, brain-computer interfaces,and financial time series 

Dense neural networks analyze input on a global scale and recognize patterns in specific areas. 
Convolutional neural networks scan through the entire input a little at a time and learn local patterns.

It outputs a response map, quantifying the presence of the filter's pattern at different locations 

Three main properties of each convolutional layer: 
Input size
the number of filters
the sample size of the filters.
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
 
#load cifar10 dataset 
#split into testing and training
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
#normalize pixel values to be between 0 and 1
train_images, test_images =train_images / 255.0 , test_images / 255.0

class_names = ['airplane','automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck']

#looking at images, change the value 1 
IMG_INDEX = 0
plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()

#CNN Architecture 
# a stack of Conv2D and MaxPooling2D layers followed by a densly connected layer. 
# Conv2D and MaxPooling2D layers extract the features from the image 
# These features are then flattened and fed to densly connected layers that determine the class of an image based on the presence of features. 

#building the Convolutional base 

model = models.Sequential()

#define amount of filters, size of filters, the activation function, the input shape 
model.add(layers.Conv2D(32, (3, 3), activation= 'relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

# to take a look at the model 
model.summary()

#Adding Dense layers to classify the extracted features
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10))

model.summary()

# Training using the recommended hyper parameters from tensorflow
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

#evaluating the model

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)