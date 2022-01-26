"""""
Creating a neural network using Keras. This model is testing the image classification
accuracy. 
It asks the user to input a number (1 to 1000), it shows the guess and the expected classification


"""

from re import L
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#load fashion_mnist dataset 
fashion_mnist = keras.datasets.fashion_mnist 

#split into testing and training
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images.shape
train_images[0,23,23]
train_labels[:10]
class_names = ['T-shirt/top','Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker','Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_images[5])
plt.colorbar()
plt.grid(False)
plt.show()

#Dats processing 

train_images = train_images / 255.0 

test_images = test_images / 255.0

#Building the model 
model = keras.Sequential([
    #input layer 
    keras.layers.Flatten(input_shape=(28,28)), 
    #hidden layer 
    keras.layers.Dense(128, activation = 'relu'), 
    #output layer 
    keras.layers.Dense(10, activation = 'softmax')
])

#Compiling the model 

model.compile(optimizer = 'adam', 
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['accuracy'])

#Train the model 
model.fit(train_images, train_labels, epochs = 3)

#evaluate/test the model 

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 1)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[3])])
test_images.shape
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model,image,correct_label):
    class_names = ['T-shirt/top','Trouser', 'Pullover', 'Dress', 'Coat', 
                    'Sandal', 'Shirt', 'Sneaker','Bag', 'Ankle boot']
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]

    show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Expected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def get_number(): 
    while True: 
        num = input("Pick a number: ")
        if num.isdigit(): 
            num = int(num)
            if 0 <= num <= 1000: 
                return int(num)
        else: 
            print("try again.. ")
num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
