import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#load data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#define class
class_names = ['T-shirt/top', 'Touser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#loaded image test
#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)

#change image to grayscale
train_images = train_images / 255.0
test_images = test_images / 255.0

#image labeling test
#plt.figure(figsize = (10, 10))
#for i in range(25) :
#    plt.subplot(5, 5, i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap = plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])

#setup neural net model
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)), #transform 2d-array(28*28) to 1d-array(28*28 = 784)
    keras.layers.Dense(128, activation=tf.nn.relu), #128unit, Denslayer = Full connection layer = Fully connected layer
    keras.layers.Dense(10, activation=tf.nn.softmax) #softmax function = probability of classification
    ])

#compile configuration define training model
#parameter = loss function / optimizer / metrics (monitoring property)
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#training model
model.fit(train_images, train_labels, epochs = 5) #x_data, y_data, epochs_size

#evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy: ' , test_acc)

#make predictions
predictions = model.predict(test_images)

print('prediction[0] array = ' , predictions[0] , ', argmax(predictions[0] = ' , np.argmax(predictions[0]))

#define accuracy chart image
def plot_image(i, predictions_array, true_label, img) :
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label :
        color = 'blue'
    else :
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})" .format(class_names[predicted_label],
                                          100*np.max(predictions_array),
                                          class_names[true_label]),
               color = color)

#define accuracy chart view
def plot_value_array(i, predictions_array, true_label) :
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = "#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


#show training result
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images) :
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)

plt.show()