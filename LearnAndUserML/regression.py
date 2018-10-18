from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

#Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

#build model
def build_model() :
    model = keras.Sequential([
        keras.layers.Dense(64, activation= tf.nn.relu,
                           input_shape = (train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
        ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss = 'mse', # Mean Squared Error -> common loss function for regression problems
                  optimizer = optimizer,
                  metrics = ['mae']) # Mean Absolute Error -> common metric for regression

    return model

model = build_model()
model.summary()

#Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback) :
    def on_epoch_end(self, epoch, logs) :
        if epoch % 100 == 0 : print('')
        print('.', end='')


EPOCHS = 500

#Store training states
history = model.fit(train_data, train_labels, epochs = EPOCHS,
                    validation_split = 0.2, verbose = 0,
                    callbacks=[PrintDot()])

#Show Training result
def plot_history(history) :
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label= 'Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label='Val loss')

    plt.legend()
    plt.ylim([0,5])
    plt.show()

plot_history(history)

model = build_model()

#The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=20) #과적합을 방지하기 위해 개선의 여지가 없을 경우 학습을 종료 시키는 콜백 함수

history = model.fit(train_data, train_labels, epochs = EPOCHS,
                    validation_split = 0.2, verbose = 0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)

#evaluate model
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error : ${:7.2f}" .format(mae * 1000))

test_predictions = model.predict(test_data).flatten()

#show prediction result
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Preictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")
plt.show()