import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#download imdb dataset
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}" .format(len(train_data), len(train_labels)))

#dictionary mapping words to integer index
word_index = imdb.get_word_index()

#The first indices are reserved
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] =1
word_index["<UNK>"] =2
word_index["<UNUSED>"] =3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text) :
    return ' ' .join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_data[0]))

#convert fixed length to review words
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

#setup training model
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16)) #word embedding, input_dim / output_dim
model.add(keras.layers.GlobalAveragePooling1D()) #averaging over the sequence dimension
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

#define training model
model.compile(optimizer = tf.train.AdamOptimizer(), 
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

#training model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size = 512,
                    validation_data = (x_val, y_val),
                    verbose = 1)

#evaluate the model
result = model.evaluate(test_data, test_labels)
print(result)

#create information graph accuracy and loss after training time 
history_dict = history.history
history_dict.keys()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)


acc_value = history_dict['acc']
val_acc_value = history_dict['val_acc']

#show training result
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.plot(epochs, acc, 'ro', label = 'Training acc')
plt.plot(epochs, val_acc, 'r', label = 'Validation acc')
plt.title('Training and validation loss and acc')
plt.xlabel('Epochs')
plt.ylabel('Loss & Acc')
plt.legend()
plt.show()