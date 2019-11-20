from keras.datasets import imdb
from keras import models, layers
from keras import optimizers, losses, activations, metrics
import pandas as pd
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)
import numpy as np
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
# print(x_train)
# print(x_train[0])
# print(x_train[0:2])

word_index = imdb.get_word_index()
print(word_index)
print('\n')
reverse_word_index = dict((value, key) for (key,value) in word_index.items())
print(reverse_word_index)
sentence = []
for i in x_train[0]:
    sentence.append(reverse_word_index[i])
print(sentence)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train_vector = vectorize_sequences(x_train)
x_test_vector = vectorize_sequences(x_test)
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')

network = models.Sequential()
network.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))
# network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
network.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
x_val = x_train_vector[:10000]
partial_x_train = x_train_vector[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
model = network.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
print(model.history)  # it is a dictionary

import matplotlib.pyplot as plt

history_dict = model.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf() # Clear the previous plot
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

results = model.evaluate(x_test, y_test)
print('Model evaluated on test data:', results)
print('Predictions on the test data are:', model.predict(x_test))