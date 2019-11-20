import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
from keras import models, layers

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# http://yann.lecun.com/exdb/mnist/

print(X_train.shape)
print(len(X_train))
print(y_train.shape)
print(len(y_train))

# plt.imshow(X_train[0])
# plt.imshow(X_train[0], cmap=plt.cm.binary_r)
# plt.show()

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
train_images = X_train.reshape(60000, 28*28)
train_images = train_images.astype('float32')/255
test_images = X_test.reshape(10000, 28*28)
test_images = test_images.astype('float32')/255

from keras.utils import to_categorical
train_labels = to_categorical(y_train)
test_labels = to_categorical(y_test)
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('Test accuracy is:', test_acc)
print('Network Summary is:', network.summary)
# network.save('mnist_predictor.model1')


print('METHOD 2\n\n')
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
opt = tf.keras.optimizers.Adam(lr=0.001, epsilon=None)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=4)
val_loss, val_acc = model.evaluate(X_test, y_test)
print('Accuracy: ', val_acc)
print(model.summary)
model.save('mnist_predictor.model2')
new_model = tf.keras.models.load_model('mnist_predictor.model1')
predictions = new_model.predict(X_test)
print(np.argmax(predictions[0]))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])