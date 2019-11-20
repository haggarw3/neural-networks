import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

print("This is the original data: \n")
data = pd.read_csv('tic-tac-toe.csv')
# data['class'].unique()
print(data['class'].value_counts())
print("The shape is:", data.shape)
print(data.head())
print('\n')


data = pd.get_dummies(data, drop_first=True)
X = data.iloc[:,1:]
y = np.where(data['class']==True, 1,0)

print("This is the training data: \n")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
print(X_train.head())
print("The shape is: ", X_train.shape)
print('\n')

print('Converting the dataframe into an array: \n')
xTrain = np.array(X_train)
xTest = np.array(X_test)
yTrain = np.array(y_train)
yTest = np.array(y_test)

print("Building the model: \n")
xTrain = tf.keras.utils.normalize(xTrain, axis=1)
xTest = tf.keras.utils.normalize(xTest, axis=1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(xTrain, yTrain, epochs=10)

val_loss, val_acc = model.evaluate(xTest, yTest)

model.save('tic-tac-toe.model')
new_model = tf.keras.models.load_model('tic-tac-toe.model')
predictions = new_model.predict(xTest)

print("Testing the first 10 predictions of the model: \n")
print('This is the original test data for Y:',yTest[0:9])
print('The predictions are: \n')

for i in range(10):
    print(np.argmax(predictions[i]))