#!/usr/bin/env python
# coding: utf-8

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv("train_params.csv")
test_data = pd.read_csv("test_params.csv")

for index, row in train_data.iterrows():
    if(train_data.at[index,'Type'] == 'N'):
        train_data.at[index,'Type'] = 0
    else:
        train_data.at[index,'Type'] = 1
train_data = train_data.astype({'Type': int})

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

X, y = train_data.iloc[: , :4].values,train_data['Type'].values
X, valX, y, valy= train_test_split(X,y,test_size=0.2)
testX, testy = test_data.iloc[: , :4].values, test_data['Type'].values

y = to_categorical(y, num_classes = 2)
valy = to_categorical(valy, num_classes = 2)
testy  = to_categorical(testy, num_classes = 2)

print("X shape=" +str(X.shape))
print("y shape=" +str(y.shape))
print("valX shape=" +str(valX.shape))
print("valy shape=" +str(valy.shape))
print("testX shape=" +str(testX.shape))
print("testy shape=" +str(testy.shape)+'\n')

from keras.models import Sequential
from keras.layers import Dense # for fully connected layers dense will be used
from keras.layers import LSTM, Dropout
from keras.optimizers import Adam

lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(4,1)))
lstm_model.add(Dense(128, activation = 'relu'))
lstm_model.add(Dropout(0.3))
lstm_model.add(Dense(2, activation = 'softmax'))

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

lstm_model.summary()

lstm_model_history = lstm_model.fit(X, y, epochs = 15, batch_size = 32, validation_data = (valX, valy))

import matplotlib.pyplot as plt
plt.plot(lstm_model_history.history['accuracy'])
plt.plot(lstm_model_history.history['val_accuracy'])
plt.legend(["accuracy","val_accuracy"])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.plot(lstm_model_history.history['loss'])
plt.plot(lstm_model_history.history['val_loss'])
plt.legend(["loss","val_loss"])
plt.xlabel('Epoch')
plt.ylabel('Loss')

lstm_model.evaluate(testX, testy)

# Make Prediction
predict = lstm_model.predict(testX)
predict

# distributional probability to integers
yhat = np.argmax(predict, axis = 1)


from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(np.argmax(testy, axis = 1), yhat)
print(classification_report(np.argmax(testy, axis=1), yhat))
