#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train_data = pd.read_csv("train_signal_1.csv")
test_data = pd.read_csv("test_signal_1.csv")


# In[3]:


train_data


# In[4]:


test_data


# In[5]:


X, y = train_data.iloc[: , :250].values,train_data['Type'].values
from sklearn.model_selection import train_test_split
X, valX, y, valy= train_test_split(X,y,test_size=0.2)
testX, testy = test_data.iloc[: , :250].values, test_data['Type'].values

from keras.utils.np_utils import to_categorical
y = to_categorical(y)
testy  = to_categorical(testy)
valy  = to_categorical(valy)

print("X shape=" +str(X.shape))
print("y shape=" +str(y.shape))
print("valX shape=" +str(valX.shape))
print("valy shape=" +str(valy.shape))
print("testX shape=" +str(testX.shape))
print("testy shape=" +str(testy.shape)+'\n')


# In[6]:


from keras.models import Sequential
from keras.layers import Dense # for fully connected layers dense will be used
from keras.layers import LSTM, Dropout
from keras.optimizers import Adam


# In[7]:


lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(250,1)))
lstm_model.add(Dense(128, activation = 'relu'))
lstm_model.add(Dropout(0.3))
lstm_model.add(Dense(2, activation = 'softmax'))

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[8]:


lstm_model.summary()


# In[9]:


lstm_model_history = lstm_model.fit(X, y, epochs = 15, batch_size = 32, validation_data = (valX, valy))


# In[10]:


import matplotlib.pyplot as plt
plt.plot(lstm_model_history.history['accuracy'])
plt.plot(lstm_model_history.history['val_accuracy'])
plt.legend(["accuracy","val_accuracy"])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')


# In[11]:


plt.plot(lstm_model_history.history['loss'])
plt.plot(lstm_model_history.history['val_loss'])
plt.legend(["loss","val_loss"])
plt.xlabel('Epoch')
plt.ylabel('Loss')


# In[12]:


lstm_model.evaluate(testX, testy)


# In[13]:


# Make Prediction
predict = lstm_model.predict(testX)
predict


# In[14]:


# distributional probability to integers
yhat = np.argmax(predict, axis = 1)


# In[15]:


from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(np.argmax(testy, axis = 1), yhat)


# In[16]:


print(classification_report(np.argmax(testy, axis=1), yhat))


# In[17]:


yact = np.argmax(testy, axis = 1)


# In[19]:


new_df = pd.DataFrame({'Actual': yact, 'LSTM': yhat}, columns=['Actual', 'LSTM'])


# In[21]:


new_df


# In[20]:


test = pd.read_csv('test_signal_1.csv')


# In[22]:


new_df['Actual'].equals(test['Type'])


# In[23]:


# Save
lstm_result = new_df.to_csv("lstm_result.csv") 


# In[ ]:




