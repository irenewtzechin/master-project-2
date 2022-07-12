#!/usr/bin/env python
# coding: utf-8

# In[25]:


# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[26]:


train_data = pd.read_csv("train_params.csv")
test_data = pd.read_csv("test_params.csv")


# In[27]:


train_data


# In[28]:


test_data


# In[30]:


for index, row in train_data.iterrows():
    if(train_data.at[index,'Type'] == 'N'):
        train_data.at[index,'Type'] = 0
    else:
        train_data.at[index,'Type'] = 1
train_data = train_data.astype({'Type': int})


# In[31]:


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


# In[32]:


X = X.reshape(len(X),X.shape[1],1)
valX = valX.reshape(len(valX),valX.shape[1],1)
testX = testX.reshape(len(testX),testX.shape[1],1)


# In[33]:


print("X shape=" +str(X.shape))
print("y shape=" +str(y.shape))
print("valX shape=" +str(valX.shape))
print("valy shape=" +str(valy.shape))
print("testX shape=" +str(testX.shape))
print("testy shape=" +str(testy.shape)+'\n')


# In[34]:


from keras.models import Sequential
from keras.layers import Dense # for fully connected layers dense will be used
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam

# avoid overfitting by normalizing the samples
from tensorflow.keras.layers import BatchNormalization


# In[35]:


def build_model():
    model = Sequential()
    
    # Filters = Units in Dense Total number of Neurons
    # Padding = 'same' , zero-padding, Add zero pixels all around input data
    model.add(Conv1D(filters = 64, kernel_size = 6, activation='relu', padding = 'same', input_shape = (4, 1))) #we pass individual values hence not 100000,187,1
    
    # Normalization to avoid overfitting
    model.add(BatchNormalization())
    
    # Pooling 
    model.add(MaxPooling1D(pool_size=(3), strides = (2), padding = 'same'))

    model.add(Conv1D(filters = 64, kernel_size = 6, activation='relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides = (2), padding = 'same'))

    model.add(Conv1D( filters = 64, kernel_size = 6, activation='relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides = (2), padding = 'same'))

    # Flatten 
    model.add(Flatten())

    # Fully connected layer
    # input layer
    model.add(Dense(units = 64, activation='relu'))
    
    # Hidden Layer
    model.add(Dense(units = 64, activation='relu'))
    
    # Output Layer
    model.add(Dense(units = 2, activation='softmax'))

    # loss = 'categorical_crossentropy'
    model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model


# In[36]:


model = build_model()


# In[37]:


model.summary()


# In[38]:


history = model.fit(X, y, epochs = 15, batch_size = 32, validation_data=(valX, valy))


# In[39]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(["accuracy","val_accuracy"])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')


# In[40]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["loss","val_loss"])
plt.xlabel('Epoch')
plt.ylabel('Loss')


# In[41]:


model.evaluate(testX, testy)


# In[42]:


# Make Prediction
predict = model.predict(testX)
predict


# In[43]:


# distributional probability to integers
yhat = np.argmax(predict, axis = 1)


# In[44]:


from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(np.argmax(testy, axis = 1), yhat)


# In[45]:


print(classification_report(np.argmax(testy, axis=1), yhat))


# In[46]:


yact = np.argmax(testy, axis = 1)


# In[47]:


new_df = pd.DataFrame({'Actual': yact, 'CNN': yhat}, columns=['Actual', 'CNN'])


# In[50]:


test = pd.read_csv('test_params.csv')


# In[51]:


new_df['Actual'].equals(test['Type'])


# In[52]:


# Save
cnn_result = new_df.to_csv("cnn_result_p.csv") 


# In[ ]:




