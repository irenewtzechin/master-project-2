#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("full_prep_set.csv") 


# In[2]:


data


# In[3]:


# Display counts of each classes - Most of Data samples are of normal HeartBeats & its a biased data
sns.catplot(x = 'Type', kind = 'count', data = data)


# In[4]:


# Convert Type to numerical value
# 0: Normal 1: Abnormal

for index, row in data.iterrows():
    if(data.at[index,'Type'] == 'N'):
        data.at[index,'Type'] = 0
    else:
        data.at[index,'Type'] = 1
data = data.astype({'Type': int})


# In[5]:


(data['Type'].value_counts()) / len(data) * 100


# In[6]:


import random

random.seed(11)
#data
# Creating train data with 80% values of original dataframe
train_data_1 = data.groupby('Type', group_keys=False).apply(lambda x: x.sample(frac=0.8))

# Creating test data with rest of the 20% values
test_data_1= data.drop(train_data_1.index)


# In[7]:


train_data_1


# In[8]:


test_data_1


# In[9]:


# Save
train_signal_1 = train_data_1.to_csv("train_signal_1.csv", index=False) # train data 1
test_signal_1 = test_data_1.to_csv("test_signal_1.csv", index=False) # test data 1


# In[10]:


from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


# In[11]:


# Simple ANN
# data1
X1, y1 = train_data_1.iloc[: , :250].values,train_data_1['Type'].values
X1, valX1, y1, valy1= train_test_split(X1,y1,test_size=0.2)
testX1, testy1 = test_data_1.iloc[: , :250].values, test_data_1['Type'].values

y1 = to_categorical(y1, num_classes = 2)
valy1 = to_categorical(valy1, num_classes = 2)
testy1  = to_categorical(testy1, num_classes = 2)

print("X1 shape=" +str(X1.shape))
print("y1 shape=" +str(y1.shape))
print("valX1 shape=" +str(valX1.shape))
print("valy1 shape=" +str(valy1.shape))
print("testX1 shape=" +str(testX1.shape))
print("testy1 shape=" +str(testy1.shape)+'\n')


# In[12]:


ann_model_1 = Sequential()
ann_model_1.add(Dense(50, activation='relu', input_shape=(250,)))
ann_model_1.add(Dense(50, activation='relu'))
ann_model_1.add(Dense(50, activation='relu'))
ann_model_1.add(Dense(50, activation='relu'))
ann_model_1.add(Dense(2, activation='softmax'))

ann_model_1.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[13]:


ann_model_1.summary()


# In[14]:


ann_model_history =  ann_model_1.fit(X1, y1, epochs = 15, batch_size = 32, validation_data=(valX1, valy1))


# In[15]:


plt.plot(ann_model_history.history['accuracy'])
plt.plot(ann_model_history.history['val_accuracy'])
plt.legend(["accuracy","val_accuracy"])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')


# In[16]:


plt.plot(ann_model_history.history['loss'])
plt.plot(ann_model_history.history['val_loss'])
plt.legend(["loss","val_loss"])
plt.xlabel('Epoch')
plt.ylabel('Loss')


# In[17]:


ann_model_1.evaluate(testX1, testy1)


# In[18]:


# Make Prediction
predict = ann_model_1.predict(testX1)
predict


# In[31]:


# distributional probability to integers
yhat = np.argmax(predict, axis = 1)


# In[32]:


from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(np.argmax(testy1, axis = 1), yhat)


# In[21]:


print(classification_report(np.argmax(testy1, axis=1), yhat))


# In[33]:


yhat


# In[34]:


yact = np.argmax(testy1, axis = 1)


# In[35]:


yact


# In[36]:


new_df = pd.DataFrame({'Actual': yact, 'ANN': yhat}, columns=['Actual', 'ANN'])


# In[37]:


new_df


# In[38]:


test = pd.read_csv('test_signal_1.csv')


# In[39]:


new_df['Actual'].equals(test['Type'])


# In[40]:


# Save
ann_result = new_df.to_csv("ann_result.csv") 


# In[ ]:




