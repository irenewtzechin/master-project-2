#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("all_params.csv")


# In[38]:


train_data = pd.read_csv("train_params.csv")
test_data = pd.read_csv("test_params.csv")


# In[41]:


for index, row in train_data.iterrows():
    if(train_data.at[index,'Type'] == 'N'):
        train_data.at[index,'Type'] = 0
    else:
        train_data.at[index,'Type'] = 1
train_data = train_data.astype({'Type': int})


# In[42]:


train_data


# In[43]:


test_data


# In[44]:


from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


# In[45]:


# Simple ANN
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


# In[46]:


ann_model = Sequential()
ann_model.add(Dense(50, activation='relu', input_shape=(4,)))
ann_model.add(Dense(50, activation='relu'))
ann_model.add(Dense(50, activation='relu'))
ann_model.add(Dense(50, activation='relu'))
ann_model.add(Dense(2, activation='softmax'))

ann_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[47]:


ann_model.summary()


# In[48]:


ann_model_history =  ann_model.fit(X, y, epochs = 15, batch_size = 32, validation_data=(valX, valy))


# In[49]:


plt.plot(ann_model_history.history['accuracy'])
plt.plot(ann_model_history.history['val_accuracy'])
plt.legend(["accuracy","val_accuracy"])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')


# In[17]:


plt.plot(ann_model_history.history['loss'])
plt.plot(ann_model_history.history['val_loss'])
plt.legend(["loss","val_loss"])
plt.xlabel('Epoch')
plt.ylabel('Loss')


# In[50]:


ann_model.evaluate(testX, testy)


# In[51]:


# Make Prediction
predict = ann_model.predict(testX)
predict


# In[52]:


# distributional probability to integers
yhat = np.argmax(predict, axis = 1)


# In[53]:


from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(np.argmax(testy, axis = 1), yhat)


# In[54]:


print(classification_report(np.argmax(testy, axis=1), yhat))


# In[55]:


yact = np.argmax(testy, axis = 1)
new_df = pd.DataFrame({'Actual': yact, 'ANN': yhat}, columns=['Actual', 'ANN'])
test = pd.read_csv('test_params.csv')
new_df['Actual'].equals(test['Type'])


# In[56]:


new_df


# In[57]:


# Save
manual_result = new_df.to_csv("ann_result_p.csv") 


# In[ ]:




