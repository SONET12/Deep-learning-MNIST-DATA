#!/usr/bin/env python
# coding: utf-8

# In[14]:


import tensorflow as tf
tf.__version__


# In[15]:


mnist = tf.keras.datasets.mnist

(X_train, Y_train), (X_test,Y_test) = mnist.load_data()

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)


# In[16]:


import matplotlib.pyplot as plt

print(X_train[0])
plt.imshow (X_train[0])
plt.show()


# In[18]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(X_train,Y_train, epochs=3)


# In[19]:


val_loss, val_acc = model.evaluate(X_test, Y_test)
print (val_loss)
print (val_acc)


# In[23]:


model.save('num_reader.model')
new_model = tf.keras.models.load_model('num_reader.model')
predictions = new_model.predict(X_test)
print(predictions)


# In[24]:


import numpy as np
print(np.argmax(predictions[0]))


# In[26]:


plt.imshow(X_test[0])
plt.show()


# In[ ]:




