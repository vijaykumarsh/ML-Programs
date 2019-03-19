
# coding: utf-8

# In[7]:

import tensorflow as tf
from tensorflow import keras

(X_train,y_train),(X_test,y_test)=keras.datasets.mnist.load_data()


# In[8]:

X_train.shape


# In[2]:

#set up the model
model=keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                        keras.layers.Dense(128,activation=tf.nn.relu),
                        keras.layers.Dense(10,activation=tf.nn.softmax)
                       ])


# In[3]:

model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[9]:

#train the model
model.fit(X_train,y_train,epochs=5)


# In[10]:

test_loss,test_acc=model.evaluate(X_test,y_test)
print('test accuracy',test_loss)


# In[11]:

pred=model.predict(X_test)

