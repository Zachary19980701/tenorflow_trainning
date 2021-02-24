#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


cifar10 = tf.keras.datasets.cifar10
(x_train , y_train) , (x_test , y_test) = cifar10.load_data()


# In[3]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[ ]:





# In[8]:


#数据标准化
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# In[13]:


#定义网络结构

#建立空框架
model = tf.keras.models.Sequential()

#第一卷积层
model.add(tf.keras.layers.Conv2D(filters = 32 ,
                                 kernel_size = (3 , 3) ,
                                 input_shape = (32 , 32 , 3) ,
                                 activation = 'relu' , 
                                 padding = 'same'))

#防止过拟合，抛弃掉一些特征数据
model.add(tf.keras.layers.Dropout(rate = 0.3))

#第一个池化层
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2 , 2)))

#第二个卷积层
model.add(tf.keras.layers.Conv2D(filters =64 ,
                                 kernel_size  = (3 , 3) ,
                                 activation = 'relu' ,
                                 padding = 'same'))

#防止过拟合
model.add(tf.keras.layers.Dropout(rate = 0.3))

#第二个池化层
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2 , 2)))

#平坦层
model.add(tf.keras.layers.Flatten())

#全连接层
model.add(tf.keras.layers.Dense(128 , activation = 'softmax'))

#输出层
model.add(tf.keras.layers.Dense(10 , activation = 'softmax'))


# In[14]:


model.summary()


# In[17]:


train_epochs = 100
batch_size = 1000
model.compile(optimizer = 'adam' ,
              loss = 'sparse_categorical_crossentropy' ,
              metrics = ['accuracy'])


# In[18]:


train_history = model.fit(x_train , y_train , 
                          validation_split = 0.2 ,
                          epochs = train_epochs ,
                          batch_size = batch_size ,
                          verbose = 2)


# In[19]:


test_loss , test_acc = model.evaluate(x_test , y_test , verbose = 2)


# In[21]:


preds = model.predict_classes(x_test)
preds


# In[ ]:




