#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#导入数据
mnist = tf.keras.datasets.mnist
(train_images , train_labels) , (test_images , test_labels) = mnist.load_data()


# In[2]:


#特征数据归一化
train_images = train_images / 255.0
test_images = test_images / 255.0


# In[3]:


#标签数据独热编码
train_labels_oh = tf.one_hot(train_labels , depth = 10).numpy()
test_labels_oh = tf.one_hot(test_labels , depth = 10).numpy()


# In[4]:


#创建空模型
model = tf.keras.models.Sequential()

#添加输入层(平坦层)
model.add(tf.keras.layers.Flatten(input_shape = (28 , 28))) #将输入数据压平成一维数据
#添加隐藏层1（全连接层）
model.add(tf.keras.layers.Dense(units = 64 , kernel_initializer = 'normal' , activation = 'relu')) #units神经元数量
#添加隐藏层2
model.add(tf.keras.layers.Dense(units = 32 , kernel_initializer = 'normal' , activation = 'relu'))
#添加输出层
model.add(tf.keras.layers.Dense(units = 10 , activation = 'softmax'))
"""
一次性建模方法
model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape = (28 , 28))
tf.keras.layers.Dense(units = 64 , kernel_initializer = 'normal' , activation = 'relu')
tf.keras.layers.Dense(units = 32 , kernel_initializer = 'normal' , activation = 'relu')
tf.keras.layer.Dense(units = 10 , activation = 'softmax')
])
"""


# In[5]:


#输出模型摘要
model.summary()


# In[6]:


#定义训练模式
model.compile(optimizer = 'adam' , #优化器
              loss = 'categorical_crossentropy' , #损失函数
              metrics = ['accuracy']) #模型评估方式


# In[7]:


#设置训练参数
train_epochs = 10
batch_size = 30


# In[8]:


#训练模型
trian_history = model.fit(train_images , train_labels_oh , 
                          validation_split = 0.2 ,
                          epochs = train_epochs ,
                          batch_size = batch_size ,
                          verbose = 2)


# In[ ]:





# In[ ]:




