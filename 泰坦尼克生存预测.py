#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from sklearn import preprocessing


# In[2]:


train_data = pd.read_csv("D:/project/train_data/titanic/train.csv" , header = 0)
test_data = pd.read_csv("D:/project/train_data/titanic/test.csv" , header = 0)
#print(train_data)
#print(test_data)
train_data.describe()
test_data.describe()


# In[3]:


#筛选提取字段
train_selected_cols = ['Survived' , 'Pclass' , 'Name' , 'Sex' , 'Age' , 'SibSp' , 'Parch' , 'Fare' , 'Embarked']
test_selected_cols = ['Pclass' , 'Name' , 'Sex' , 'Age' , 'SibSp' , 'Parch' , 'Fare' , 'Embarked']
train_data = train_data[train_selected_cols]
test_data = test_data[test_selected_cols]
train_data.describe()
test_data.describe()


# In[4]:


train_data.isnull().any()


# In[5]:


test_data.isnull().any()


# In[6]:


#填充空值
train_age_average = train_data['Age'].mean()
train_data['Age'] = train_data['Age'].fillna(train_age_average)
test_age_average = test_data['Age'].mean()
test_data['Age'] = test_data['Age'].fillna(test_age_average)

train_fare_average = train_data['Fare'].mean()
train_data['Fare'] = train_data['Fare'].fillna(train_age_average)
test_age_average = test_data['Fare'].mean()
test_data['Fare'] = test_data['Fare'].fillna(test_age_average)

train_data['Embarked'] = train_data['Embarked'].fillna('S')


# In[7]:


train_data.isnull().any()


# In[8]:


test_data.isnull().any()


# In[9]:


#替换文本特征为数字特征
train_data['Sex'] = train_data['Sex'].map({'female':0 , 'male':1}).astype(int)
test_data['Sex'] = test_data['Sex'].map({'female':0 , 'male':1}).astype(int)

train_data['Embarked'] = train_data['Embarked'].map({'C':0 , 'S':1 , 'Q':2}).astype(int)
test_data['Embarked'] = test_data['Embarked'].map({'C':0 , 'S':1 , 'Q':2}).astype(int)


# In[10]:


train_data[:3]


# In[11]:


test_data[:3]


# In[12]:


#name不需要，使用drop命令暂时剔除
train_data = train_data.drop(['Name'] , axis = 1)
test_data = test_data.drop(['Name'] , axis = 1)


# In[13]:


train_data[:3]


# In[14]:


test_data[:3]


# In[15]:


train_data = train_data.values
x_data = train_data[: , 1:]
y_data = train_data[: , 0]
test_data = test_data.values
test_x = test_data[: , 0:]


# In[16]:


#归一化
gap = preprocessing.MinMaxScaler(feature_range = (0 , 1))
x_data = gap.fit_transform(x_data)
test_x = gap.fit_transform(test_x)


# In[17]:


#模型定义
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units = 64 , input_shape = (7 , ) , kernel_initializer = 'normal' , activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 32 , kernel_initializer = 'normal' , activation = 'sigmoid'))
model.add(tf.keras.layers.Dense(units = 1 , kernel_initializer = 'normal' , activation = 'sigmoid'))


# In[18]:


model.summary()


# In[19]:


model.compile(optimizer = 'adam' , #优化器
              loss = 'binary_crossentropy' , #损失函数
              metrics = ['accuracy']) #模型评估方式


# In[20]:


train_epochs = 100
batch_size = 40


# In[24]:


trian_history = model.fit(x_data , y_data , 
                          validation_split = 0.2 ,
                          epochs = train_epochs ,
                          batch_size = batch_size ,
                          verbose = 2)


# In[25]:


test_pred = model.predict(test_x)
test_pred.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




