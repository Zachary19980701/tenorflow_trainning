#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os  
os.environ['CUDA_VISIBLE_DEVICES']= '0'
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#导入数据
mnist = tf.keras.datasets.mnist
(train_images , train_labels) , (test_images , test_labels) = mnist.load_data()


# In[3]:


#划分验证集
total_num = len(train_images)
valid_split = 0.2
train_num = int(total_num * (1 - valid_split))

train_x = train_images[:train_num]
train_y = train_labels[:train_num]

valid_x = train_images[train_num:]
valid_y = train_labels[train_num:]

test_x = test_images
test_y = test_labels


# In[4]:


#数据塑性，数据稀疏化
train_x = train_x.reshape(-1 , 784)
valid_x = valid_x.reshape(-1 , 784)
test_x = test_x.reshape(-1 , 784)


# In[5]:


#数据归一化处理
train_x = tf.cast(train_x / 255.0 , tf.float32)
valid_x = tf.cast(valid_x / 255.0 , tf.float32)
test_x = tf.cast(test_x / 255.0 , tf.float32)


# In[6]:


#独热编码，解决分类器的属性问题
train_y = tf.one_hot(train_y , depth = 10)
valid_y = tf.one_hot(valid_y , depth = 10)
test_y = tf.one_hot(test_y , depth = 10)


# In[7]:


#构建模型，单隐藏层全连接神经网络

#创建变量
#创建输入到隐藏层变量W1,B1
Input_Dim = 784 #输入层神经元个数
H1_NN = 64 #隐藏层神经元个数
W1 = tf.Variable(tf.random.normal([Input_Dim , H1_NN] , mean = 0.0 , stddev = 1.0 , dtype = tf.float32))
B1 = tf.Variable(tf.zeros([H1_NN] , dtype=tf.float32))
#创建隐藏层到输出层的变量W2,B2
Output_Dim = 10
W2 = tf.Variable(tf.random.normal([H1_NN , Output_Dim] , mean = 0.0 , stddev = 1.0 , dtype = tf.float32))
B2 = tf.Variable(tf.zeros([Output_Dim] , dtype = tf.float32))


# In[8]:


#创建优化列表
W = [W1 , W2]
B = [B1 , B2]


# In[9]:


#定义模型计算
def model(x , w , b):
    x = tf.matmul(x , w[0]) + b[0]
    x = tf.nn.relu(x) #激活函数
    x = tf.matmul(x , w[1]) + b[1]
    pred = tf.nn.softmax(x)
    return pred


# In[10]:


#定义损失函数，直接调用tensorflow的交叉熵函数
def loss(x , y , w , b):
    pred = model(x , w , b)
    loss_ = tf.keras.losses.categorical_crossentropy(y_true = y , y_pred = pred , from_logits=False, label_smoothing=0)
    return tf.reduce_mean(loss_)


# In[11]:


#设置训练超参数
train_epochs = 40
batch_size = 50
learning_rate = 0.01


# In[12]:


#定义梯度函数
def grad(x , y , w , b):
    var_list = w + b ;
    with tf.GradientTape() as tape:
        loss_ = loss(x , y , w , b)
    return tape.gradient(loss_ , var_list)


# In[13]:


#选择Adam优化器
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)


# In[14]:


#定义准确率
def accuracy(x , y , w , b):
    pred = model(x , w , b)
    correct_prediction = tf.equal(tf.argmax(pred , 1) , tf.argmax(y , 1))
    return tf.reduce_mean(tf.cast(correct_prediction , tf.float32))


# In[15]:


#模型训练
steps = int(train_num / batch_size)

loss_list_train = []
loss_list_valid = []
acc_list_train = []
acc_list_valid = []

for epoch in range(train_epochs):
    for step in range(steps):
        xs = train_x[step * batch_size : (step + 1) * batch_size]
        ys = train_y[step * batch_size : (step + 1) * batch_size]
        
        grads = grad(xs , ys , W , B)
        
        optimizer.apply_gradients(zip(grads , W + B))
        
    loss_train = loss(train_x , train_y , W , B).numpy()
    loss_valid = loss(valid_x , valid_y , W , B).numpy()
    acc_train = accuracy(train_x , train_y , W , B).numpy()
    acc_valid = accuracy(valid_x , valid_y , W , B).numpy()
    
    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    acc_list_train.append(acc_train)
    acc_list_valid.append(acc_valid)
    
    print("epoch = {:3d},train_loss={:.4f},train_acc={:.4f},val_loss={:.4f},val_acc={:.4f}".format(epoch,loss_train,acc_train,loss_valid,acc_valid))


# In[ ]:




