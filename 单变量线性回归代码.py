#!/usr/bin/env python
# coding: utf-8

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

np.random.seed(5) #设置随机数种子

x_data = np.linspace(-1 , 1 , 100) #np生成等差数列，100个点，取值在[-1 , 1]
y_data = 2 * x_data + 1.0 + np.random.randn(*x_data.shape) * 0.4 #y=2x+1+噪声，噪声维度和x_data一致

plt.scatter(x_data , y_data) #生成散点图
plt.plot (x_data , 2 * x_data + 1.0 , color = 'red' , linewidth = 3) #画出要学习到的线性函数


# In[14]:


#定义占位符
x = tf.placeholder("float" , name = "x")
y = tf.placeholder("float" , name = "y")


# In[15]:


#定义模型函数
def model(x , w , b):
    return tf.multiply(x , w) + b


# In[16]:


#创建变量
w = tf.Variable(1.0 , name = "w0")
b = tf.Variable(0.0 , name = "b0")
pred = model(x , w , b)


# In[18]:


#模型训练
#设置训练参数
train_epochs = 10 #训练次数
learning_rate = 0.05 #学习率
pred = model(x , w , b) #预测值
#设置损失函数
loss_function = tf.reduce_mean(tf.square(y - pred)) #定义均方差函数作为代价函数
#定义优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function) #设置梯度下降优化器，设置优化率，设置优化器的功能为最小化代价函数。

#创建会话
sess = tf.Session() #声明会话

init = tf.global_variables_initializer() 
sess.run(init) #变量初始化

#开始训练
for epoch in range(train_epochs): #循环十次优化
    for xs , ys in zip(x_data , y_data): #将x_data、y_data的数据拆赋给xs 、ys
        _ , loss = sess.run((optimizer , loss_function) , feed_dict = {x : xs , y : ys}) #运行optimizer、lossfunction。将xs、ys赋给x、y
    b0temp = b.eval(session = sess)
    w0temp = w.eval(session = sess)
    plt.plot (x_data , w0temp * x_data + b0temp) #画图


# In[ ]:




