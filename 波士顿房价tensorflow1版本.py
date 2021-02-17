#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import pandas as pd
from sklearn.utils import shuffle
tf.reset_default_graph()


# In[2]:


#处理数据
bp = pd.read_csv("D:/project/tensorflow_training_data/boston_house_prices.csv" , header = 0)
print(bp.describe())

bp = bp.values
print(bp)
bp = np.array(bp)


# In[3]:


bp = np.array(bp)

#归一化处理
for i in range(12):
    bp[: , i] = bp[: , i] / (bp[: , i].max() - bp[: , i].min())

x_data = bp[: , :12]
y_data = bp[: , 12]
print(x_data , x_data.shape , y_data , y_data.shape)


# In[4]:


#构建模型
#定义特征数据和标签数据占位符
x = tf.placeholder(tf.float32 , [None , 12] , name = "X")
y = tf.placeholder(tf.float32 , [None , 1] , name = "Y")

#构建模型函数
with tf.name_scope("Model"):
    #初始化值为shape=（12 ， 1）的随机数
    w = tf.Variable(tf.random_normal([12 , 1] , stddev = 0.01) , name = "W")
    #b的初始化值为1
    b = tf.Variable(1.0 , name = "b")
    #矩阵乘法
    def model(x , w , b):
        return tf.matmul(x , w) + b
    
    #预测计算操作
    pred = model(x , w , b)


# In[6]:


#模型训练
#设置超参数
#迭代次数
train_epochs = 100
#学习率
learning_rate = 0.01




#定义损失函数
with tf.name_scope("LossFunction"):
    loss_function = tf.reduce_mean(tf.pow(y-pred , 2)) #均方误差
    
#选择优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

#声明会话，进行训练
sess = tf.Session()

logdir = "D:\project\log01"
sum_loss_op = tf.summary.scalar("loss" , loss_function)#记录损失loss值
merged = tf.summary.merge_all() #日志文件合并，一次性写入

#初始化参数
init = tf.global_variables_initializer()
sess.run(init)

#创建摘要文件写入器
writer = tf.summary.FileWriter(logdir , sess.graph)
#保存loss列表
loss_list = []

#迭代计算
for epoch in range (train_epochs):
    loss_sum = 0.0
    for xs,ys in zip (x_data,y_data):
        
        xs = xs.reshape(1 , 12)
        ys = ys.reshape(1 , 1)
        
        _ , summary_str ,loss = sess.run([optimizer , sum_loss_op , loss_function] , feed_dict = {x : xs , y : ys})
        writer.add_summary(summary_str , epoch)
        loss_sum = loss_sum + loss
        
    #打乱数据顺序，保证训练效果。避免规律性输入
    xvalues , yvalues = shuffle(x_data , y_data)
    
    b0temp = b.eval(session = sess)
    w0temp = w.eval(session = sess)
    loss_average = loss_sum / len(y_data)
    loss_list.append(loss_average)
    print("epoch=" , epoch + 1 , "loss=" , loss_average , "b=" , b0temp , "w=" , w0temp)
    
plt.plot(loss_list)

writer = tf.summary.FileWriter(logdir , tf.get_default_graph())
writer.close


# In[7]:


#模型应用
n = np.random.randint(506)
x_test = x_data[n]

x_test = x_test.reshape(1 , 12)
predict = sess.run(pred , feed_dict = {x : x_test})
print("预测值： %f" %predict)

target = y_data[n]
print("真实值：%f" %target)


# In[ ]:





# In[ ]:





# In[ ]:




