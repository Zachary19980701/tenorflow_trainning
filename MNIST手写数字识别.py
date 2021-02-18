#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#下载mnist文件集
mnist = tf.keras.datasets.mnist
(train_images , train_labels) , (test_images , test_labels) = mnist.load_data()


# In[2]:


# 查看数据
print("Train image shape:" , train_images.shape , "Train label shape:" , train_labels.shape)
print("Test image shape:" , test_images.shape , "Test label shape:" , test_labels.shape)


# In[3]:


type(train_images[1 , 1 , 1])


# In[4]:


#图像数据可视化
def plot_image(image):
    plt.imshow(image.reshape(28 , 28) , cmap = 'binary') #以灰度模式显示
    plt.show()


# In[5]:


#图像显示
plot_image(train_images[20000])
print(train_labels[20000])
plot_image(test_images[5000])


# In[6]:


#数据规范化
import numpy as np
int_array = np.array([i for i in range(64)])
print (int_array)
int_array.reshape(8 , 8)
int_array.reshape(4 , 16)
plt.imshow(train_images[20000].reshape(14 , 56) , cmap = 'binary')


# In[7]:


#划分验证集
total_num = len(train_images)
valid_split = 0.2 #验证集比例20%
train_num = int(total_num * (1 - valid_split)) #训练集数目

train_x = train_images[:train_num]
train_y = train_labels[:train_num]

valid_x = train_images[train_num:]
valid_y = train_labels[train_num:]

test_x = test_images
test_y = test_labels


# In[8]:


valid_x.shape


# In[9]:


#图像数据拉伸为一行784列
train_x = train_x.reshape(-1 , 784)
valid_x = valid_x.reshape(-1 , 784)
test_x = test_x.reshape(-1 , 784)


# In[10]:


#特征数据归一化
train_x = tf.cast(train_x / 255.0 , tf.float32)
valid_x = tf.cast(valid_x / 255.0 , tf.float32)
test_x = tf.cast(test_x / 255.0 , tf.float32)


# In[11]:


train_x[1] #查看数据是否合理


# In[12]:


#标签的独热编码，实现稀疏数据
train_y = tf.one_hot(train_y , depth = 10)
valid_y = tf.one_hot(valid_y , depth = 10)
test_y = tf.one_hot(test_y , depth = 10)


# In[13]:


train_y


# In[14]:


#构建模型
def model(x , w , b):
    pred = tf.matmul(x , w) + b
    return tf.nn.softmax(pred)


# In[15]:


#创建变量
W = tf.Variable(tf.random.normal([784 , 10] , mean = 0.0 , stddev = 1.0 , dtype = tf.float32)) #正态分布随机数初始化W
B = tf.Variable(tf.zeros([10]) , dtype = tf.float32) #常数0初始化B


# In[16]:


#定义交叉熵损失函数
def loss(x , y , w , b):
    pred = model(x , w , b)
    loss_ = tf.keras.losses.categorical_crossentropy(y_true = y , y_pred = pred) #tensor提供的交叉熵函数
    return tf.reduce_mean(loss_) #得到均方差


# In[17]:


# 训练模型
#设置训练参数
train_epochs = 20
batch_size = 50
learning_rate = 0.001

#定义梯度计算函数
def grad(x , y , w , b):
    with tf.GradientTape() as tape:
        loss_ = loss(x , y , w , b)
    return tape.gradient(loss_ , [w , b]) # 返回梯度向量


# In[18]:


#选择优化器
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)


# In[19]:


# 定义准确率
def accuracy(x , y , w , b):
    pred = model(x , w , b)
    correct_prediction = tf.equal(tf.argmax(pred , 1) , tf.argmax(y , 1))
    return tf.reduce_mean(tf.cast(correct_prediction , tf.float32))


# In[20]:


#模型训练
total_step = int(train_num / batch_size)

loss_list_train = []
loss_list_valid = []
acc_list_train = []
acc_list_valid = []

for epoch in range(train_epochs):
    for step in range(total_step):
        xs = train_x[step * batch_size : (step + 1) * batch_size]
        ys = train_y[step * batch_size : (step + 1) * batch_size]
        
        grads = grad(xs , ys , W , B)#计算梯度
        optimizer.apply_gradients(zip(grads , [W , B]))
    
    loss_train = loss(train_x , train_y , W , B).numpy()
    loss_valid = loss(valid_x , valid_y , W , B).numpy()
    acc_train = accuracy(train_x , train_y , W , B).numpy()
    acc_valid = accuracy(valid_x , valid_y , W , B).numpy()
    
    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    acc_list_train.append(acc_train)
    acc_list_valid.append(acc_valid)

    print("epoch = {:3d},train_loss={:.4f},train_acc={:.4f},val_loss={:.4f},val_acc={:.4f}".format(
    epoch,loss_train,acc_train,loss_valid,acc_valid))
    #print("epoch" , epoch+1)
    #print("train_loss" , loss_train)
    #print("train_acc" , acc_train)
    #print("val_loss" , loss_valid)
    #print("val_acc" , acc_valid)


# In[21]:


plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(loss_list_train , 'blue' , label = "Train_loss")
plt.plot(loss_list_valid , 'red' , label = "Valid_loss")
plt.legend(loc = 1)


# In[22]:


plt.xlabel("Epochs")
plt.ylabel("Accuary")
plt.plot(acc_list_train , 'blue' , label = "Train_acc")
plt.plot(acc_list_valid , 'red' , label = "Valid_acc")
plt.legend(loc = 1)


# In[23]:


#评估模型
acc_test = accuracy(test_x , test_y , W , B).numpy()
print("TEST_acc" , acc_test)


# In[29]:


#定义预测函数
def predict(x , w , b):
    pred = model(x , w , b)
    result = tf.argmax(pred , 1).numpy()
    return result
pred_test = predict(test_x , W , B)


# In[32]:


test_random = np.random.randint(9999)
plot_image(test_images[test_random])
pred_test[test_random]


# In[ ]:




