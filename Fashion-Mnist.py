# 导入函数库
import tensorflow as tf
import numpy as np
import os
import cv2
import glob
from skimage import io,transform
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score


# 数据集地址 && 模型保存地址
train_path = "./my_work/mnist_train/"
test_path = "./my_work/mnist_test/"
save_path = "./my_work/save/"

# 将所有的图片resize成28*28*1的大小
w, h, c = 28, 28, 1

# 设置超参数
batch_size = 100
learning_rate = 0.01
learning_rate_decay = 0.99
max_steps = 10000
regularizer = tf.contrib.layers.l2_regularizer(0.0001)

def formats(idx):
    x = np.zeros(10)
    if idx == 0:
        x[0] = 1
    elif idx ==1:
        x[1] = 1
    elif idx ==2:
        x[2] = 1
    elif idx ==3:
        x[3] = 1
    elif idx ==4:
        x[4] = 1
    elif idx ==5:
        x[5] = 1
    elif idx ==6:
        x[6] = 1
    elif idx ==7:
        x[7] = 1
    elif idx ==8:
        x[8] = 1
    elif idx ==9:
        x[9] = 1
    return x


# 读取图片
def read_img(path):
    cate = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs = []
    labels = []
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.png'):
            print('reading the images:%s'%(im))
            img = io.imread(im)
            img = cv2.resize(img, (w, h))
            img = np.reshape(img, (w, h, c))
            img = img / 255
            imgs.append(img)
            labels.append(formats(idx))
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

with tf.name_scope('read_img'):
    x_train, y_train = read_img(train_path)
    x_val, y_val = read_img(test_path)

# 打乱顺序
def disorder(data, label):
    np.random.seed(int(time.time()))
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]
    return(data, label)

with tf.name_scope('disorder'):
    x_train, y_train = disorder(x_train, y_train)
    x_val, y_val = disorder(x_val, y_val)

def hidden_layer(input_tensor, regularizer, avg_class, resuse):

    # 输入网络的尺寸为64×28×28×1
    with tf.variable_scope("C1-conv", reuse=resuse):
        conv1_weights = tf.get_variable("weight", [5, 5, 1, 32],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1],
                         padding="SAME", use_cudnn_on_gpu=True,)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    # 得到特征图大小为64×32@28x28

    # 输入的特征图大小为64×32@28×28
    with tf.name_scope("S2-max_pool",):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding="SAME")
    # 得到特征图大小为64×32@14×14

    # 输入特征图大小为64×32@14×14
    with tf.variable_scope("C3-conv",reuse=resuse):
        conv2_weights = tf.get_variable("weight", [5, 5, 32, 64],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    # 得到特征图大小为64×64@14×14

    # 输入特征图大小为64×64@14×14
    with tf.name_scope("S4-max_pool",):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        shape = pool2.get_shape().as_list()
        nodes = shape[1] * shape[2] * shape[3]
        reshaped = tf.reshape(pool2, [shape[0], nodes])
    # 得到了64×3136的矩阵

    # 输入64×3136的矩阵
    with tf.variable_scope("layer5-full1",reuse=resuse):
        Full_connection1_weights = tf.get_variable("weight", [nodes, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection("losses", regularizer(Full_connection1_weights))
        Full_connection1_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))
        if avg_class ==None:
            Full_1 = tf.nn.relu(tf.matmul(reshaped, Full_connection1_weights) + Full_connection1_biases)
        else:
            Full_1 = tf.nn.relu(tf.matmul(reshaped, avg_class.average(Full_connection1_weights)) + avg_class.average(Full_connection1_biases))
    # 输出64×512的矩阵

    # 输入64×512的矩阵
    with tf.variable_scope("layer6-full2",reuse=resuse):
        Full_connection2_weights = tf.get_variable("weight", [512, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection("losses", regularizer(Full_connection2_weights))
        Full_connection2_biases = tf.get_variable("bias", [10], initializer=tf.constant_initializer(0.1))
        if avg_class == None:
            result = tf.matmul(Full_1, Full_connection2_weights) + Full_connection2_biases
        else:
            result = tf.matmul(Full_1, avg_class.average(Full_connection2_weights)) + avg_class.average(Full_connection2_biases)
    # 输出64×10的矩阵
    return result


# 获取数据信息
x = tf.placeholder(tf.float32, [batch_size ,28,28,1],name="x-input")
y_ = tf.placeholder(tf.float32, [None, 10], name="y-input")


# 前向传播得到y值
y = hidden_layer(x,regularizer,avg_class=None,resuse=False)

training_step = tf.Variable(0, trainable=False)
variable_averages = tf.train.ExponentialMovingAverage(0.99, training_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())
average_y = hidden_layer(x,regularizer,variable_averages,resuse=True)
with tf.name_scope('cross_entropy_mean'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)
with tf.name_scope('loss'):
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('loss', loss)
with tf.name_scope('learning_rate'):
    learning_rate = tf.train.exponential_decay(learning_rate,
                                 training_step, 55000 /batch_size , learning_rate_decay, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=training_step)
with tf.control_dependencies([train_step, variables_averages_op]):
    train_op = tf.no_op(name='train')
crorent_predicition = tf.equal(tf.arg_max(average_y,1),tf.argmax(y_,1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(crorent_predicition,tf.float32))
    tf.summary.scalar('accuracy', accuracy)
#定义一个函数，按批次取数据
def next_batches(x_train, y_train, batch_size):
    with tf.name_scope('accuracy'):
        x_train, y_train = disorder(x_train, y_train)
        return x_train[:batch_size], y_train[:batch_size]


# 走起
saver = tf.train.Saver()
merged = tf.summary.merge_all()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    writer = tf.summary.FileWriter("log/",sess.graph)
    for i in range(max_steps):
        if i % 100 == 0:
            x_val_a, y_val_a = next_batches(x_val, y_val, batch_size=batch_size)
            reshaped_x2 = np.reshape(x_val_a, (batch_size,28, 28, 1))
            validate_feed = {x: reshaped_x2, y_: y_val_a}
            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            print("After %d trainging step(s) ,validation accuracy"
                  "using average model is %g%%" % (i, validate_accuracy * 100))
            y_array = sess.run(y, feed_dict=validate_feed)
            y_array = 1 / (1 + np.exp(-y_array))


            auc = metrics.roc_auc_score(y_val_a, y_array, average='macro')
     #       print(auc)
            fpr, tpr, thresholds = metrics.roc_curve(y_val_a.ravel(),y_array.ravel())
#            auc = metrics.auc(fpr, tpr)

            print(y_array, auc)
            

    #FPR就是横坐标,TPR就是纵坐标
            plt.plot(fpr, tpr, c = 'green', lw = 6, alpha = 0.7, label = 'AUC=%.3f' % auc)
            plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
            plt.xlim((-0.01, 1.02))
            plt.ylim((-0.01, 1.02))
            plt.xticks(np.arange(0, 1.1, 0.1))
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.xlabel('False Positive Rate', fontsize=13)
            plt.ylabel('True Positive Rate', fontsize=13)
            plt.grid(b=True, ls=':')
            plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
            plt.title('ROC', fontsize=17)
            plt.show()
            
        x_train_a, y_train_a = next_batches(x_train, y_train, batch_size=batch_size)
        reshaped_xs = np.reshape(x_train_a, (batch_size ,28,28,1))
        summary,_ = sess.run([merged,train_op], feed_dict={x: reshaped_xs, y_: y_train_a})
        writer.add_summary(summary,i)
       
    saver.save(sess,save_path)




