import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from datetime import datetime
import collections
import os
# from PIL import Image
import random


slim = tf.contrib.slim

global first
first = True
# Hyperparameter
growth_k = 12
nb_block = [2,4,8] # how many (dense block + Transition Layer) ?
epsilon = 1e-8 # AdamOptimizer epsilon
dropout_rate = 0.2

# Momentum Optimizer will use
#nesterov_momentum = 0.9
#weight_decay = 1e-4

# Label & batch_size
class_num = 12
#batch_size = 128
total_epochs = 20

#testnum = 32
#trainnum = 64
#validnum = 200
#test_last_num = 700

testnum = tf.placeholder(tf.int32)
trainnum = tf.placeholder(tf.int32)
validnum = tf.placeholder(tf.int32)

def getinputs(path):
    filename_queue=tf.train.string_input_producer(path)
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    features=tf.parse_single_example(serialized_example,
                                     features={
                                        'label':tf.FixedLenFeature([], tf.int64),
                                        'img_raw' : tf.FixedLenFeature([], tf.string),
                                     })
    image=tf.decode_raw(features['img_raw'],tf.uint8)
    label=tf.cast(features['label'],tf.int32)
    image=tf.reshape(image,[64,64,1])
    return image,label

def get_batch(image,label,batch_size,crop_size):
    #print(image.shape)
    #print(label.shape)
    images,labels=tf.train.shuffle_batch([image,label],
        batch_size=batch_size,num_threads=10,capacity=10000,min_after_dequeue=100)
    return images,tf.one_hot(tf.reshape(labels,[batch_size]), class_num)

def get_test_batch(image,label,batch_size):
    #images,labels=tf.train.batch([image,label],batch_size=batch_size)
    images,labels=tf.train.shuffle_batch([image,label],
        batch_size=batch_size,num_threads=10,capacity=10000,min_after_dequeue=200)
    return images,tf.one_hot(tf.reshape(labels,[batch_size]), class_num)
def get_valid_batch(image,label,batch_size):
    #images,labels=tf.train.batch([image,label],batch_size=batch_size)
    images,labels=tf.train.shuffle_batch([image,label],
        batch_size=batch_size,num_threads=10,capacity=10000,min_after_dequeue=200)
    return images,tf.one_hot(tf.reshape(labels,[batch_size]), class_num)

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Global_Average_Pooling(x, stride=1):

    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    #It is global average pooling without tflearn

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')



class DenseNet():
    def __init__(self, x, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model = self.Dense_net(x)
        #self.dropout=dropout_rate
        #print("dropout=",self.dropout)
        


    def bottleneck_layer(self, x, scope):
        # print(x)
        #if self.training==True:
        #    self.dropout=dropout_rate
        #else:
        #    self.dropout=1
        #print("dropoutaa=",self.dropout)
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            #print("dropoutbb=",self.dropout)
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):
        input_x = tf.cast(input_x, tf.float32) / 255.0
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[3,3], stride=1, layer_name='conv0')
        #x = Max_Pooling(x, pool_size=[3,3], stride=2)



        for i,v in enumerate(self.nb_blocks) :
            # 6 -> 12 -> 48
            print(i,v)
            x = self.dense_block(input_x=x, nb_layers=v, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))

        """
        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')
        """

        x = self.dense_block(input_x=x, nb_layers=2, layer_name='dense_final')
        self.filters = 2*self.filters
        x = self.transition_layer(x, scope='trans_final')

        # 100 Layer
        #x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        #x = Relu(x)
        #x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)
        x = tf.reshape(x, [-1, class_num])
        return x


path = r'C:\workspace\Alexnet\model\train_model.ckpt'
trainpath = [r'C:\workspace\模型优化\densenet\train.tfrecords',
             r'C:\workspace\模型优化\densenet\train_001.tfrecords']
testpath = [r'C:\workspace\模型优化\densenet\test.tfrecords',
            r'C:\workspace\模型优化\densenet\test_001.tfrecords']
validpath = [r'C:\workspace\模型优化\densenet\validation.tfrecords']

train_image, train_label = getinputs(trainpath)
train_image_batch, train_label_batch =  get_batch(train_image, train_label, trainnum, 0)

test_image, test_label = getinputs(testpath)
test_image_batch, test_label_batch = get_test_batch(test_image, test_label, testnum)



batch_images =  tf.placeholder(tf.float32, shape= [None, 64, 64, 1])
label = tf.placeholder(tf.float32, shape=[None, class_num])
training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = DenseNet(x=batch_images, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
pre_labels = tf.argmax(logits, 1)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon).minimize(cost)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver(tf.global_variables())

valid_image,valid_label = getinputs(validpath)
valid_image_batch, valid_label_batch = get_valid_batch(valid_image, valid_label, validnum)
valid_correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
valid_accuracy = tf.reduce_mean(tf.cast(valid_correct_prediction, tf.float32))

from sklearn.metrics import classification_report

target_names = ['class sg', 'class bm', 'class wd', 'class wt', 'class wj', 'class wo', 'class ym', 'class shq', 'class shj',
                'class no', 'class yh', 'class fb']


config=tf.ConfigProto()
config.gpu_options.allow_growth=True


def train(train_num=64,test_num=32,lr=1e-4,loop_count=10000,decay_step=5000,report_step=100,save_step=1000,restore=False):
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        #tf.train.Saver().restore(sess,path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        epoch_learning_rate = lr
        if restore:
            tf.train.Saver().restore(sess,path)
        for i in range(loop_count):
            if i>0 and i % decay_step==0:
                epoch_learning_rate = epoch_learning_rate / 10
            
            batch_x, batch_y = sess.run([train_image_batch, train_label_batch],feed_dict={trainnum:train_num})
            train_feed_dict = {
                batch_images: np.array(batch_x),
                label: np.array(batch_y),
                learning_rate: epoch_learning_rate,
                training_flag : True
            }
            loss = sess.run([optimizer, cost], feed_dict=train_feed_dict)
            #print(step)
            if i > 0 and i % report_step == 0:
                batch_x,batch_y = sess.run([test_image_batch, test_label_batch],feed_dict={testnum:test_num})
                test_feed_dict = {
                    batch_images: np.array(batch_x),
                    label: np.array(batch_y),
                    learning_rate: epoch_learning_rate,
                    training_flag: False
                }
                accuracy_np = sess.run([accuracy], feed_dict=test_feed_dict)
                print("i:", i,"   test_accuracy:", float(accuracy_np[0]),"train_loss:", float(loss[1]))
                print('---------------------end------------------------------')
                #print(training_flag.eval())
            if i > 0 and i % save_step == 0:
                tf.train.Saver().save(sess, path)
        tf.train.Saver().save(sess, path)
        coord.request_stop()
        coord.join(threads)
    
def test_and_valid(test_loop=1,valid_loop=1,test_num=64,valid_num=64):
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tf.train.Saver().restore(sess,path)
        #test
        test_acc_avg = 0.0
        for i in range(test_loop):
             test_batch_x, test_batch_y = sess.run([test_image_batch, test_label_batch],feed_dict={testnum:test_num})
             test_feed_dict = {
                batch_images: np.array(test_batch_x),
                label: np.array(test_batch_y),
                training_flag: False
            }
        accuracy_np = sess.run([accuracy], feed_dict=test_feed_dict)
        test_acc_avg += accuracy_np[0]
        print("i:", i,'------test_accuracy-----')
        print(test_acc_avg / test_loop)
        print('------test_accuracy-----')
    
        test_true_total = []
        test_pre_total = []
        for i in range(test_loop):
            test_batch_x, test_batch_y = sess.run([test_image_batch, test_label_batch],feed_dict={testnum:test_num})
            test_feed_dict = {
                batch_images: np.array(test_batch_x),
                label: np.array(test_batch_y),
                training_flag: False
            }
            test_pre = sess.run([pre_labels],feed_dict=test_feed_dict)
            test_pre = np.array(test_pre[0])
            test_pre_total = np.append(test_pre_total,test_pre)
            test_true_total = np.append(test_true_total,tf.argmax(test_batch_y, 1).eval())
        print('------test_classification_report-----')
        print(test_true_total, test_pre_total)
        print(test_true_total.shape, test_pre_total.shape)
        print(classification_report(test_true_total, test_pre_total, target_names=target_names))
        print('------test_classification_report-----')
        print('------test_confusion_matrix-----')
        cm = confusion_matrix(y_true=test_true_total, y_pred=test_pre_total)
        print(cm)
        print('------test_confusion_matrix-----')
        
        #valid
        valid_acc_avg = 0.0
        for i in range(valid_loop):
            valid_batch_x, valid_batch_y = sess.run([valid_image_batch, valid_label_batch],feed_dict={validnum:valid_num})
            valid_feed_dict = {
                batch_images: np.array(valid_batch_x),
                label: np.array(valid_batch_y),
                training_flag: False
            }
            valid_accuracy_np = sess.run([valid_accuracy],feed_dict=valid_feed_dict)
            valid_acc_avg += valid_accuracy_np[0]
        print("i:", i,'------valid_accuracy-----')
        print(valid_acc_avg /valid_loop )
        print('------valid_accuracy-----')
    
        valid_true_total = []
        valid_pre_total = []
        for i in range(valid_loop):
            valid_batch_x, valid_batch_y = sess.run([valid_image_batch, valid_label_batch],feed_dict={validnum:valid_num})
            valid_feed_dict = {
                validnum: valid_num,
                batch_images: np.array(valid_batch_x),
                label:np.array( valid_batch_y),
                training_flag: False
            }
            valid_pre = sess.run([pre_labels],feed_dict=valid_feed_dict)
            valid_pre = np.array(valid_pre[0])
            valid_pre_total = np.append(valid_pre_total,valid_pre)
            valid_true_total = np.append(valid_true_total,tf.argmax(valid_batch_y, 1).eval())
        print('------valid_classification_report-----')
        print(classification_report(valid_true_total, valid_pre_total, target_names=target_names))
        print('------valid_classification_report-----')
        print('------valid_confusion_matrix-----')
        cm = confusion_matrix(y_true=valid_true_total, y_pred=valid_pre_total)
        print(cm)
        print('------valid_confusion_matrix-----')
        
        coord.request_stop()
        coord.join(threads)
        
def predict_time(total_num=100):
     with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tf.train.Saver().restore(sess,path)
        total=0.0
        for i in range(total_num):
            test_batch_x, test_batch_y = sess.run([test_image_batch, test_label_batch],feed_dict={testnum:1})
            test_feed_dict = {
             batch_images: np.array(test_batch_x),
             label: np.array(test_batch_y),
             training_flag: False
             }
            a = datetime.now()
            accuracy_np = sess.run([accuracy],feed_dict=test_feed_dict)
            b = datetime.now()
            c = (b - a).microseconds
            if i>0:
               total+=c
            print("c:",c,"total:", total)
        avg=total/((total_num-1)*1000)
        print("-----avg_inference_time_ms：",avg)
        coord.request_stop()
        coord.join(threads)
            
train(train_num=32,loop_count=1002)
test_and_valid(10,10,200,200)
predict_time(1000)
