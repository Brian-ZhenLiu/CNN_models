import tensorflow as tf
import os
#from PIL import Image
import random
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

slim = tf.contrib.slim

global first
first = True

classnum=12
testnum = tf.placeholder(tf.int32)
trainnum = tf.placeholder(tf.int32)
validnum = tf.placeholder(tf.int32)
learnrate = tf.placeholder(tf.float32)

def getinputs(path):
    filename_queue=tf.train.string_input_producer([path])
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    features=tf.parse_single_example(serialized_example,
                                     features={
                                        'label':tf.FixedLenFeature([], tf.int64),
                                        'img_raw' : tf.FixedLenFeature([], tf.string),
                                     })
    image=tf.decode_raw(features['img_raw'],tf.uint8)
    label=tf.cast(features['label'],tf.int32)
    image=tf.reshape(image,[4096,1])
    return image,label

def get_batch(image,label,batch_size,crop_size):
    #print(image.shape)
    #print(label.shape)
    images,labels=tf.train.shuffle_batch([image,label],
        batch_size=batch_size,num_threads=10,capacity=10000,min_after_dequeue=200)
    return tf.reshape(images,[batch_size,4096]),tf.reshape(labels,[batch_size])

def get_test_batch(image,label,batch_size):
    images,labels=tf.train.batch([image,label],batch_size=batch_size)
    return tf.reshape(images,[batch_size,4096]),tf.reshape(labels,[batch_size])

def get_valid_batch(image,label,batch_size):
    images,labels=tf.train.batch([image,label],batch_size=batch_size)
    return tf.reshape(images,[batch_size,4096]),tf.reshape(labels,[batch_size])
    
class trainwork(object):
    def __init__(self):
        with tf.variable_scope('scop'):
            self.w1=tf.get_variable('w1', [4096,2048],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            self.w2=tf.get_variable('w2', [2048,1024],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            self.w3=tf.get_variable('w3', [1024,classnum],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            self.b1 = tf.get_variable('b1', [2048],initializer=tf.constant_initializer(0.0))
            self.b2 = tf.get_variable('b2', [1024],initializer=tf.constant_initializer(0.0))
            self.b3 = tf.get_variable('b3', [classnum],initializer=tf.constant_initializer(0.0))

    def inference(self,images):
        images=tf.cast(images,tf.float32)/255.0
        l1 = tf.matmul(images, self.w1)+self.b1
        l1=tf.nn.relu(l1)
        l2 = tf.matmul(l1, self.w2)+self.b2
        l2=tf.nn.relu(l2)
        out=tf.matmul(l2, self.w3)+self.b3
        return out

    def test_inference(self,images):
        images=tf.cast(images,tf.float32)/255.0
        l1 = tf.matmul(images, self.w1)+self.b1
        l1=tf.nn.relu(l1)
        l2 = tf.matmul(l1, self.w2)+self.b2
        l2=tf.nn.relu(l2)
        out=tf.matmul(l2, self.w3)+self.b3
        return out
    
    def valid_inference(self,images):
        images=tf.cast(images,tf.float32)/255.0
        l1 = tf.matmul(images, self.w1)+self.b1
        l1=tf.nn.relu(l1)
        l2 = tf.matmul(l1, self.w2)+self.b2
        l2=tf.nn.relu(l2)
        out=tf.matmul(l2, self.w3)+self.b3
        return out
    
    def softmax_loss(self,predicts,labels):
        predicts=tf.nn.softmax(predicts)
        labels=tf.one_hot(labels,classnum)
        loss=-tf.reduce_sum(labels*tf.log(predicts))
        return loss

    def optimer(self,loss,lr=0.001):
        train_step=tf.train.GradientDescentOptimizer(lr).minimize(loss)
        return train_step

path=r'C:\JC\test\train_model.ckpt'
image,label=getinputs(r'C:\JC\tfrecord\64_shuffle/train.tfrecords')
test_image,test_label=getinputs(r'C:\JC\tfrecord\64_shuffle/test.tfrecords')
valid_image,valid_label= getinputs(r'C:\JC\tfrecord\64_shuffle\validation.tfrecords')

batch_image,batch_label=get_batch(image,label,trainnum,0)
work=trainwork()
inf=work.inference(batch_image)
loss=work.softmax_loss(inf,batch_label)
opti=work.optimer(loss,learnrate)

test_image_batch,test_label_batch=get_test_batch(test_image,test_label,testnum)
test_inf=work.test_inference(test_image_batch)
test_labels=tf.one_hot(test_label_batch,classnum)
test_pre = tf.reshape(test_inf, [testnum, classnum])
correct_prediction=tf.equal(tf.argmax(test_inf,1),tf.argmax(test_labels,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
test_pre = tf.argmax(test_pre, 1)
test_true = tf.argmax(test_labels, 1)

valid_image_batch,valid_label_batch=get_valid_batch(valid_image,valid_label,validnum)
valid_inf=work.valid_inference(valid_image_batch)
valid_labels=tf.one_hot(valid_label_batch,classnum)
#train_step=tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
valid_pre = tf.reshape(valid_inf, [validnum, classnum])
valid_correct_prediction=tf.equal(tf.argmax(valid_inf,1),tf.argmax(valid_labels,1))
valid_accuracy=tf.reduce_mean(tf.cast(valid_correct_prediction,tf.float32))
valid_pre = tf.argmax(valid_pre, 1)
valid_true = tf.argmax(valid_labels, 1)

target_names = ['class sg', 'class bm', 'class wd', 'class wt', 'class wj', 'class wo', 'class ym', 'class shq', 'class shj',
                'class no', 'class yh', 'class fb']

init = tf.initialize_all_variables()
config=tf.ConfigProto()
config.gpu_options.allow_growth=True

#init=tf.initialize_all_variables()
def train(train_num=64,test_num=32,lr=1e-4,loop_count=10000,report_step=100,save_step=1000,restore=False):
    with tf.Session(config=config) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        if restore:
            tf.train.Saver().restore(sess,path)
        feed_dict={
            testnum: test_num,
            trainnum: train_num,
            learnrate:lr
        }
        for i in range(loop_count):
            loss_np, _, label_np, image_np, inf_np = sess.run(
                [loss, opti, batch_label, batch_image, inf],feed_dict=feed_dict)
            if i > 0 and i % report_step == 0:
                accuracy_np = sess.run([accuracy],feed_dict=feed_dict)
                print(i, accuracy_np, loss_np)
            if i > 0 and i % save_step == 0:
                tf.train.Saver().save(sess, path)
        tf.train.Saver().save(sess, path)
        coord.request_stop()
        coord.join(threads)

    
def test_and_valid(test_loop=1,valid_loop=1,test_num=64,valid_num=64):
    feed_dict={
        testnum: test_num,
        validnum: valid_num
    }
    with tf.Session(config=config) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tf.train.Saver().restore(sess,path)
        #test
        test_acc_avg = 0.0
        test_true_total=np.array([])
        test_pre_total=np.array([])
        for i in range(0, test_loop):
            accuracy_np = sess.run([accuracy],feed_dict=feed_dict)
            test_pre_1, test_true_1 = sess.run([test_pre, test_true],feed_dict=feed_dict)
            test_pre_1 = np.array(test_pre_1)
            test_true_1 = np.array(test_true_1)
            
            test_acc_avg = test_acc_avg + accuracy_np[0]
            test_true_total = np.concatenate((test_true_total,test_true_1),axis=0)
            test_pre_total = np.concatenate((test_pre_total,test_pre_1), axis=0)
        print('------test_accuracy-----')
        print(test_acc_avg / test_loop)
        print('------test_accuracy-----')

        print('------test_classification_report-----')
        print(classification_report(test_true_total, test_pre_total, target_names=target_names))
        print('------test_classification_report-----')
        print('------test_confusion_matrix-----')
        cm = confusion_matrix(y_true=test_true_total, y_pred=test_pre_total)
        print(cm)
        print('------test_confusion_matrix-----')

        #valid
        if valid_loop>0:
            valid_acc_avg = 0.0
            valid_true_total=np.array([])
            valid_pre_total=np.array([])
            for i in range(0, valid_loop):
                accuracy_np = sess.run([valid_accuracy],feed_dict=feed_dict)        
                valid_pre_1, valid_true_1 = sess.run([valid_pre, valid_true],feed_dict=feed_dict)
                valid_pre_1 = np.array(valid_pre_1)
                valid_true_1 = np.array(valid_true_1)
                                                                                            
                valid_acc_avg = valid_acc_avg + accuracy_np[0]
                valid_true_total = np.concatenate((valid_true_total,valid_true_1),axis=0)
                valid_pre_total = np.concatenate((valid_pre_total,valid_pre_1), axis=0)
        
            print('------valid_accuracy-----')
            print(valid_acc_avg / valid_loop)
            print('------valid_accuracy-----')
    
            print('------valid_classification_report-----')
            print(classification_report(valid_true_total, valid_pre_total, target_names=target_names))
            print('------valid_classification_report-----')
            print('------valid_confusion_matrix-----')
            cm = confusion_matrix(y_true=valid_true_total, y_pred=valid_pre_total)
            print(cm)
            print('------valid_confusion_matrix-----')
        
        coord.request_stop()
        coord.join(threads)

def predict_time(loop=100):
    feed_dict={
        testnum:1
    }
    with tf.Session(config=config) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tf.train.Saver().restore(sess,path)
        total=0.0
        for i in range(loop):
            a = datetime.now()
            accuracy_np = sess.run([accuracy],feed_dict=feed_dict)
            b = datetime.now()
            c = (b - a).microseconds
            total+=c
        print('predict_time(ms): ',total/(loop*1000))
        coord.request_stop()
        coord.join(threads)
        
        
train(train_num=128,loop_count=1200)
test_and_valid(10,10,200,200)
predict_time(1000)





