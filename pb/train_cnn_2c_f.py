import tensorflow as tf
import os
#from PIL import Image
import random
import numpy as np

classnum=4
testnum=128
batchsize=128
learnrate=0.0001

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
    image=tf.reshape(image,[32,32,3])
    return image,label

def get_batch(image,label,batch_size,crop_size):
    #print(image.shape)
    #print(label.shape)
    images,labels=tf.train.shuffle_batch([image,label],
        batch_size=batch_size,num_threads=10,capacity=10000,min_after_dequeue=200)
    return images,tf.reshape(labels,[batch_size])

def get_test_batch(image,label,batch_size):
    images,labels=tf.train.shuffle_batch([image,label],
        batch_size=batch_size,num_threads=10,capacity=10000,min_after_dequeue=300)
    #images,labels=tf.train.batch([image,label],batch_size=batch_size)
    return images,tf.reshape(labels,[batch_size])
    
class trainwork(object):
    def __init__(self):
        with tf.variable_scope('scop'):
            self.weights={
                'conv1':tf.get_variable('w1', [3,3,3,16],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv2':tf.get_variable('w2', [3,3,16,32],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'fc1':tf.get_variable('w3', [8*8*32,512],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'out':tf.get_variable('w4', [512,classnum],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            }
            self.biases={
                'conv1': tf.get_variable('b1', [16],initializer=tf.constant_initializer(0.0)),
                'conv2': tf.get_variable('b2', [32],initializer=tf.constant_initializer(0.0)),
                'fc1': tf.get_variable('b3', [512],initializer=tf.constant_initializer(0.0)),
                'out': tf.get_variable('b4', [classnum],initializer=tf.constant_initializer(0.0))
            }

    def inference(self,images):
        images=tf.cast(images,tf.float32)/255.0
        l1=tf.nn.conv2d(images,self.weights['conv1'],strides=[1,1,1,1],padding='SAME')+self.biases['conv1']
        relu1=tf.nn.relu(l1)
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        l2 = tf.nn.conv2d(pool1,self.weights['conv2'],strides=[1,1,1,1],padding='SAME')+self.biases['conv2']
        relu2=tf.nn.relu(l2)
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        flat=tf.reshape(pool2,[-1,8*8*32])
        fc1=tf.matmul(flat, self.weights['fc1'])+self.biases['fc1']
        fc1_relu=tf.nn.relu(fc1)
        fc1_drop=tf.nn.dropout(fc1_relu,0.8)
        out=tf.matmul(fc1_relu,self.weights['out'])+self.biases['out']
        return out

    def test_inference(self,images):
        images=tf.cast(images,tf.float32)/255.0
        l1=tf.nn.conv2d(images,self.weights['conv1'],strides=[1,1,1,1],padding='SAME')+self.biases['conv1']
        relu1=tf.nn.relu(l1)
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        l2 = tf.nn.conv2d(pool1,self.weights['conv2'],strides=[1,1,1,1],padding='SAME')+self.biases['conv2']
        relu2=tf.nn.relu(l2)
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        flat=tf.reshape(pool2,[-1,8*8*32])
        fc1=tf.matmul(flat, self.weights['fc1'])+self.biases['fc1']
        fc1_relu=tf.nn.relu(fc1)
        #fc1_drop=tf.nn.dropout(fc1_relu,0.8)
        out=tf.matmul(fc1_relu,self.weights['out'])+self.biases['out']
        return out
    
    def softmax_loss(self,predicts,labels):
        
        predicts=tf.nn.softmax(predicts)
        labels=tf.one_hot(labels,classnum)
        loss=-tf.reduce_sum(labels*tf.log(predicts))
        
        
        #loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(predicts, labels))
        return loss

    def optimer(self,loss,lr=0.001):
        #train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)
        train_step=tf.train.GradientDescentOptimizer(lr).minimize(loss)
        return train_step

path=r'C:\workspace\波士顿评分\model\train_model.ckpt'
image,label=getinputs(r'C:\workspace\波士顿评分\tfrecord\train.tfrecords')
batch_image,batch_label=get_batch(image,label,batchsize,0)
work=trainwork()
inf=work.inference(batch_image)
loss=work.softmax_loss(inf,batch_label)
opti=work.optimer(loss,learnrate)

test_image,test_label=getinputs(r'C:\workspace\波士顿评分\tfrecord\test.tfrecords')
test_image_batch,test_label_batch=get_test_batch(test_image,test_label,testnum)
test_inf=work.test_inference(test_image_batch)
test_labels=tf.one_hot(test_label_batch,classnum)
#train_step=tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(test_inf,1),tf.argmax(test_labels,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init=tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    #tf.train.Saver().restore(sess,path)
    #a=sess.run(work.weights)
    #print(a)
    
   # accuracy_np=sess.run([accuracy])
    #print(accuracy_np)
    
    
    
    
    
    
    
    if os.path.exists(path) is True:
        tf.train.Saver().restore(sess,path)
        print('OK')
    else:
        print('not exist')
    
    

    
    for i in range(8000):
        loss_np,_,label_np,image_np,inf_np=sess.run([loss,opti,batch_label,batch_image,inf])
        if i>0 and i%10==0:
            accuracy_np=sess.run([accuracy])
            print(i,loss_np,accuracy_np)
            print ('-------------------')
        if i%200==0:
            tf.train.Saver().save(sess,path)
    tf.train.Saver().save(sess,path)

    
    
    
    
    coord.request_stop()
    coord.join(threads)






