import tensorflow as tf  
from tensorflow.python.framework import graph_util  

    
img_width=32
class_num=4

class parameter(object):
    def __init__(self,classnum,reuse):
        with tf.variable_scope('scop',reuse=reuse):
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
            
parameter=parameter(class_num,None)
images=tf.placeholder("float",[None,img_width,img_width,3],name='input')

#images=tf.cast(images,tf.float32)/255.0  
l1=tf.nn.conv2d(images,parameter.weights['conv1'],strides=[1,1,1,1],padding='SAME')+parameter.biases['conv1']
relu1=tf.nn.relu(l1)
pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
l2 = tf.nn.conv2d(pool1,parameter.weights['conv2'],strides=[1,1,1,1],padding='SAME')+parameter.biases['conv2']
relu2=tf.nn.relu(l2)
pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
flat=tf.reshape(pool2,[-1,8*8*32])
fc1=tf.matmul(flat, parameter.weights['fc1'])+parameter.biases['fc1']
fc1_relu=tf.nn.relu(fc1)
fc1_drop=tf.nn.dropout(fc1_relu,0.8)
out=tf.matmul(fc1_drop,parameter.weights['out'])+parameter.biases['out']

out= tf.nn.softmax(out,name='output')



config=tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
with tf.Session(config=config) as sess:
    saver=tf.train.Saver()
    saver.restore(sess,r'C:\workspace\波士顿评分\model\train_model.ckpt')
    graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])
    tf.train.write_graph(graph, r'C:\workspace\波士顿评分\model', 'cnn_2.pb', as_text=False)
print('done')


            