'''
Created on 2017年11月1日

@author: adm
'''
import tensorflow as tf
import numpy
from PIL import Image
import sys

config=tf.ConfigProto()
config.gpu_options.allow_growth=True
pb_path=r'C:\workspace\波士顿评分\model\cnn_2.pb'

pb_file=open(pb_path, 'rb')
graph = tf.GraphDef()
graph.ParseFromString(pb_file.read())

image_path=r'C:\Users\MEDCARE-OO\Desktop\train\154_three.jpg'

with tf.Session(config=config) as sess:
    xs=numpy.empty((1,32,32,3),dtype='float32')
    image = Image.open(image_path)
    image=image.resize((32,32))
    img_ndarray = numpy.asarray(image, dtype='float32')
    #print(img_ndarray.shape)
    xs= numpy.ndarray.reshape(img_ndarray,[1,32,32,3])
    xs = numpy.multiply(xs, 1.0 / 255.0)
    output = tf.import_graph_def(graph, input_map={'input:0':xs}, return_elements=['output:0'], name='a') 
    ret=sess.run(output)
    print(ret)
#graph.close()

        
        
