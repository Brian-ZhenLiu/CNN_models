import tensorflow as tf
import numpy as np
import collections
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from datetime import datetime

slim = tf.contrib.slim

global first
first = True


classnum = 12
testnum = tf.placeholder(tf.int32)
trainnum = tf.placeholder(tf.int32)
validnum = tf.placeholder(tf.int32)
learnrate = tf.placeholder(tf.float32)

path = r'C:\workspace\Alexnet\model\train_model.ckpt'
trainpath = [r'C:\workspace\模型优化\densenet\train.tfrecords',
             r'C:\workspace\模型优化\densenet\train_001.tfrecords']
testpath = [r'C:\workspace\模型优化\densenet\test.tfrecords',
            r'C:\workspace\模型优化\densenet\test_001.tfrecords']
validpath = [r'C:\workspace\模型优化\densenet\validation.tfrecords']



trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def getinputs(path):
    filename_queue = tf.train.string_input_producer(path)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    image = tf.reshape(image, [128, 128, 1])
    return image, label


def get_batch(image, label, batch_size, crop_size):
    # print(image.shape)
    # print(label.shape)
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=batch_size, num_threads=10, capacity=10000,
                                            min_after_dequeue=100)
    return images, tf.reshape(labels, [batch_size])


def get_test_batch(image, label, batch_size):
    # images,labels=tf.train.batch([image,label],batch_size=batch_size)
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=batch_size, num_threads=10, capacity=10000,
                                            min_after_dequeue=200)
    return images, tf.reshape(labels, [batch_size])


def get_valid_batch(image, label, batch_size):
    # images,labels=tf.train.batch([image,label],batch_size=batch_size)
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=batch_size, num_threads=10, capacity=10000,
                                            min_after_dequeue=200)
    return images, tf.reshape(labels, [batch_size])


def inception_v3_base(inputs, scope=None):
    end_points = {}

    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='VALID'):
            # 299 x 299 x 3
            # net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='Conv2d_1a_3x3')

            # 149 x 149 x 32     (128*128*1)
            net = slim.conv2d(inputs, 32, [3, 3], scope='Conv2d_2a_3x3')
            # 147 x 147 x 32
            net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_2b_3x3')
            # 147 x 147 x 64
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
            # 73 x 73 x 64
            net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_1x1')
            # 73 x 73 x 80.
            net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')
            # 71 x 71 x 192.
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_5a_3x3')
            # 35 x 35 x 192.

        # Inception blocks
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            # mixed: 35 x 35 x 256.
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # mixed_1: 35 x 35 x 288.
            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv_1_0c_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # mixed_2: 35 x 35 x 288.
            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # mixed_3: 17 x 17 x 768.
            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)

            # mixed4: 17 x 17 x 768.
            with tf.variable_scope('Mixed_6b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # mixed_5: 17 x 17 x 768.
            with tf.variable_scope('Mixed_6c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # mixed_6: 17 x 17 x 768.
            with tf.variable_scope('Mixed_6d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # mixed_7: 17 x 17 x 768.
            with tf.variable_scope('Mixed_6e'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points['Mixed_6e'] = net

            # mixed_8: 8 x 8 x 1280.
            with tf.variable_scope('Mixed_7a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            # mixed_9: 8 x 8 x 2048.
            with tf.variable_scope('Mixed_7b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # mixed_10: 8 x 8 x 2048.
            with tf.variable_scope('Mixed_7c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')], 3)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            return net, end_points


def inception_v3(inputs,
                 num_classes=12,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV3'):
    with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes],
                           reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):

            net, end_points = inception_v3_base(inputs, scope=scope)

            # Auxiliary Head logits
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                aux_logits = end_points['Mixed_6e']
                with tf.variable_scope('AuxLogits'):
                    aux_logits = slim.avg_pool2d(
                        aux_logits, [5, 5], stride=3, padding='VALID',
                        scope='AvgPool_1a_5x5')
                    aux_logits = slim.conv2d(aux_logits, 128, [1, 1],
                                             scope='Conv2d_1b_1x1')

                    # Shape of feature map before the final layer.
                    aux_logits = slim.conv2d(
                        aux_logits, 768, [3, 3],
                        weights_initializer=trunc_normal(0.01),
                        padding='VALID', scope='Conv2d_2a_5x5')
                    aux_logits = slim.conv2d(
                        aux_logits, num_classes, [1, 1], activation_fn=None,
                        normalizer_fn=None, weights_initializer=trunc_normal(0.001),
                        scope='Conv2d_2b_1x1')
                    # golbal_pool???
                    aux_logits = slim.avg_pool2d(aux_logits, [np.shape(aux_logits)[1], np.shape(aux_logits)[2]],
                                                 padding='VALID',
                                                 scope='AvgPool_1a_ALxAL')
                    if spatial_squeeze:
                        aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
                    end_points['AuxLogits'] = aux_logits
                    end_points['AuxPredictions'] = prediction_fn(aux_logits, scope='Predictions')

            # Final pooling and prediction
            with tf.variable_scope('Logits'):

                net = slim.avg_pool2d(net, [np.shape(net)[1], np.shape(net)[2]], padding='VALID',
                                      scope='AvgPool_1a_LxL')
                # 1 x 1 x 2048
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                end_points['PreLogits'] = net
                # 2048
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                # 13
            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points


def inception_v3_arg_scope(weight_decay=0.00004,
                           stddev=0.1,
                           batch_norm_var_collection='moving_vars'):
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=trunc_normal(stddev),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params) as sc:
            return sc

def softmax_loss(predicts, labels):
    predicts = tf.reshape(predicts, [trainnum, classnum])
    labels = tf.one_hot(labels, classnum)
    # loss=-tf.reduce_sum(labels*tf.log(predicts))
    loss = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(predicts, 1e-10, 1.0)))
    return loss


def optimer(loss, lr=0.001):
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    return train_step

global first
first = True


class inception_v3_model(object):

    def inference(self, inputs,
                 num_classes=12,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV3'):
        
        global first
        if first == False:
            reuse = True
        first = False
        inputs = tf.cast(inputs, tf.float32) / 255.0
        return inception_v3(inputs, num_classes, is_training,dropout_keep_prob=dropout_keep_prob
                            ,prediction_fn = prediction_fn,spatial_squeeze = spatial_squeeze, reuse=reuse, scope=scope)



image, label = getinputs(trainpath)
batch_image, batch_label = get_batch(image, label, trainnum, 0)

inceptionV3 = inception_v3_model()
with slim.arg_scope(inception_v3_arg_scope()):
  logits, end_points = inceptionV3.inference(batch_image, classnum,is_training=True)
  
batch_images =  tf.placeholder(tf.float32, shape= [None, 64, 64, 1])

loss = 0.3*softmax_loss(end_points['Predictions'], batch_label) + 0.7*softmax_loss(end_points['AuxPredictions'], batch_label)

opti = optimer(loss, learnrate)

test_image, test_label = getinputs(testpath)
test_image_batch, test_label_batch = get_test_batch(test_image, test_label, testnum)
'''
with slim.arg_scope(resnet_arg_scope(is_training=False)):
    _, test_inf = ResNetModel.inference(test_image_batch, classnum)

'''
with slim.arg_scope(inception_v3_arg_scope()):
    _, test_inf = inceptionV3.inference(test_image_batch, classnum)
    


test_labels = tf.one_hot(test_label_batch, classnum)
test_pre = tf.reshape(test_inf['Predictions'], [testnum, classnum])
correct_prediction = tf.equal(tf.argmax(test_pre, 1), tf.argmax(test_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_pre = tf.argmax(test_pre, 1)
test_true = tf.argmax(test_labels, 1)

# valid_set
valid_image, valid_labels = getinputs(validpath)
valid_image_batch, valid_label_batch = get_test_batch(valid_image, valid_labels, validnum)
with slim.arg_scope(inception_v3_arg_scope()):
    _, valid_infer = inceptionV3.inference(valid_image_batch, classnum)

valid_labels = tf.one_hot(valid_label_batch, classnum)
valid_pre = tf.reshape(valid_infer['Predictions'], [validnum, classnum])
valid_correct_prediction = tf.equal(tf.argmax(valid_pre, 1), tf.argmax(valid_labels, 1))
valid_accuracy = tf.reduce_mean(tf.cast(valid_correct_prediction, tf.float32))
valid_pre = tf.argmax(valid_pre, 1)
valid_true = tf.argmax(valid_labels, 1)


target_names = ['class sg', 'class bm', 'class wd', 'class wt', 'class wj', 'class wo', 'class ym', 'class shq', 'class shj',
                'class no', 'class yh', 'class fb']
init = tf.initialize_all_variables()
config=tf.ConfigProto()
config.gpu_options.allow_growth=True

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
                [loss, opti, batch_label, batch_image, end_points['Predictions']],feed_dict=feed_dict)
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
