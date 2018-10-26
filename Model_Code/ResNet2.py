import tensorflow as tf
import collections
import os
# from PIL import Image
import random
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix

slim = tf.contrib.slim

global first
first = True

classnum = 12
testnum = tf.placeholder(tf.int32)
trainnum = tf.placeholder(tf.int32)
validnum = tf.placeholder(tf.int32)
learnrate = tf.placeholder(tf.float32)

path = r'C:\workspace\优化后\model\ResNet2.ckpt'
trainpath = [r'C:\workspace\优化后\Alexnet cnn_f Densent数据集\train.tfrecords',
             r'C:\workspace\优化后\Alexnet cnn_f Densent数据集\train_001.tfrecords']
testpath = [r'C:\workspace\优化后\Alexnet cnn_f Densent数据集\test.tfrecords',
            r'C:\workspace\优化后\Alexnet cnn_f Densent数据集\test_001.tfrecords']
validpath = [r'C:\workspace\优化后\Alexnet cnn_f Densent数据集\validation.tfrecords']


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
    image = tf.reshape(image, [64, 64, 1])
    return image, label


def get_batch(image, label, batch_size, crop_size):
    # print(image.shape)
    # print(label.shape)
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=batch_size, num_threads=10, capacity=10000,
                                            min_after_dequeue=100)
    return images, tf.reshape(labels, [batch_size])


def get_test_batch(image, label, batch_size):
    images,labels=tf.train.batch([image,label],batch_size=batch_size)
    #images, labels = tf.train.shuffle_batch([image, label],
    #                                        batch_size=batch_size, num_threads=10, capacity=10000,
    #                                        min_after_dequeue=200)
    return images, tf.reshape(labels, [batch_size])


def get_valid_batch(image, label, batch_size):
    # images,labels=tf.train.batch([image,label],batch_size=batch_size)
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=batch_size, num_threads=10, capacity=10000,
                                            min_after_dequeue=200)
    return images, tf.reshape(labels, [batch_size])


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.

    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ResNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ResNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride) tuple for each unit in the
        block to serve as argument to unit_fn.
    """


def subsample(inputs, factor, scope=None):
    """Subsamples the input along the spatial dimensions.

    Args:
      inputs: A `Tensor` of size [batch, height_in, width_in, channels].
      factor: The subsampling factor.
      scope: Optional variable_scope.

    Returns:
      output: A `Tensor` of size [batch, height_out, width_out, channels] with the
        input, either intact (if factor == 1) or subsampled (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    """Strided 2-D convolution with 'SAME' padding.

    When stride > 1, then we do explicit zero-padding, followed by conv2d with
    'VALID' padding.

    Note that

       net = conv2d_same(inputs, num_outputs, 3, stride=stride)

    is equivalent to

       net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
       net = subsample(net, factor=stride)

    whereas

       net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

    is different when the input's height or width is even, which is why we add the
    current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

    Args:
      inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
      num_outputs: An integer, the number of output filters.
      kernel_size: An int with the kernel_size of the filters.
      stride: An integer, the output stride.
      rate: An integer, rate for atrous convolution.
      scope: Scope.

    Returns:
      output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           padding='SAME', scope=scope)
    else:
        # kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           padding='VALID', scope=scope)


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):
    """Bottleneck residual unit variant with BN before convolutions.

    This is the full preactivation residual unit variant proposed in [2]. See
    Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
    variant which has an extra bottleneck layer.

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.

    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.

    Returns:
      The ResNet unit's output.
    """
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks,
                       outputs_collections=None):
    """Stacks ResNet `Blocks` and controls output feature density.

    First, this function creates scopes for the ResNet in the form of
    'block_name/unit_1', 'block_name/unit_2', etc.


    Args:
      net: A `Tensor` of size [batch, height, width, channels].
      blocks: A list of length equal to the number of ResNet `Blocks`. Each
        element is a ResNet `Block` object describing the units in the `Block`.
      outputs_collections: Collection to add the ResNet block outputs.

    Returns:
      net: Output tensor

    """
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net,
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)

            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    """Defines the default ResNet arg scope.

    TODO(gpapan): The batch-normalization related default values above are
      appropriate for use in conjunction with the reference ResNet models
      released at https://github.com/KaimingHe/deep-residual-networks. When
      training ResNets from scratch, they might need to be tuned.

    Args:
      is_training: Whether or not we are training the parameters in the batch
        normalization layers of the model.
      weight_decay: The weight decay to use for regularizing the model.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      batch_norm_epsilon: Small constant to prevent division by zero when
        normalizing activations by their variance in batch normalization.
      batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
        activations in the batch normalization layer.

    Returns:
      An `arg_scope` to use for the resnet models.
    """
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            # The following implies padding='SAME' for pool1, which makes feature
            # alignment easier for dense prediction tasks. This is also used in
            # https://github.com/facebook/fb.resnet.torch. However the accompanying
            # code of 'Deep Residual Learning for Image Recognition' uses
            # padding='VALID' for pool1. You can switch to that choice by setting
            # slim.arg_scope([slim.max_pool2d], padding='VALID').
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              include_root_block=True,
              reuse=None,
              scope=None):
    """Generator for v2 (preactivation) ResNet models.

    This function generates a family of ResNet v2 models. See the resnet_v2_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce ResNets of various depths.


    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      blocks: A list of length equal to the number of ResNet blocks. Each element
        is a resnet_utils.Block object describing the units in the block.
      num_classes: Number of predicted classes for classification tasks. If None
        we return the features before the logit layer.
      include_root_block: If True, include the initial convolution followed by
        max-pooling, if False excludes it. If excluded, `inputs` should be the
        results of an activation-less convolution.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.


    Returns:
      net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
        If global_pool is False, then height_out and width_out are reduced by a
        factor of output_stride compared to the respective height_in and width_in,
        else both height_out and width_out equal one. If num_classes is None, then
        net is the output of the last ResNet block, potentially after global
        average pooling. If num_classes is not None, net contains the pre-softmax
        activations.
      end_points: A dictionary from components of the network to the corresponding
        activation.

    Raises:
      ValueError: If the target output_stride is not valid.
    """
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             stack_blocks_dense],
                            outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:
                # We do not include batch normalization or activation functions in conv1
                # because the first ResNet unit will perform these. Cf. Appendix of [2].
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(net, 32, 3, stride=1, scope='conv1')
                # net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')
            net = stack_blocks_dense(net, blocks)
            # This is needed because the pre-activation variant does not have batch
            # normalization or activation functions in the residual unit output. See
            # Appendix of [2].
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            if global_pool:
                # Global average pooling.
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')
            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions')
            return net, end_points


class resnet_v2_30(object):

    def inference(self, inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_30'):
        global first

        if first == False:
            reuse = True

        first = False

        inputs = tf.cast(inputs, tf.float32) / 255.0
        blocks = [
            Block('block1', bottleneck, [(128, 64, 1)] * 3 + [(128, 64, 2)]),
            Block('block2', bottleneck, [(256, 128, 1)] * 3 + [(256, 128, 2)]),
            Block('block3', bottleneck, [(512, 256, 1)] * 4 + [(512, 256, 2)])
            #Block('block5', bottleneck, [(1024, 512, 1)] * 6 + [(1024, 512, 2)])
        ]
        return resnet_v2(inputs, blocks, num_classes, global_pool,
                         include_root_block=True, reuse=reuse, scope=scope)


def softmax_loss(predicts, labels):
    predicts = tf.reshape(predicts, [trainnum, classnum])
    labels = tf.one_hot(labels, classnum)
    # loss=-tf.reduce_sum(labels*tf.log(predicts))
    loss = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(predicts, 1e-10, 1.0)))
    return loss


def optimer(loss, lr=0.001):
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    return train_step


image, label = getinputs(trainpath)
batch_image, batch_label = get_batch(image, label, trainnum, 0)

ResNetModel = resnet_v2_30()
net, end_points = ResNetModel.inference(batch_image, classnum)
loss = softmax_loss(end_points['predictions'], batch_label)
opti = optimer(loss, learnrate)

test_image, test_label = getinputs(testpath)
test_image_batch, test_label_batch = get_test_batch(test_image, test_label, testnum)

_, test_inf = ResNetModel.inference(test_image_batch, classnum)

test_labels = tf.one_hot(test_label_batch, classnum)
test_pre = tf.reshape(test_inf['predictions'], [testnum, classnum])
correct_prediction = tf.equal(tf.argmax(test_pre, 1), tf.argmax(test_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_pre = tf.argmax(test_pre, 1)
test_true = tf.argmax(test_labels, 1)

# valid_set
valid_image, valid_labels = getinputs(validpath)
valid_image_batch, valid_label_batch = get_test_batch(valid_image, valid_labels, validnum)
_, valid_infer = ResNetModel.inference(valid_image_batch, classnum)
valid_labels = tf.one_hot(valid_label_batch, classnum)
valid_pre = tf.reshape(valid_infer['predictions'], [validnum, classnum])
valid_correct_prediction = tf.equal(tf.argmax(valid_pre, 1), tf.argmax(valid_labels, 1))
valid_accuracy = tf.reduce_mean(tf.cast(valid_correct_prediction, tf.float32))
valid_pre = tf.argmax(valid_pre, 1)
valid_true = tf.argmax(valid_labels, 1)

from sklearn.metrics import classification_report

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
                [loss, opti, batch_label, batch_image, end_points['predictions']],feed_dict=feed_dict)
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
        if valid_loop > 0:
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
        
        
train(train_num=128,loop_count=1002)
test_and_valid(10,10,200,200)
predict_time(1000)
