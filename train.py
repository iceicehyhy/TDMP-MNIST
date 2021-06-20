import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from network import *
import random
import time
from PIL import Image
import os
import pickle
import argparse
from GetImage import getimage


num_classes = 10

def train_stage(net, train_data, test_data, epoch, batch_size, learning_rate, sess, train_var_list, save_weight=False, save_weight_name=None, save_weight_epoch=10):
    # train_images: [N X 224 X 224 X 3]
    train_images = train_data[0]
    # train labels: [N]
    train_labels = train_data[1]

    # test_images: [N X 224 X 224 X 3]
    test_images = test_data[0]
    # test_labels: [N]
    test_labels = test_data[1]

    train_num = train_images.shape[0]
    test_num = test_images.shape[0]

    # iteration number
    iter_num = train_num // batch_size if train_num % batch_size == 0 else train_num // batch_size + 1

    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    boundaries = [30, 60, 90]
    learning_rates = [learning_rate, learning_rate / 10, learning_rate / 100, learning_rate / 1000]

    ep = tf.placeholder(tf.int32)

    lr = tf.train.piecewise_constant(ep, boundaries=boundaries, values=learning_rates)

    # MomentumOptimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

    # train_optimizer
    train_op = optimizer.minimize(net.loss, var_list=train_var_list)
    sess.run(tf.variables_initializer(optimizer.variables()))
    #train_op = tf.train.AdamOptimizer(learning_rate).minimize(net.loss)

    for e in range(epoch):
        index = np.array(range(train_num)).astype(np.int32)
        np.random.shuffle(index)
        time1 = time.time()
        loss = 0.0

        for i in range(iter_num):
            #print("iter:" + str(i))
            index_s = i * batch_size
            index_e = min((i + 1) * batch_size, train_num)
            batch_x = train_images[index[index_s : index_e]]
            batch_y = train_labels[index[index_s : index_e]]
            result = sess.run([train_op, net.loss], feed_dict={net.x:batch_x, net.y:batch_y, ep:e})
            loss += result[1] * (index_e - index_s)

        loss /= train_num
        time2 = time.time()

        f = open('./log_epoch_all_a1e-4_prototype' + str(FLAGS.p) + '_recurrent' + str(FLAGS.r), 'a')
        print('epoch {}: training: loss --> {:.3f}, time: {:.3f} minutes'.format(e + 1, loss, (time2- time1) / 60.0))
        print('epoch {}: training: loss --> {:.3f}, time: {:.3f} minutes'.format(e + 1, loss, (time2- time1) / 60.0), file=f)

        if save_weight and (e + 1) % save_weight_epoch == 0:
            net.save_all_weights(save_weight_name + str(int(e + 1)) + '.ckpt', sess)
            train_acc = test_accuracy(net, train_data, batch_size, sess)
            test_acc = test_accuracy(net, test_data, batch_size, sess)
            print('train acc: {:.3f}, test acc: {:.3f}'.format(train_acc, test_acc))
            print('train acc: {:.3f}, test acc: {:.3f}'.format(train_acc, test_acc), file=f)

        f.close()


def test_accuracy(net, data, batch_size, sess):
    images = data[0]
    labels = data[1]
    test_num = images.shape[0]

    iter_num = test_num // batch_size if test_num % batch_size == 0 else test_num // batch_size + 1

    acc = 0

    index = np.array(range(test_num)).astype(np.int32)
    np.random.shuffle(index)

    for i in range(iter_num):
        index_s = i * batch_size
        index_e = min((i + 1) * batch_size, test_num)
        batch_x = images[index[index_s : index_e]]
        batch_y = labels[index[index_s : index_e]]
        acc += sess.run(net.acc, feed_dict={net.x:batch_x, net.y:batch_y})
    acc /= test_num

    return acc

# training on MNIST dataset
def train():
    """ 
    num_classes = 10  
    p = 4  (prototype number)
    r = 1  (recurrence number)
    add_vc_loss = True
    distance = 'EUCLIDEAN'
    normalize b4 attention = True
    use Thres  = True
    """
    net = VGG_ATTENTION_Prototype(num_classes, FLAGS.p, FLAGS.r, add_vc_loss=True, distance_kind='euclidean', normalize_before_attention=True, use_threshold=True)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # load pre-trained weight for VGG16 on MNIST dataset
    net.load_vgg_weights(FLAGS.vggpath, sess)
    # load pre-build feature dictionary for VGG16-MNIST
    net.load_VCs(FLAGS.dictpath, sess)

    #tf.summary.FileWriter('./log', sess.graph)

    # get images in [N X 224 x 224 x 3], labels in [N]
    train_images, train_labels = getimage('train', FILE_PATH=FLAGS.imgpath)
    test_images, test_labels = getimage('test', occ_level='ZERO', occ_type='', FILE_PATH=FLAGS.imgpath)

    # load prototypes, need to generate prototype first
    net.load_prototypes('./cluster_prototype_p' + str(FLAGS.p) + '_r' + str(FLAGS.r) + '_initialization', sess)

    # recurrence >= 3
    if FLAGS.r >= 3 or (FLAGS.r >= 2 and FLAGS.p > 4):
        batch_size = 32
    else:
        batch_size = 64

    # learning rate & epoch
    learning_rate_prototype = 1e-3
    learning_rate_all = 1e-4
    prototype_epoch = 10
    epoch = FLAGS.epoch  # 10


    print("begin train")

    # train prototype first
    print('train prototype')
    f = open('./log_epoch_all_a1e-4_prototype' + str(FLAGS.p) + '_recurrent' + str(FLAGS.r), 'a')
    print('train prototype', file=f)
    f.close()

    train_var_list = [var for var in tf.trainable_variables() if 'prototype' in var.name]
    """
    net: VGG_ATTENTION_Prototype
    train_images:
    """
    train_stage(net, [train_images, train_labels], [test_images, test_labels], prototype_epoch, batch_size, learning_rate_prototype, sess, train_var_list, save_weight=True, save_weight_name='prototype' + str(FLAGS.p) + '_recurrent' + str(FLAGS.r) + '_train_1e-3_weight_')

    # train all
    print('all train')

    f = open('./log_epoch_all_a1e-4_prototype' + str(FLAGS.p) + '_recurrent' + str(FLAGS.r), 'a')
    print('all train', file=f)
    f.close()

    train_var_list = [var for var in tf.trainable_variables()]
    train_stage(net, [train_images, train_labels], [test_images, test_labels], epoch, batch_size, learning_rate_all, sess, train_var_list, save_weight=True, save_weight_name='all_train_prototype' + str(FLAGS.p) + '_recurrent' + str(FLAGS.r) + '_1e-4_weight_')

    print("training end")

    sess.close()


if __name__ == '__main__':
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=4, help='prototype number')    # prototype number = 4
    parser.add_argument('--r', type=int, default=1, help='recurrence number')   # recurrence = 1
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')     # gpu id = 0
    parser.add_argument('--epoch', type=int, default=10, help='finetune epoch number')  # finetune epoch number
    parser.add_argument('--imgpath', type=str, default=None, help='image dataset file path')  # dataset path = None
    parser.add_argument('--vggpath', type=str, default='./vgg16_weights.npz', help='pre-trained VGG-16 file path')   # vgg pretrained-weight = ...
    parser.add_argument('--e', type=str, default='./dictionary_mnist_VGG_pool4_K512_vMFMM30.pickle', help='feature dictionary file path') # dictionary path
    
    FLAGS, unparsed = parser.parse_known_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    train()
