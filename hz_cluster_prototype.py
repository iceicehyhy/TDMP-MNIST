import tensorflow as tf
import numpy as np
from network import *
import random
import time
from PIL import Image
import os
import pickle
from sklearn.cluster import KMeans
import argparse
from read_dataset import *
from tensorflow import keras

num_classes = 10

ds_file = '/home/iceicehyhy/Dataset/MNIST_224X224_3/pairs_train.txt'
root_dir = '/home/iceicehyhy/Dataset/MNIST_224X224_3/train'
"""
net:model architecture
train_images: train_ds in [60000, 224, 224, 3]
train_labels: labels in [60000]
weight_file: './cluster_prototype_p4_r1__initialization'
"""

def cluster_prototype(net, weight_file, sess):
    batch_size = 64
    
    # create a parameter for prototype weight in DIM: [10 x 4 x 7 x 7 x 512]
    # each prototype has a feature map of [7 x 7 x 512], i: number of classes, j: number of spatial activations
    prototype_weight = np.ndarray([net.class_num * net.prototype_num, 7, 7, 512]).astype(np.float32)
    
    sample_each_class, img_list = number_samples_in_class(ds_file)
    img_list.sort()

    # create prototype for each class
    
    for i in range(num_classes):
        print ("generating prototype for class: ", i)
        l = sample_each_class[i]
        # create a tensor with [60000 X 7 X 7 X 512]
        pool5s = np.ndarray([l, 7*7*512]).astype(np.float32)

        iter_num = l // batch_size if l % batch_size == 0 else l // batch_size + 1
        
        for t in range(iter_num):
            # start index
            index_s = t * batch_size
            # end index
            index_e = np.minimum((t + 1) * batch_size, l)
            for c_b in range (index_e - index_s):
                img_p, label_ = img_list[index_s + c_b]
                img_ = np.asarray(PIL_loader(os.path.join(root_dir, img_p)))
                img_ = np.expand_dims(img_, axis= 0)
                label_ = np.expand_dims(label_, axis=0)
                if c_b == 0:
                    batch_x = img_
                    batch_y = label_
                else:
                    batch_x = np.concatenate((batch_x, img_), axis= 0)
                    batch_y = np.concatenate((batch_y, label_), axis= 0)

            p5 = sess.run(net.pool5, feed_dict={net.x:batch_x, net.y:batch_y})
            pool5s[index_s:index_e, :] = np.reshape(p5, [-1, 7*7*512])

            #pool5s[index_s:index_e, :] /= np.linalg.norm(pool5s[index_s:index_e, :], axis=1, keepdims=True)

        # reshape to L x 7x7x512
        pool5s = np.reshape(pool5s, [l, 7*7*512])

        # find 4 cluster centres using KMeans
        kmeans = KMeans(n_clusters=net.prototype_num).fit(pool5s)

        centers = kmeans.cluster_centers_

        centers = np.reshape(centers, [net.prototype_num, 7, 7, 512])

        prototype_weight[i * net.prototype_num : (i + 1) * net.prototype_num, :, :, :] = centers

    with open(weight_file, 'wb') as jar:
        pickle.dump(prototype_weight, jar)


def cluster():
    # create the network model
    net = VGG_ATTENTION_Prototype(num_classes, FLAGS.p, FLAGS.r, add_vc_loss=True, distance_kind='euclidean', normalize_before_attention=True, use_threshold=True)
    
    pre_trained_model = keras.models.load_model(FLAGS.pre_trained_model)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # load vgg weights
    net.load_vgg_weights_from_model(pre_trained_model, sess)
    # load feature dictionary
    net.load_VCs(FLAGS.dictpath, sess)

    # get train dataset, not possible to retrieve train data as a whole, as the limit of memory

    # cluster_prototype
    cluster_prototype(net, './cluster_prototype_p' + str(FLAGS.p) + '_r' + str(FLAGS.r) + '_initialization', sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=4, help='prototype number')
    parser.add_argument('--r', type=int, default=1, help='recurrence number')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--imgpath', type=str, default=None, help='image dataset file path')
    parser.add_argument('--vggpath', type=str, default='./vgg16_weights.npz', help='pre-trained VGG-16 file path')
    parser.add_argument('--dictpath', type=str, default='./dictionary_mnist_VGG_pool4_K512_vMFMM30.pickle', help='feature dictionary file path')
    parser.add_argument('--pre_trained_model', type=str, default='/home/iceicehyhy/weight/VGG16_MNIST/vgg16_mnist_224_adam_b64.h5')
    FLAGS, unparsed = parser.parse_known_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    cluster()
