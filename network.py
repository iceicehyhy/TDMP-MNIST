
import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()

class VGG_ATTENTION_Prototype:
    def __init__(self, class_num, prototype_num, recurrent_step, normalize_feature_map=False, use_threshold=True, add_vc_loss=True, distance_kind='euclidean', normalize_before_attention=True, use_sigmoid=False):
        
        self.class_num = class_num   # class_num = 10 for MNIST dataset
        self.prototype_num = prototype_num  # prototype number = 4
        self.recurrent_step = recurrent_step # recurrence step = 1
        self.normalize_feature_map = normalize_feature_map  # normalize feature map = False
        self.use_threshold = use_threshold # use_threshold = True
        self.distance_kind = distance_kind # euclidean
        self.normalize_before_attention = normalize_before_attention   # normalize b4 attention = True
        self.use_sigmoid = use_sigmoid  # use_sigmoid = False

        self.x = tf.placeholder(tf.float32, [None, 224, 224, 3])   # create a image input - placeholder with: NONE  X 224 x 224 x 3
        self.y = tf.placeholder(tf.int32, [None])  # create a label - placeholder with: NONE

        # parameter gama: control the hardness of probability assignment
        self.gama = tf.get_variable('gama', dtype=tf.float32, initializer=tf.constant(20.0, dtype=tf.float32))  

        """
        distance_class: a tensor with dim [-1, 10]. the maximum value of protoypes in all classes
        assginment: the predicted label in int32       
        pr_loss: prototype loss
        vc_loss: feature dictionary loss
        """
        # pass placeholder of x and y to network, and return necessary parameters
        self.distances_class, self.assignment, self.pr_loss, self.vc_loss = self.network(self.x, self.y)

        # the weight of each type of losses
        alpha1 = 1
        alpha2 = 1

        # the corss-entropy loss
        self.loss = self.dce_loss(self.distances_class, self.gama, self.y)

        # add prototype loss
        self.loss += alpha1 * self.pr_loss

        # tf.add_n performs the element-wise addition
        self.loss += 5e-4 * tf.add_n(tf.get_collection('losses'))

        # feature dictionary loss
        if add_vc_loss:
            self.loss += alpha2 * self.vc_loss

        # number of correct predictions
        self.acc = self.compute_acc(self.distances_class, self.y)

    # CONV layers
    def Conv(self, x, scope_name, shape, stride=[1, 1, 1, 1], activation=True, vgg_parameters=False, all_parameters=False, weight_decay=True):
        # intializer - truncated normal distribution
        initer = tf.truncated_normal_initializer(stddev=0.01)

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weight', dtype=tf.float32, shape=shape, initializer=initer)
            conv = tf.nn.conv2d(x, kernel, stride, padding='SAME')
            bias = tf.get_variable('bias', dtype=tf.float32, initializer=tf.constant(0.0, shape=[shape[3]], dtype=tf.float32))
            out = tf.nn.bias_add(conv, bias)

            if activation:
                out = tf.nn.relu(out)

            if vgg_parameters:
                self.vgg_parameters += [kernel, bias]

            if all_parameters:
                self.all_parameters += [kernel, bias]

            if weight_decay:
                tf.add_to_collection('losses', tf.nn.l2_loss(kernel))

        return out

    # network architecture
    """
    x: input image
    y: input label
    """
    def network(self, x, y):
        self.vgg_parameters = []
        self.all_parameters = []

        # zero-mean input
        with tf.variable_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = x - mean

        initer = tf.truncated_normal_initializer(stddev=0.01)

        # 64 x 224 x 224
        self.conv1_1 = self.Conv(images, 'conv1_1', [3, 3, 3, 64], vgg_parameters=True, all_parameters=True)

        # 
        self.conv1_2 = self.Conv(self.conv1_1, 'conv1_2', [3, 3, 64, 64], vgg_parameters=True, all_parameters=True)

        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        self.conv2_1 = self.Conv(self.pool1, 'conv2_1', [3, 3, 64, 128], vgg_parameters=True, all_parameters=True)

        self.conv2_2 = self.Conv(self.conv2_1, 'conv2_2', [3, 3, 128, 128], vgg_parameters=True, all_parameters=True)

        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        self.conv3_1 = self.Conv(self.pool2, 'conv3_1', [3, 3, 128, 256], vgg_parameters=True, all_parameters=True)

        self.conv3_2 = self.Conv(self.conv3_1, 'conv3_2', [3, 3, 256, 256], vgg_parameters=True, all_parameters=True)

        self.conv3_3 = self.Conv(self.conv3_2, 'conv3_3', [3, 3, 256, 256], vgg_parameters=True, all_parameters=True)

        self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        self.conv4_1 = self.Conv(self.pool3, 'conv4_1', [3, 3, 256, 512], vgg_parameters=True, all_parameters=True)

        self.conv4_2 = self.Conv(self.conv4_1, 'conv4_2', [3, 3, 512, 512], vgg_parameters=True, all_parameters=True)

        self.conv4_3 = self.Conv(self.conv4_2, 'conv4_3', [3, 3, 512, 512], vgg_parameters=True, all_parameters=True)

        # N x 14 x 14 x 512
        self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        # VGG 16 pool4 layer
        self.original_pool4 = self.pool4

        #self.attentions_without_vc = tf.norm(self.original_pool4, axis=3)
        #fa = tf.reshape(self.attentions_without_vc, [-1, 14*14])
        #pl = 0.8
        #pu = 0.2
        #pln = int(14*14*pl)
        #pun = int(14*14*pu)
        #thl = tf.nn.top_k(fa, pln).values[:, -1]
        #thu = tf.nn.top_k(fa, pun).values[:, -1]
        #thl = tf.reshape(thl, [-1,1,1])
        #self.attentions_without_vc = tf.nn.relu(self.attentions_without_vc - thl) + thl
        #thu = tf.reshape(thu, [-1,1,1])
        #thu = tf.tile(thu, [1,14,14])
        #self.attentions_without_vc = tf.clip_by_value(self.attentions_without_vc, 0, thu)
        #self.attentions_without_vc = self.attentions_without_vc / thu

        vc_loss = 0
        
        # recurrence step = 1
        for i in range(self.recurrent_step):
            # visual concept attention
            with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
                # VCs in dim of [1 x 1 x 512 x 512] AKA feature dictionary, [1x1] convolution with input/output channel = 512
                VCs = tf.get_variable('visual_concept', dtype=tf.float32, shape=[1, 1, 512, 512], initializer=initer)
                if i == 0:
                    self.all_parameters += [VCs]
                    self.VCs = VCs

                # l2 normalize dim=[2], tensor size unchanged
                VCs = tf.nn.l2_normalize(VCs, dim=[2])

                # normalzie pool4 layer in dim[3], pool4 in dim = []
                pool4_n = tf.nn.l2_normalize(self.pool4, dim=[3])
                # FALSE
                if self.normalize_feature_map:
                    similarity = tf.nn.conv2d(pool4_n, VCs, [1, 1, 1, 1], padding='SAME')
                    similarity_normalize = similarity
                else:
                    # tf.nn.conv2d(input, filters, strides, padding, data_format='NHWC', dilations=None,name=None)
                    # similarity is the similarity score of each position on feature map L, w.r.t 512 object centres
                    similarity = tf.nn.conv2d(self.pool4, VCs, [1, 1, 1, 1], padding='SAME')
                    similarity_normalize = tf.nn.conv2d(pool4_n, VCs, [1, 1, 1, 1], padding='SAME')
                
                # get the max along dim = [3], N X 14 X 14 X 512 --> N X 14 X 14
                # get the max similarity score at each pixel, w.r.t to 512 object centres
                self.attentions = tf.reduce_max(similarity, reduction_indices=[3])
                self.attentions_not_normalize = self.attentions

                # vc: visual_concept, AKA feature dictionary. 

                # returns the index with the largest value across dim = 3
                vc_assign = tf.argmax(similarity_normalize, axis=3)
                vcs = tf.transpose(tf.reshape(VCs, [512, 512]), [1, 0])
                vc_loss += tf.reduce_mean(tf.square(tf.gather(vcs, vc_assign) - pool4_n) / 2)

                if self.use_threshold:
                    # conver to N X 196
                    flatten_attention = tf.reshape(self.attentions, [-1, 14 * 14])
                    percentage_l = 0.8
                    percentage_u = 0.2
                    
                    
                    percentage_ln = int(14 * 14 * percentage_l) # 156
                    percentage_un = int(14 * 14 * percentage_u) # 39
                    # tok_k: Finds values and indices of the k largest entries for the last dimension.
                    threshold_l = tf.nn.top_k(flatten_attention, percentage_ln).values[:, -1]   # get the lower limit, the minimum of the top 156(80%) scores
                    threshold_u = tf.nn.top_k(flatten_attention, percentage_un).values[:, -1]   # get the upper limit, the minimum of the top 39(20%) scores

                    # reshape it to [1,1,1]
                    threshold_l = tf.reshape(threshold_l, [-1, 1, 1])
                    # take relu of the diff between (similarity scores and lower threshold), for scores below lower threshold will be flatten to 0
                    # relu + threshold_l: for values below threshold_l, just force them to be threshold_l, for scores >= threshold_l: attention = scores
                    # max(ai,j ; al);
                    attentions = tf.nn.relu(self.attentions - threshold_l) + threshold_l

                    # reshape to [1,1,1]
                    threshold_u = tf.reshape(threshold_u, [-1, 1, 1])
                    # replicate the tensor along the dim specified: [1, 14, 14] multiple times
                    threshold_u = tf.tile(threshold_u, [1, 14, 14])

                    # clip it by min:0 and max:threshold_u
                    # min(0, au)
                    attentions = tf.clip_by_value(attentions, 0, threshold_u)

                    attentions = attentions / threshold_u
                    self.attentions = attentions
                # FALSE
                elif self.use_sigmoid:
                    self.attentions = tf.nn.sigmoid(self.attentions - tf.reduce_mean(self.attentions, reduction_indices=[1,2], keepdims=True))
                else:
                    # normalize the attention map by attention/max(attention), because the scores could be large
                    self.attentions = self.attentions / tf.reduce_max(self.attentions, reduction_indices=[1,2], keepdims=True)

            # apply attentions to pool1
            # p1: [N, 112, 112, 64] --> [N, 14, 8, 14, 8, 64]
            pool1 = tf.reshape(self.pool1, [-1, 14, 8, 14, 8, 64])
            # attentions: [N,14,14] --> [N, 14, 1, 14, 1, 1]
            attentions = tf.reshape(self.attentions, [-1, 14, 1, 14, 1, 1])
            # apply attention to all features @ p1
            pool1 = pool1 * attentions
            new_pool1 = tf.reshape(pool1, [-1, 112, 112, 64])

            # feed-forward again
            self.conv2_1 = self.Conv(new_pool1, 'conv2_1', [3, 3, 64, 128], vgg_parameters=False, all_parameters=False)

            self.conv2_2 = self.Conv(self.conv2_1, 'conv2_2', [3, 3, 128, 128], vgg_parameters=False, all_parameters=False)

            self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

            self.conv3_1 = self.Conv(self.pool2, 'conv3_1', [3, 3, 128, 256], vgg_parameters=False, all_parameters=False)

            self.conv3_2 = self.Conv(self.conv3_1, 'conv3_2', [3, 3, 256, 256], vgg_parameters=False, all_parameters=False)

            self.conv3_3 = self.Conv(self.conv3_2, 'conv3_3', [3, 3, 256, 256], vgg_parameters=False, all_parameters=False)

            self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

            self.conv4_1 = self.Conv(self.pool3, 'conv4_1', [3, 3, 256, 512], vgg_parameters=False, all_parameters=False)

            self.conv4_2 = self.Conv(self.conv4_1, 'conv4_2', [3, 3, 512, 512], vgg_parameters=False, all_parameters=False)

            self.conv4_3 = self.Conv(self.conv4_2, 'conv4_3', [3, 3, 512, 512], vgg_parameters=False, all_parameters=False)

            self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')


        self.new_pool4 = self.pool4

        #self.attentions_without_vc_new = tf.norm(self.original_pool4, axis=3)
        #fa = tf.reshape(self.attentions_without_vc_new, [-1, 14*14])
        #pl = 0.8
        #pu = 0.2
        #pln = int(14*14*pl)
        #pun = int(14*14*pu)
        #thl = tf.nn.top_k(fa, pln).values[:, -1]
        #thu = tf.nn.top_k(fa, pun).values[:, -1]
        #thl = tf.reshape(thl, [-1,1,1])
        #self.attentions_without_vc_new = tf.nn.relu(self.attentions_without_vc_new - thl) + thl
        #thu = tf.reshape(thu, [-1,1,1])
        #thu = tf.tile(thu, [1,14,14])
        #self.attentions_without_vc_new = tf.clip_by_value(self.attentions_without_vc_new, 0, thu)
        #self.attentions_without_vc_new = self.attentions_without_vc_new / thu

        # new visual concept attention
        with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
            VCs = tf.get_variable('visual_concept', dtype=tf.float32, shape=[1, 1, 512, 512], initializer=initer)
            if self.recurrent_step == 0:
                self.all_parameters += [VCs]
                self.VCs = VCs
            VCs = tf.nn.l2_normalize(VCs, dim=[2])

            pool4_n = tf.nn.l2_normalize(self.pool4, dim=[3])
            if self.normalize_feature_map:
                similarity = tf.nn.conv2d(pool4_n, VCs, [1, 1, 1, 1], padding='SAME')
                similarity_normalize = similarity
            else:
                similarity = tf.nn.conv2d(self.pool4, VCs, [1, 1, 1, 1], padding='SAME')
                similarity_normalize = tf.nn.conv2d(pool4_n, VCs, [1, 1, 1, 1], padding='SAME')

            self.attentions_new = tf.reduce_max(similarity, reduction_indices=[3], keepdims=True)
            self.attentions_new_not_normalize = self.attentions_new

            vc_assign = tf.argmax(similarity_normalize, axis=3)
            vcs = tf.transpose(tf.reshape(VCs, [512, 512]), [1, 0])
            vc_loss += tf.reduce_mean(tf.square(tf.gather(vcs, vc_assign) - pool4_n) / 2)

            if self.use_threshold:
                flatten_attention = tf.reshape(self.attentions_new, [-1, 14 * 14])
                percentage_l = 0.8
                percentage_u = 0.2
                percentage_ln = int(14 * 14 * percentage_l)
                percentage_un = int(14 * 14 * percentage_u)
                threshold_l = tf.nn.top_k(flatten_attention, percentage_ln).values[:, -1]
                threshold_u = tf.nn.top_k(flatten_attention, percentage_un).values[:, -1]

                threshold_l = tf.reshape(threshold_l, [-1, 1, 1, 1])
                attentions = tf.nn.relu(self.attentions_new - threshold_l) + threshold_l

                threshold_u = tf.reshape(threshold_u, [-1, 1, 1, 1])
                threshold_u = tf.tile(threshold_u, [1, 14, 14, 1])
                attentions = tf.clip_by_value(attentions, 0, threshold_u)

                attentions = attentions / threshold_u   # normalization
                self.attentions_new = attentions
            elif self.use_sigmoid:
                self.attentions_new = tf.nn.sigmoid(self.attentions_new - tf.reduce_mean(self.attentions_new, reduction_indices=[1,2], keepdims=True))
            else:
                self.attentions_new = self.attentions_new / tf.reduce_max(self.attentions_new, reduction_indices=[1,2], keepdims=True)

            self.attentions_new_pool4 = self.attentions_new

        # new_pool4 has the features after filtering

        # 3 conv layers with [3x3] kernels and 512 in/out channels
        self.conv5_1 = self.Conv(self.pool4, 'conv5_1', [3, 3, 512, 512], vgg_parameters=True, all_parameters=True)

        self.conv5_2 = self.Conv(self.conv5_1, 'conv5_2', [3, 3, 512, 512], vgg_parameters=True, all_parameters=True)

        self.conv5_3 = self.Conv(self.conv5_2, 'conv5_3', [3, 3, 512, 512], vgg_parameters=True, all_parameters=True)

        # pool5 layer, 7x7x512
        self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

        # tf.nn.avg_pool(input, ksize, strides, padding, data_format=None, name=None)
        # down_sample the attention map from [14x14] to [7x7] by avg pooling
        self.attentions_new = tf.nn.avg_pool(self.attentions_new, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # load the prototypes
        with tf.variable_scope('prototype', reuse=tf.AUTO_REUSE):
            self.prototypes = tf.get_variable('prototypes', dtype=tf.float32, shape=[self.class_num * self.prototype_num, 7, 7, 512])
            self.all_parameters += [self.prototypes]
            self.prototypes_load = self.prototypes

        # TRUE
        if self.normalize_before_attention:
            # l2_norm of p5
            self.pool5 = tf.nn.l2_normalize(self.pool5, dim=[1,2,3])
            # l2_norm of prototypes
            self.prototypes = tf.nn.l2_normalize(self.prototypes, dim=[1,2,3])

        # apply attention
        attentions = tf.reshape(self.attentions_new, [-1, 7, 7, 1])
        # each feature times the attention score
        features = self.pool5 * attentions

        # expand the dim from [40x7x7x512] to [1, 40, 7, 7, 512]
        prototypes = tf.expand_dims(self.prototypes, 0)

        # get batch size
        batch_size = tf.shape(features)[0]

        # replicate the prototypes batch_size times
        prototypes = tf.tile(prototypes, [batch_size, 1, 1, 1, 1])

        # apply attentions to prototypes
        attentions = tf.reshape(attentions, [-1, 1, 7, 7, 1])
        prototypes = prototypes * attentions

        # define the type of distance to use
        if self.distance_kind == 'cosine':
            # normalize
            features = tf.reshape(features, [-1, 7*7*512, 1])
            features = tf.nn.l2_normalize(features, dim=[1])
            prototypes = tf.reshape(prototypes, [-1, self.class_num * self.prototype_num, 7*7*512])
            prototypes = tf.nn.l2_normalize(prototypes, dim=[2])

            # compare distance
            distances = tf.matmul(prototypes, features)
            distances = tf.reshape(distances, [-1, self.class_num * self.prototype_num])

            # get nearest in each class
            distances_class = tf.reduce_max(tf.reshape(distances, [-1, self.class_num, self.prototype_num]), reduction_indices=[2])

            # get assignment
            assignment = tf.cast(tf.argmax(distances, 1), tf.int32)

        # euclidean is used for MNIST
        elif self.distance_kind == 'euclidean':
            # reshape the feature to [N, 7x7x512, 1]: 25088
            features = tf.reshape(features, [-1, 7*7*512, 1])

            # reshape the prototypes to [N, 10X4, 7X7X512]
            # each prototype is in dim of HxWxC, where i is the class index, and j is the orientation/spatial index
            prototypes = tf.reshape(prototypes, [-1, self.class_num * self.prototype_num, 7*7*512])

            # compute sum across a dimension, dim of un-specified colume remains the same
            """
            x2: [N,1,1]
            p2: [N,40,1]
            """
            x2 = tf.reduce_sum(tf.square(features), reduction_indices=[1,2], keepdims=True)  # square of the features, then take the sum of dim [1,2]
            p2 = tf.reduce_sum(tf.square(prototypes), reduction_indices=[2], keepdims=True)  # square of the prototypes, then take the sum of dim[2]

            # distance = x^2 + y^2 - 2xy,  equals to (x-y)^2
            # x2 + p2 = element-wise addtion
            # tf.matmul: matrix multiplication with output in dim : [N, 40, 1]
            # distances : [N, 40 ,1]
            distances = x2 + p2 - 2 * tf.matmul(prototypes, features)  # prototype x features = [N x 40 x 7x7x512] . [N x 7x7x512 x 1]
            distances = -distances      # take the negative of distances
            distances = tf.reshape(distances, [-1, self.class_num * self.prototype_num]) # reshape to [N, 40]

            # reshape to [N, 10, 4] and get max along dim [2]. return a tensor with [N, 10]
            # get the closest distance between 4 prototypes and features for different classes
            distances_class = tf.reduce_max(tf.reshape(distances, [-1, self.class_num, self.prototype_num]), reduction_indices=[2])

            # get assignment: cast tensor to new type - int32,  argmax: returns the index of the maximum value across a dim
            # index is in dim [N], each sample is associated with one index
            # assignment is actually the index of prototype assigned to the features [0, 40)
            assignment = tf.cast(tf.argmax(distances, 1), tf.int32)

        # prototype loss
        rindex = tf.range(batch_size) # create a arange(batch_size)
        rindex = tf.expand_dims(rindex, 1)  # [BATCH_SIZE] ---> [BATCH_SIZE, 1]
        assignment1 = tf.expand_dims(assignment, 1) # [N] ------> [N, 1]
        # concat two [N, 1] tensor to [N, 2] along dim = 1
        gather_index = tf.concat([rindex, assignment1], 1)
        # gather_nd: Gather slices from params into a Tensor with shape specified by indices.
        # tf.gather_nd(params, indices, batch_dims=0, name=None)
        """
        prototypes : [N, 40, 25088]
        gather_index: [N, 2]
        assigned_prototypes: [N, 25088]
        find the assigned prototype
        """
        assigned_prototype = tf.gather_nd(prototypes, gather_index)
        features = tf.reshape(features, [-1, 7*7*512])    #---->[N, 25088]
        # prototype loss: A prototype loss is added as the regularization of prototype learning
        # min(i,j){d(f(x); pi,j)}

        """
        squared diff sum: [N]
        reduce_mean is returning the avg squared diff among all samples in one batch
        """
        pr_loss = tf.reduce_mean(tf.reduce_sum(tf.square(features - assigned_prototype), [1]) / 2)

        return distances_class, assignment, pr_loss, vc_loss


    def dce_loss(self, distance, gama, y):
        logits = distance * gama
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))


    def compute_acc(self, distances, y):
        # predicted label
        pred_y = tf.cast(tf.argmax(distances, 1), tf.int32)

        # take the number of correct predictions
        acc = tf.reduce_sum(tf.cast(tf.equal(pred_y, y), tf.float32))

        return acc

    # Load pre-trained VGG weight on MNIST dataset
    def load_vgg_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i >= len(self.vgg_parameters):
                break
            #print(i, k, np.shape(weights[k]))
            sess.run(self.vgg_parameters[i].assign(weights[k]))

    # Load pre-trained VGG weight on MNIST dataset
    def load_vgg_weights_from_model(self, model, sess):     
        layers = [v.name for v in model.weights]
        for i in range (len(layers)):
            if i >= len(self.vgg_parameters):
                break
            #print(i, k, np.shape(weights[k]))
            sess.run(self.vgg_parameters[i].assign(model.weights[i]))
  
    # Load feature dictionary for VGG16-MNIST
    def load_VCs(self, weight_file, sess):
        weight = np.load(weight_file, allow_pickle=True).astype(np.float32)
        weight = tf.reshape(tf.transpose(weight, [1, 0]), [1, 1, 512, 512])
        sess.run(self.VCs.assign(weight))

    # Load prototypes
    def load_prototypes(self, weight_file, sess):
        weight = np.load(weight_file, allow_pickle=True).astype(np.float32)
        sess.run(self.prototypes_load.assign(weight))


    def load_all_weights(self, weight_file, sess):
        saver = tf.train.Saver()
        saver.restore(sess, weight_file)


    def save_all_weights(self, weight_file, sess):
        saver = tf.train.Saver()
        saver.save(sess, weight_file)



class VGG_Prototype:
    def __init__(self, class_num, prototype_num, recurrent_step=0, normalize_feature_map=False, use_threshold=True, add_vc_loss=True, distance_kind='euclidean', normalize_before_attention=True, use_sigmoid=False):
        self.class_num = class_num
        self.prototype_num = prototype_num
        self.recurrent_step = recurrent_step
        self.normalize_feature_map = normalize_feature_map
        self.use_threshold = use_threshold
        self.distance_kind = distance_kind
        self.normalize_before_attention = normalize_before_attention
        self.use_sigmoid = use_sigmoid
        self.x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.y = tf.placeholder(tf.int32, [None])

        self.gama = tf.get_variable('gama', dtype=tf.float32, initializer=tf.constant(20.0, dtype=tf.float32))

        self.distances_class, self.assignment, self.pr_loss, self.vc_loss = self.network(self.x)

        alpha1 = 1
        alpha2 = 1

        self.loss = self.dce_loss(self.distances_class, self.gama, self.y)
        self.loss += alpha1 * self.pr_loss
        self.loss += 5e-4 * tf.add_n(tf.get_collection('losses'))
        if add_vc_loss:
            self.loss += alpha2 * self.vc_loss

        self.acc = self.compute_acc(self.distances_class, self.y)


    def Conv(self, x, scope_name, shape, stride=[1, 1, 1, 1], padding='SAME', activation=True, vgg_parameters=False, all_parameters=False, weight_decay=True):
        initer = tf.truncated_normal_initializer(stddev=0.01)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weight', dtype=tf.float32, shape=shape, initializer=initer)
            conv = tf.nn.conv2d(x, kernel, stride, padding=padding)
            bias = tf.get_variable('bias', dtype=tf.float32, initializer=tf.constant(0.0, shape=[shape[3]], dtype=tf.float32))
            out = tf.nn.bias_add(conv, bias)

            if activation:
                out = tf.nn.relu(out)

            if vgg_parameters:
                self.vgg_parameters += [kernel, bias]

            if all_parameters:
                self.all_parameters += [kernel, bias]

            if weight_decay:
                tf.add_to_collection('losses', tf.nn.l2_loss(kernel))

        return out


    def network(self, x):
        self.vgg_parameters = []
        self.all_parameters = []

        # zero-mean input
        with tf.variable_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = x - mean

        initer = tf.truncated_normal_initializer(stddev=0.01)

        self.conv1_1 = self.Conv(images, 'conv1_1', [3, 3, 3, 64], vgg_parameters=True, all_parameters=True)

        self.conv1_2 = self.Conv(self.conv1_1, 'conv1_2', [3, 3, 64, 64], vgg_parameters=True, all_parameters=True)

        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        self.conv2_1 = self.Conv(self.pool1, 'conv2_1', [3, 3, 64, 128], vgg_parameters=True, all_parameters=True)

        self.conv2_2 = self.Conv(self.conv2_1, 'conv2_2', [3, 3, 128, 128], vgg_parameters=True, all_parameters=True)

        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        self.conv3_1 = self.Conv(self.pool2, 'conv3_1', [3, 3, 128, 256], vgg_parameters=True, all_parameters=True)

        self.conv3_2 = self.Conv(self.conv3_1, 'conv3_2', [3, 3, 256, 256], vgg_parameters=True, all_parameters=True)

        self.conv3_3 = self.Conv(self.conv3_2, 'conv3_3', [3, 3, 256, 256], vgg_parameters=True, all_parameters=True)

        self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        self.conv4_1 = self.Conv(self.pool3, 'conv4_1', [3, 3, 256, 512], vgg_parameters=True, all_parameters=True)

        self.conv4_2 = self.Conv(self.conv4_1, 'conv4_2', [3, 3, 512, 512], vgg_parameters=True, all_parameters=True)

        self.conv4_3 = self.Conv(self.conv4_2, 'conv4_3', [3, 3, 512, 512], vgg_parameters=True, all_parameters=True)

        self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        vc_loss = 0

        for i in range(self.recurrent_step):
            # visual concept attention
            with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
                VCs = tf.get_variable('visual_concept', dtype=tf.float32, shape=[1, 1, 512, 512], initializer=initer)
                if i == 0:
                    self.all_parameters += [VCs]
                    self.VCs = VCs

                VCs = tf.nn.l2_normalize(VCs, dim=[2])

                pool4_n = tf.nn.l2_normalize(self.pool4, dim=[3])

                if self.normalize_feature_map:
                    similarity = tf.nn.conv2d(pool4_n, VCs, [1, 1, 1, 1], padding='SAME')
                    similarity_normalize = similarity
                else:
                    similarity = tf.nn.conv2d(self.pool4, VCs, [1, 1, 1, 1], padding='SAME')
                    similarity_normalize = tf.nn.conv2d(pool4_n, VCs, [1, 1, 1, 1], padding='SAME')

                # returns the largest value of similarity along dim = 3 
                self.attentions = tf.reduce_max(similarity, reduction_indices=[3])
                self.attentions_not_normalize = self.attentions

                # get the index of VCs w.r.t features at each position on the feature map(i,j)
                vc_assign = tf.argmax(similarity_normalize, axis=3)
                
                # transpose the VCs from features x number of object centres -----> number of object centres x features
                vcs = tf.transpose(tf.reshape(VCs, [512, 512]), [1, 0])

                # get features from VCs and compute the difference between VCs and feature map
                vc_loss += tf.reduce_mean(tf.square(tf.gather(vcs, vc_assign) - pool4_n) / 2)


                # threshold is used
                if self.use_threshold:
                    flatten_attention = tf.reshape(self.attentions, [-1, 14 * 14])
                    percentage_l = 0.8
                    percentage_u = 0.2
                    percentage_ln = int(14 * 14 * percentage_l)
                    percentage_un = int(14 * 14 * percentage_u)
                    threshold_l = tf.nn.top_k(flatten_attention, percentage_ln).values[:, -1]
                    threshold_u = tf.nn.top_k(flatten_attention, percentage_un).values[:, -1]

                    threshold_l = tf.reshape(threshold_l, [-1, 1, 1])
                    attentions = tf.nn.relu(self.attentions - threshold_l) + threshold_l

                    threshold_u = tf.reshape(threshold_u, [-1, 1, 1])
                    threshold_u = tf.tile(threshold_u, [1, 14, 14])
                    attentions = tf.clip_by_value(attentions, 0, threshold_u)

                    attentions = attentions / threshold_u
                    self.attentions = attentions
                elif self.use_sigmoid:
                    self.attentions = tf.nn.sigmoid(self.attentions - tf.reduce_mean(self.attentions, reduction_indices=[1,2], keepdims=True))
                else:
                    self.attentions = self.attentions / tf.reduce_max(self.attentions, reduction_indices=[1,2], keepdims=True)

            # apply attentions to pool1
            pool1 = tf.reshape(self.pool1, [-1, 14, 8, 14, 8, 64])
            attentions = tf.reshape(self.attentions, [-1, 14, 1, 14, 1, 1])
            pool1 = pool1 * attentions
            new_pool1 = tf.reshape(pool1, [-1, 112, 112, 64])

            self.conv2_1 = self.Conv(new_pool1, 'conv2_1', [3, 3, 64, 128], vgg_parameters=False, all_parameters=False)

            self.conv2_2 = self.Conv(self.conv2_1, 'conv2_2', [3, 3, 128, 128], vgg_parameters=False, all_parameters=False)

            self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

            self.conv3_1 = self.Conv(self.pool2, 'conv3_1', [3, 3, 128, 256], vgg_parameters=False, all_parameters=False)

            self.conv3_2 = self.Conv(self.conv3_1, 'conv3_2', [3, 3, 256, 256], vgg_parameters=False, all_parameters=False)

            self.conv3_3 = self.Conv(self.conv3_2, 'conv3_3', [3, 3, 256, 256], vgg_parameters=False, all_parameters=False)

            self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

            self.conv4_1 = self.Conv(self.pool3, 'conv4_1', [3, 3, 256, 512], vgg_parameters=False, all_parameters=False)

            self.conv4_2 = self.Conv(self.conv4_1, 'conv4_2', [3, 3, 512, 512], vgg_parameters=False, all_parameters=False)

            self.conv4_3 = self.Conv(self.conv4_2, 'conv4_3', [3, 3, 512, 512], vgg_parameters=False, all_parameters=False)

            self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        self.conv5_1 = self.Conv(self.pool4, 'conv5_1', [3, 3, 512, 512], vgg_parameters=True, all_parameters=True)

        self.conv5_2 = self.Conv(self.conv5_1, 'conv5_2', [3, 3, 512, 512], vgg_parameters=True, all_parameters=True)

        self.conv5_3 = self.Conv(self.conv5_2, 'conv5_3', [3, 3, 512, 512], vgg_parameters=True, all_parameters=True)

        self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

        with tf.variable_scope('prototype', reuse=tf.AUTO_REUSE):
            self.prototypes = tf.get_variable('prototypes', dtype=tf.float32, shape=[self.class_num * self.prototype_num, 7, 7, 512])
            self.all_parameters += [self.prototypes]

        # normalize
        features = self.pool5
        features = tf.reshape(features, [-1, 7*7*512])
        features = tf.nn.l2_normalize(features, dim=[1])
        prototypes = self.prototypes
        prototypes = tf.reshape(prototypes, [self.class_num * self.prototype_num, 7*7*512])
        prototypes = tf.nn.l2_normalize(prototypes, dim=[1])
        prototypes = tf.transpose(prototypes, [1, 0])

        # compare distance
        distances = tf.matmul(features, prototypes)
        distances = tf.reshape(distances, [-1, self.class_num * self.prototype_num])

        # get nearest in each class
        distances_class = tf.reduce_max(tf.reshape(distances, [-1, self.class_num, self.prototype_num]), reduction_indices=[2])

        # get assignment
        assignment = tf.cast(tf.argmax(distances, 1), tf.int32)

        # prototype loss
        prototypes = tf.transpose(prototypes, [1, 0])
        assigned_prototype = tf.gather(prototypes, assignment)
        features = tf.reshape(features, [-1, 7*7*512])
        pr_loss = tf.reduce_mean(tf.reduce_sum(tf.square(features - assigned_prototype), [1]) / 2)

        return distances_class, assignment, pr_loss, vc_loss


    def dce_loss(self, distance, gama, y):
        logits = distance * gama

        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))


    def compute_acc(self, distances, y):
        pred_y = tf.cast(tf.argmax(distances, 1), tf.int32)

        acc = tf.reduce_sum(tf.cast(tf.equal(pred_y, y), tf.float32))

        return acc


    def load_vgg_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i >= len(self.vgg_parameters):
                break
            sess.run(self.vgg_parameters[i].assign(weights[k]))


    def load_prototypes(self, weight_file, sess):
        weight = np.load(weight_file, allow_pickle=True).astype(np.float32)
        sess.run(self.prototypes.assign(weight))


    def load_VCs(self, weight_file, sess):
        weight = np.load(weight_file, allow_pickle=True).astype(np.float32)
        weight = tf.reshape(tf.transpose(weight, [1, 0]), [1, 1, 512, 512])
        sess.run(self.VCs.assign(weight))


    def load_all_weights(self, weight_file, sess):
        saver = tf.train.Saver()
        saver.restore(sess, weight_file)


    def save_all_weights(self, weight_file, sess):
        saver = tf.train.Saver()
        saver.save(sess, weight_file)


class VGG_ATTENTION:
    def __init__(self, class_num, recurrent_step, add_vc_loss=True):
        self.class_num = class_num
        self.recurrent_step = recurrent_step
        self.x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.y = tf.placeholder(tf.int32, [None])

        self.logits, self.vc_loss = self.network(self.x)

        self.loss = self.ce_loss(self.logits, self.y)
        self.loss += 5e-4 * tf.add_n(tf.get_collection('losses'))
        if add_vc_loss:
            self.loss += self.vc_loss

        self.acc = self.compute_acc(self.logits, self.y)


    def Conv(self, x, scope_name, shape, stride=[1, 1, 1, 1], padding='SAME', activation=True, vgg_parameters=False, all_parameters=False, weight_decay=True):
        initer = tf.truncated_normal_initializer(stddev=0.01)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weight', dtype=tf.float32, shape=shape, initializer=initer)
            conv = tf.nn.conv2d(x, kernel, stride, padding=padding)
            bias = tf.get_variable('bias', dtype=tf.float32, initializer=tf.constant(0.0, shape=[shape[3]], dtype=tf.float32))
            out = tf.nn.bias_add(conv, bias)

            if activation:
                out = tf.nn.relu(out)

            if vgg_parameters:
                self.vgg_parameters += [kernel, bias]

            if all_parameters:
                self.all_parameters += [kernel, bias]

            if weight_decay:
                tf.add_to_collection('losses', tf.nn.l2_loss(kernel))

        return out

    def FC(self, x, scope_name, shape, activation=True, vgg_parameters=False, all_parameters=False, weight_decay=True):
        initer = tf.truncated_normal_initializer(stddev=0.01)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            weight = tf.get_variable('weight', dtype=tf.float32, shape=shape, initializer=initer)
            bias = tf.get_variable('bias', dtype=tf.float32, initializer=tf.constant(0.0, shape=[shape[1]], dtype=tf.float32))
            out = tf.matmul(x, weight) + bias

            if activation:
                out = tf.nn.relu(out)

            if vgg_parameters:
                self.vgg_parameters += [weight, bias]

            if all_parameters:
                self.all_parameters += [weight, bias]

            if weight_decay:
                tf.add_to_collection('losses', tf.nn.l2_loss(weight))

        return out


    def network(self, x):
        self.vgg_parameters = []
        self.all_parameters = []

        # zero-mean input
        with tf.variable_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = x - mean

        initer = tf.truncated_normal_initializer(stddev=0.01)

        self.conv1_1 = self.Conv(images, 'conv1_1', [3, 3, 3, 64], vgg_parameters=True, all_parameters=True)

        self.conv1_2 = self.Conv(self.conv1_1, 'conv1_2', [3, 3, 64, 64], vgg_parameters=True, all_parameters=True)

        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        self.conv2_1 = self.Conv(self.pool1, 'conv2_1', [3, 3, 64, 128], vgg_parameters=True, all_parameters=True)

        self.conv2_2 = self.Conv(self.conv2_1, 'conv2_2', [3, 3, 128, 128], vgg_parameters=True, all_parameters=True)

        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        self.conv3_1 = self.Conv(self.pool2, 'conv3_1', [3, 3, 128, 256], vgg_parameters=True, all_parameters=True)

        self.conv3_2 = self.Conv(self.conv3_1, 'conv3_2', [3, 3, 256, 256], vgg_parameters=True, all_parameters=True)

        self.conv3_3 = self.Conv(self.conv3_2, 'conv3_3', [3, 3, 256, 256], vgg_parameters=True, all_parameters=True)

        self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        self.conv4_1 = self.Conv(self.pool3, 'conv4_1', [3, 3, 256, 512], vgg_parameters=True, all_parameters=True)

        self.conv4_2 = self.Conv(self.conv4_1, 'conv4_2', [3, 3, 512, 512], vgg_parameters=True, all_parameters=True)

        self.conv4_3 = self.Conv(self.conv4_2, 'conv4_3', [3, 3, 512, 512], vgg_parameters=True, all_parameters=True)

        self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        vc_loss = 0.0

        for i in range(self.recurrent_step):
            # visual concept attention
            with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
                VCs = tf.get_variable('visual_concept', dtype=tf.float32, shape=[1, 1, 512, 512], initializer=initer)
                if i == 0:
                    self.all_parameters += [VCs]
                    self.VCs = VCs

                VCs = tf.nn.l2_normalize(VCs, dim=[2])

                pool4_n = tf.nn.l2_normalize(self.pool4, dim=[3])
                
                similarity = tf.nn.conv2d(self.pool4, VCs, [1, 1, 1, 1], padding='SAME')
                similarity_normalize = tf.nn.conv2d(pool4_n, VCs, [1, 1, 1, 1], padding='SAME')

                self.attentions = tf.reduce_max(similarity, reduction_indices=[3])
                self.attentions_not_normalize = self.attentions

                vc_assign = tf.argmax(similarity_normalize, axis=3)
                vcs = tf.transpose(tf.reshape(VCs, [512, 512]), [1, 0])
                vc_loss += tf.reduce_mean(tf.square(tf.gather(vcs, vc_assign) - pool4_n) / 2)

                flatten_attention = tf.reshape(self.attentions, [-1, 14 * 14])
                percentage_l = 0.8
                percentage_u = 0.2
                percentage_ln = int(14 * 14 * percentage_l)
                percentage_un = int(14 * 14 * percentage_u)
                threshold_l = tf.nn.top_k(flatten_attention, percentage_ln).values[:, -1]
                threshold_u = tf.nn.top_k(flatten_attention, percentage_un).values[:, -1]

                threshold_l = tf.reshape(threshold_l, [-1, 1, 1])
                attentions = tf.nn.relu(self.attentions - threshold_l) + threshold_l

                threshold_u = tf.reshape(threshold_u, [-1, 1, 1])
                threshold_u = tf.tile(threshold_u, [1, 14, 14])
                attentions = tf.clip_by_value(attentions, 0, threshold_u)

                attentions = attentions / threshold_u
                self.attentions = attentions

            # apply attentions to pool1
            pool1 = tf.reshape(self.pool1, [-1, 14, 8, 14, 8, 64])
            attentions = tf.reshape(self.attentions, [-1, 14, 1, 14, 1, 1])
            pool1 = pool1 * attentions
            new_pool1 = tf.reshape(pool1, [-1, 112, 112, 64])

            self.conv2_1 = self.Conv(new_pool1, 'conv2_1', [3, 3, 64, 128], vgg_parameters=False, all_parameters=False)

            self.conv2_2 = self.Conv(self.conv2_1, 'conv2_2', [3, 3, 128, 128], vgg_parameters=False, all_parameters=False)

            self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

            self.conv3_1 = self.Conv(self.pool2, 'conv3_1', [3, 3, 128, 256], vgg_parameters=False, all_parameters=False)

            self.conv3_2 = self.Conv(self.conv3_1, 'conv3_2', [3, 3, 256, 256], vgg_parameters=False, all_parameters=False)

            self.conv3_3 = self.Conv(self.conv3_2, 'conv3_3', [3, 3, 256, 256], vgg_parameters=False, all_parameters=False)

            self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

            self.conv4_1 = self.Conv(self.pool3, 'conv4_1', [3, 3, 256, 512], vgg_parameters=False, all_parameters=False)

            self.conv4_2 = self.Conv(self.conv4_1, 'conv4_2', [3, 3, 512, 512], vgg_parameters=False, all_parameters=False)

            self.conv4_3 = self.Conv(self.conv4_2, 'conv4_3', [3, 3, 512, 512], vgg_parameters=False, all_parameters=False)

            self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        self.conv5_1 = self.Conv(self.pool4, 'conv5_1', [3, 3, 512, 512], vgg_parameters=True, all_parameters=True)

        self.conv5_2 = self.Conv(self.conv5_1, 'conv5_2', [3, 3, 512, 512], vgg_parameters=True, all_parameters=True)

        self.conv5_3 = self.Conv(self.conv5_2, 'conv5_3', [3, 3, 512, 512], vgg_parameters=True, all_parameters=True)

        self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

        flatten_pool5 = tf.reshape(self.pool5, [-1, 7*7*512])

        self.fc6 = self.FC(flatten_pool5, 'fc6', [7*7*512, 4096], vgg_parameters=True, all_parameters=True)

        self.fc7 = self.FC(self.fc6, 'fc7', [4096, 4096], vgg_parameters=True, all_parameters=True)

        self.fc8 = self.FC(self.fc7, 'fc8', [4096, self.class_num], activation=False, vgg_parameters=False, all_parameters=True)

        logits = tf.reshape(self.fc8, [-1, self.class_num])
        
        return logits, vc_loss


    def ce_loss(self, logits, y):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))


    def compute_acc(self, logits, y):
        pred_y = tf.cast(tf.argmax(logits, 1), tf.int32)

        acc = tf.reduce_sum(tf.cast(tf.equal(pred_y, y), tf.float32))

        return acc


    def load_vgg_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i >= len(self.vgg_parameters):
                break
            #print(i, k, np.shape(weights[k]))
            sess.run(self.vgg_parameters[i].assign(weights[k]))


    def load_VCs(self, weight_file, sess):
        weight = np.load(weight_file, allow_pickle=True).astype(np.float32)
        weight = tf.reshape(tf.transpose(weight, [1, 0]), [1, 1, 512, 512])
        sess.run(self.VCs.assign(weight))


    def load_all_weights(self, weight_file, sess):
        saver = tf.train.Saver()
        saver.restore(sess, weight_file)


    def save_all_weights(self, weight_file, sess):
        saver = tf.train.Saver()
        saver.save(sess, weight_file)

