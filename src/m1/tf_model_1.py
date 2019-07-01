import operator
import pickle
import math
import tensorflow as tf
import numpy as np
import glob
import pandas as pd
import os
from sklearn.manifold import TSNE

from numpy.core._multiarray_umath import ndarray
from sklearn.manifold import TSNE
import sys
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import time
import inspect
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
import yaml
from tensorflow.contrib.tensorboard.plugins import projector

class model:
    def __init__(self, MODEL_NAME, SAVE_DIR, OP_DIR):

        self.inference = False
        self.save_dir = SAVE_DIR
        self.op_dir = OP_DIR
        self.frozen_file = None
        self.ts = None
        self.use_bias = True
        self.save_loss_fig = True
        self.show_figure = False
        self.use_activation = False
        self.test_SerialID = None
        self._model_name = MODEL_NAME
        return

    def set_model_hyperparams(
            self,
            domain_dims,
            emb_dims,
            use_bias=True,
            batch_size=128,
            num_epochs=20,
            learning_rate=0.001,
            alpha=0.0025
    ):
        MODEL_NAME = self._model_name
        self.learning_rate = learning_rate
        self.num_domains = len(domain_dims)
        self.domain_dims = domain_dims
        self.num_emb_layers = len(emb_dims)
        self.emb_dims = emb_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_signature = MODEL_NAME + '_'.join([str(e) for e in emb_dims])
        self.use_bias = use_bias
        self.alpha = alpha
        return

    def set_model_options(
            self,
            show_loss_figure,
            save_loss_figure,
            set_w_mean=True
    ):
        self.show_loss_figure = show_loss_figure
        self.save_loss_figure = save_loss_figure
        self.set_w_mean = True
        return

    def set_SerialID(self, test_SerialID):
        self.test_SerialID = test_SerialID

    def get_weight_variable(
            self,
            shape,
            name=None
    ):
        initializer = tf.contrib.layers.xavier_initializer()
        if name is not None:
            return tf.Variable(initializer(shape), name=name)
        else:
            return tf.Variable(initializer(shape))

    def define_wbs(self):
        print('>> Defining weights :: start')

        self.W = [None] * self.num_emb_layers
        self.b = [None] * self.num_emb_layers

        wb_scope_name = 'params'
        # doman dimensions

        layer_1_dims = []
        for i in self.domain_dims:
            _d = int(math.ceil(math.log(i, 2)))
            if _d <= 1:
                _d += 1
            layer_1_dims.append(_d)
        # print(layer_1_dims)

        with tf.name_scope(wb_scope_name):
            prefix = self.model_scope_name + '/' + wb_scope_name + '/'
            self.wb_names = []

            # -------
            # For each layer define W , b
            # -------
            for l in range(self.num_emb_layers):
                self.W[l] = [None] * self.num_domains
                self.b[l] = [None] * self.num_domains

                # print("----> Layer", (l + 1))
                if l == 0:
                    layer_inp_dims = self.domain_dims
                    layer_op_dims = layer_1_dims

                else:
                    if l == 1:
                        layer_inp_dims = layer_1_dims
                    else:
                        layer_inp_dims = [self.emb_dims[l - 1]] * self.num_domains
                    layer_op_dims = [self.emb_dims[l]] * self.num_domains

                for d in range(self.num_domains):

                    _name = 'W_layer_' + str(l) + '_domain_' + str(d)

                    if self.inference is True:
                        n = prefix + _name + ':0'
                        self.W[l][d] = self.restore_graph.get_tensor_by_name(n)
                    else:

                        z = self.get_weight_variable(
                            [layer_inp_dims[d],
                             layer_op_dims[d]],
                            name=_name)
                        self.W[l][d] = z
                        self.wb_names.append(prefix + _name)

                if self.use_bias:
                    for d in range(self.num_domains):
                        _name_b = 'bias_layer_' + str(l) + '_domain_' + str(d)
                        b_dims = [layer_op_dims[d]]  # opdim1, opdim 2

                        if self.inference is True:
                            n = prefix + _name_b + ':0'
                            self.b[l][d] = self.restore_graph.get_tensor_by_name(n)
                        else:
                            z = self.get_weight_variable(b_dims, _name_b)
                            self.b[l][d] = z
                            self.wb_names.append(prefix + _name_b)

        print('>> Defining weights :: end')

    def restore_model(self):

        # Model already restored!
        if self.inference is True:
            return

        self.inference = True
        if self.frozen_file is None:
            # ensure embedding dimensions are correct
            emb = '_'.join([str(_) for _ in self.emb_dims])
            files = glob.glob(os.path.join(self.save_dir, 'checkpoints', '*' + emb + '*.pb'))
            f_name = files[-1]
            self.frozen_file = f_name

        if self.ts is None:
            self.ts = '.'.join(
                (''.join(
                    self.frozen_file.split('_')[-1]
                )
                ).split('.')[:1])
        # print('ts ::', self.ts)

        tf.reset_default_graph()
        # print('Frozen file', self.frozen_file)

        with tf.gfile.GFile(self.frozen_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.restore_graph = None

        with tf.Graph().as_default() as g:
            try:
                tf.graph_util.import_graph_def(
                    graph_def,
                    input_map=None,
                    name='',
                    return_elements=None,
                    op_dict=None,
                    producer_op_list=None
                )
            except:
                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    name='',
                    return_elements=None,
                    op_dict=None,
                    producer_op_list=None
                )
            self.restore_graph = g
            self.inference = True
            self.build_model()
        return

    # ---------------------------------------------------------- #

    def _add_var_summaries(self):
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
            continue

            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)

            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
        return




    def build_model(self):
        print('Building model : start ')
        self.model_scope_name = 'model'

        with tf.variable_scope(self.model_scope_name):
            # batch_size ,domains, label_id
            self.x_pos_inp = tf.placeholder(
                tf.int32,
                [None, self.num_domains]
            )

            # Inside the scope	- define the weights and biases
            self.define_wbs()

            x_pos_inp = tf.split(
                self.x_pos_inp,
                self.num_domains,
                axis=-1
            )

            x_pos_WXb = [None] * self.num_domains

            for d in range(self.num_domains):
                # for each domain
                prev = None
                for l in range(self.num_emb_layers):

                    if l == 0:
                        a = tf.nn.embedding_lookup(
                            self.W[l][d],
                            x_pos_inp[d]
                        )
                        _wx = tf.squeeze(a, axis=1)

                    else:
                        _x = prev
                        _wx = tf.matmul(
                            _x,
                            self.W[l][d]
                        )

                    if self.use_bias:
                        _wx_b = tf.add(_wx, self.b[l][d])
                    else:
                        _wx_b = _wx

                    prev = _wx_b
                x_pos_WXb[d] = prev
                # print(x_pos_WXb[d].shape)

            emb_op_pos = x_pos_WXb
            # emb_op_pos = [tf.nn.l2_normalize(_, axis=-1) for _ in emb_op_pos]
            print(emb_op_pos[0].shape)

            self.joint_emb_op = tf.stack(emb_op_pos, axis=1)
            self.concat_emb_op = tf.concat(emb_op_pos, axis=1)

            # Calculate the combined embedding of the record

            self.mean_emb_op = tf.reduce_mean(
                self.joint_emb_op,
                axis=1,
                keepdims=False
            )

            '''
            Optimization stage
            '''

            # small epsilon to avoid any sort of underflows
            _epsilon = tf.constant(math.pow(10.0, -7.0))

            # pairwise similarity
            cos_dist = []
            for i in range(self.num_domains):
                for j in range(i+1, self.num_domains):

                    r1 = tf.reduce_sum(
                        tf.multiply(
                            tf.nn.l2_normalize(emb_op_pos[i], axis=-1),
                            tf.nn.l2_normalize(emb_op_pos[j], axis=-1),
                        ),
                        1,
                        keepdims=True
                    )
                    d = 1.0 - r1
                    cos_dist.append(d)

            cos_dist = tf.stack(cos_dist,axis=1)
            cos_dist = tf.squeeze(cos_dist, axis=-1)
            self.pairwise_cosine = cos_dist


            self.loss_1 = tf.reduce_mean(
                tf.math.square(cos_dist),
                axis=-1
            )

            # ensure embeddings sum to 1

            norm_loss = []
            for i in range(self.num_domains):
                pass
                t1 = tf.norm(emb_op_pos[i],axis=-1,keep_dims=True)
                t1 = tf.square(1 - t1)
                norm_loss.append(t1)

            norm_loss = tf.stack(norm_loss,axis=1)
            norm_loss = tf.squeeze(norm_loss, axis=-1)
            norm_loss = tf.reduce_mean(norm_loss,axis=-1)
            print(norm_loss.shape)

            self.loss_2 =  norm_loss
            self.loss = 0.5 * self.loss_1 + 0.5 * self.loss_2

            regularizer_beta = tf.constant(math.pow(10, -7))

            for _W in self.W :
                regularizer = [ regularizer_beta * tf.nn.l2_loss(_) for _ in _W]
                for _ in regularizer :
                    self.loss += _
            for _B in self.b:
                regularizer = [regularizer_beta * tf.nn.l2_loss(_) for _ in _B]
                for _ in regularizer:
                    self.loss += _

            print(' -->> ', self.loss.shape)


            # print('Loss shape', self.loss.shape)
            tf.summary.scalar('loss', tf.reduce_mean(self.loss))
            tf.summary.scalar('loss_1', tf.reduce_mean(self.loss_1))
            tf.summary.scalar('loss_2', tf.reduce_mean(self.loss_2))

            if self.inference:
                return

            self._add_var_summaries()
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate
            )

            # self.train_opt = self.optimizer.minimize(self.loss)
            gvs = self.optimizer.compute_gradients(self.loss)

            grad_summ_op = tf.summary.merge(
                [tf.summary.histogram(
                    "%s-grad" % g[1].name, g[0]
                ) for g in gvs
                ]
            )

            self.gradients = gvs

            tf.summary.all_v2_summary_ops()
            capped_gvs = [
                (tf.clip_by_value(grad, -1.0, 1.0), var)
                for grad, var in gvs
            ]

            self.train_opt = self.optimizer.apply_gradients(capped_gvs)

            self.summary = tf.summary.merge_all()
            return

    def set_pretrained_model_file(self, f_path):
        self.frozen_file = f_path
        return

    def train_model(self, x):
        print('Start of training :: ')
        self.ts = str(time.time()).split('.')[0]

        f_name = 'frozen' + '_' + self.model_signature + '_' + self.ts + '.pb'


        self.frozen_file = os.path.join(
            self.save_dir, 'checkpoints', f_name
        )

        self.sess = tf.InteractiveSession()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.saver = tf.train.Saver()
        bs = self.batch_size
        x_pos = x

        num_batches = x_pos.shape[0] // bs
        losses = []

        frozen_graph_def = None
        Check_Save_Prev = False

        print('Num batches :', num_batches)

        self.output_dir = './..'
        summary_writer = tf.summary.FileWriter(self.output_dir + '/test')
        step = 0

        for e in range(self.num_epochs):

            t1 = time.time()
            for _b in range(num_batches):
                _x_pos = x_pos[_b * bs: (_b + 1) * bs]
                if _b == num_batches - 1:
                    _x_pos = x_pos[_b * bs:]
                if _b == 0:
                    print(_x_pos.shape)

                _, summary, loss1, loss2, loss = self.sess.run(
                    [self.train_opt, self.summary, self.loss_1, self.loss_2, self.loss],
                    feed_dict={
                        self.x_pos_inp: _x_pos,
                    }
                )

                batch_loss = np.mean(loss)
                losses.append(batch_loss)
                print(batch_loss)

                summary_writer.add_summary(summary, step)
                step += 1
                if np.isnan(batch_loss):
                    Check_Save_Prev = True
                    print('[ERROR] Loss is NaN !!!, breaking...')
                    break

            if Check_Save_Prev is True:
                break
            else:
                graph_def = tf.get_default_graph().as_graph_def()
                frozen_graph_def = convert_variables_to_constants(
                    self.sess,
                    graph_def,
                    self.wb_names
                )
                with tf.gfile.GFile(self.frozen_file, "wb") as f:
                    f.write(frozen_graph_def.SerializeToString())

                t2 = time.time()
                t = (t2 - t1) / 60
                print('Epoch ', e + 1, 'Time elapsed in epoch : ', t, 'minutes')

        if self.save_loss_fig or self.show_loss_figure:
            plt.figure()
            plt.title('Training Loss')
            plt.xlabel('batch')
            plt.ylabel('loss')
            plt.plot(range(len(losses)), losses, 'r-')

            if self.save_loss_figure:
                fig_name = 'loss_' + self.model_signature + '_epochs_' + str(self.num_epochs) + '_' + self.ts + '.png'
                file_path = os.path.join(self.op_dir, fig_name)
                plt.savefig(file_path)

            if self.show_loss_figure:
                plt.show()
            plt.close()
        return self.frozen_file

    # This is an external function
    # x is the index data
    # ep is entity probability

    def get_embedding_mean(self, x):
        self.set_w_mean = False
        self.restore_model()
        output = []
        bs = self.batch_size
        num_batches = x.shape[0] // bs

        with tf.Session(graph=self.restore_graph) as sess:
            for _b in range(num_batches):
                _x = x[_b * bs: (_b + 1) * bs]
                if _b == num_batches - 1:
                    _x = x[_b * bs:]

                _output = sess.run(
                    self.concat_emb_op,
                    feed_dict={
                        self.x_pos_inp: _x
                    }
                )
                output.extend(_output)
            res = np.array(output)

        return res

    def get_embeddings(self, x):

        self.restore_model()
        output = []
        bs = self.batch_size
        num_batches = x.shape[0] // bs

        with tf.Session(graph=self.restore_graph) as sess:
            for _b in range(num_batches):
                _x = x[_b * bs: (_b + 1) * bs]
                if _b == num_batches - 1:
                    _x = x[_b * bs:]

                _output = sess.run(
                    self.mean_emb_op,
                    feed_dict={
                        self.x_pos_inp: _x
                    }
                )
                output.extend(_output)

            res = np.array(output)
            tf_data = tf.Variable(res)
            saver = tf.train.Saver([tf_data])
            sess.run(tf_data.initializer)
            saver.save(sess, os.path.join('./..', 'tf_data.ckpt'))
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = tf_data.name
            projector.visualize_embeddings(
                tf.summary.FileWriter('./..'),
                config
            )

        return res

    def get_w_embedding_mean(self, x, ep):
        self.set_w_mean = True
        self.restore_model()
        output = []
        bs = self.batch_size
        num_batches = x.shape[0] // bs

        with tf.Session(graph=self.restore_graph) as sess:
            for _b in range(num_batches):
                _x = x[_b * bs: (_b + 1) * bs]
                _ep = ep[_b * bs: (_b + 1) * bs]
                if _b == num_batches - 1:
                    _x = x[_b * bs:]
                    _ep = ep[_b * bs:]
                _output = sess.run(
                    self.w_mean_emb_op,
                    feed_dict={
                        self.x_pos_inp: _x,
                        self.entity_prob_x: _ep
                    }
                )
                output.extend(_output)
            res = np.array(output)

        # remove the 1st singular vector

        A = np.transpose(res)
        U, S, Vt = randomized_svd(A, n_components=1)
        # U, s, VT = svd(A,full_matrices=False)
        U_1 = U
        tmp = np.dot(U_1, np.transpose(U_1))

        R = []
        for r in res:
            r = r - np.matmul(tmp, r)
            R.append(r)
        R = np.array(R)

        return R



    def get_pairwise_cosine_dist(self, x):

        self.restore_model()
        output = []
        bs = self.batch_size
        num_batches = x.shape[0] // bs

        with tf.Session(graph=self.restore_graph) as sess:
            for _b in range(num_batches):
                _x = x[_b * bs: (_b + 1) * bs]
                if _b == num_batches - 1:
                    _x = x[_b * bs:]

                _output = sess.run(
                    self.pairwise_cosine,
                    feed_dict={
                        self.x_pos_inp: _x
                    }
                )
                output.extend(_output)
            res = np.array(output)

        return res