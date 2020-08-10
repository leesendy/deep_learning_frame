# -*- coding: utf-8 -*-
# Author: Li Disen
# E-mail: lidisen@126.com
# Affiliation: Northeastern University (China, Shenyang)
# Tools: TensorFlow version 1.12.1


import tensorflow as tf
import sklearn
from sklearn import metrics
import numpy as np
import os, time, collections, shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class Model(object):
    def __init__(self, num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                 regularization=0, dropout=0, batch_size=100, eval_frequency=200, dir_name='', height=224, width=224, channel=3):

        self.regularizers = []
        self.height, self.width, self.channel = height, width, channel
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name

        # Build the computational graph.
        self.build_graph()

    def build_graph(self):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, self.channel), name='data')
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), name='labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), name='dropout')

            # Model.
            op_logits = self.inference(self.ph_data, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                                          self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)

        self.graph.finalize()

    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):

            with tf.name_scope('cross_entropy'):
                labels = tf.to_int64(labels)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cross_entropy = tf.reduce_mean(cross_entropy)

            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)

            loss = cross_entropy + regularization

            # Summaries for TensorBoard.
            tf.summary.scalar('loss/cross_entropy', cross_entropy)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, regularization, loss])
                tf.summary.scalar('loss/avg_cross_entropy', averages.average(cross_entropy))
                tf.summary.scalar('loss/avg_regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg_total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average

    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                    learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)

            # Optimizer.
            if momentum == 0:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    def fit(self, train_data, train_labels, val_data, val_labels):
        t_process, t_wall = time.process_time(), time.time()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=self.graph, config=config)

        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)

        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')

        sess.run(self.op_init)

        # Training.
        accuracies = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        for step in range(1, num_steps + 1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]

            batch_data, batch_labels = train_data[idx, :, :], train_labels[idx]

            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: self.dropout}
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)

            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                string, accuracy, f1, loss = self.evaluate(val_data, val_labels, sess)
                accuracies.append(accuracy)
                losses.append(loss)
                print('  validation {}'.format(string))
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time() - t_process, time.time() - t_wall))

                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validation/accuracy', simple_value=accuracy)
                summary.value.add(tag='validation/f1', simple_value=f1)
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)

                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
        writer.close()
        sess.close()

        t_step = (time.time() - t_wall) / num_steps
        return accuracies, losses, t_step

    def predict(self, data, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2], data.shape[3]))
            tmp_data = data[begin:end, :, :, :]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end - begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}

            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros(self.batch_size)
                batch_labels[:end - begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)

            predictions[begin:end] = batch_pred[:end - begin]

        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions

    def evaluate(self, data, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x H x W x C
            N: number of signals (samples)
            H: height of image
            W: width of image
            C: channel of image
        labels: size N
            N: number of signals (samples)
        """
        t_process, t_wall = time.process_time(), time.time()
        predictions, loss = self.predict(data, labels, sess)
        # print(predictions)
        ncorrects = sum(predictions == labels)

        accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')

        string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
            accuracy, ncorrects, len(labels), f1, loss)
        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time() - t_process, time.time() - t_wall)
        return string, accuracy, f1, loss

    def prediction(self, logits):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=1)
            return prediction

    def inference(self, data, dropout):
        """
        data: size N x H x W x C
            N: number of signals (samples)
            H: height of data
            W: width of data
            C: channel of data
        dropout: dropout keep (int) i.e. 1 mean save all the op
        """
        logits = self._inference(data, dropout)
        return logits

    def _inference(self, x, dropout):
        with tf.variable_scope('conv1'):
            x = self.conv2D(x, 3, 32, relu=True)
            x = tf.layers.flatten(x)
        with tf.variable_scope('logits'):
            x = self.fc(x, 2, relu=False)
            x = tf.nn.dropout(x, dropout)
        return x

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def conv2D(self, x, in_channel, out_channel, relu=True):

        W = self._weight_variable([3, 3, in_channel, out_channel], regularization=True)
        b = self._bias_variable([out_channel], regularization=True)
        x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

        return tf.nn.relu(x + b) if relu else x

    # help tools
    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, folder, self.dir_name)

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, regularization=True):

        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var
