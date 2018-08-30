import tensorflow as tf

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.client.session import Session
from tensorflow.python.keras.datasets import mnist


class MnistMlp(object):
    def __init__(self, init_learning_rate=0.1):
        self._init_learning_rate=init_learning_rate

        self._is_training = tf.placeholder(dtype=tf.bool, shape=())
        self._learning_rate = tf.placeholder(dtype=tf.float32, shape=())
        self._batch_size = tf.placeholder(dtype=tf.int64, shape=())

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self._train_dataset()
        self._test_dataset()

    def _train_dataset(self, name_scope='train_dataset'):
        with tf.name_scope(name_scope):
            dataset = dataset_ops.Dataset.from_tensor_slices((self.x_train, self.y_train))
            dataset = dataset.shuffle(buffer_size=self._batch_size * 5)
            dataset = dataset.repeat().batch(self._batch_size).prefetch(self._batch_size * 5)

            self._train_init_op = dataset.make_initializable_iterator()
            self._train_next_op = self._train_init_op.get_next('next_batch')

    def _test_dataset(self, name_scope='test_dataset'):
        with tf.name_scope(name_scope):
            dataset = dataset_ops.Dataset.from_tensor_slices((self.x_test, self.y_test))
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.repeat().batch(10000).prefetch(10000)

            self._test_init_op = dataset.make_initializable_iterator()
            self._test_next_op = self._train_init_op.get_next('next_batch')

    @staticmethod
    def _dense_layer(input, num_units=256, dropout_rate=None, is_training=True, name_scope='hidden_layer1'):
        with tf.name_scope(name_scope):
            kernel_initializer = tf.glorot_normal_initializer()
            hidden_layer = tf.layers.dense(inputs=input, units=num_units, activation=tf.nn.relu,
                                           kernel_initializer=kernel_initializer, bias_initializer=tf.zeros_initializer())

            if dropout_rate is not None and isinstance(dropout_rate, float) and 0.0 <= dropout_rate < 1.0:
                hidden_layer = tf.layers.dropout(hidden_layer, rate=dropout_rate, training=is_training)

        return hidden_layer

    @staticmethod
    def _conv_layer(inputs, out_filters=30, name_scope='conv_layer'):
        with tf.name_scope(name_scope):
            bias_initializer=tf.zeros_initializer()
            kernel_initializer=tf.orthogonal_initializer()
            conv_output = tf.layers.conv2d(inputs=inputs, filters=out_filters, use_bias=True, kernel_size=3,
                             bias_initializer=bias_initializer,
                             kernel_initializer=kernel_initializer)
            conv_output = tf.layers.conv2d(inputs=conv_output, filters=out_filters, use_bias=True, kernel_size=3,
                             bias_initializer=bias_initializer,
                             kernel_initializer=kernel_initializer)
            conv_output = tf.layers.conv2d(inputs=conv_output, filters=out_filters, use_bias=True, kernel_size=3,
                             bias_initializer=bias_initializer,
                             kernel_initializer=kernel_initializer)
            conv_output = tf.nn.relu(conv_output)
            output = tf.layers.max_pooling2d(conv_output, pool_size=3, strides=(2, 2))
        return output

    def model(self):
        print(self._train_next_op, self._test_next_op)
        next_op = tf.cond(self._is_training, lambda: self._train_next_op, lambda: self._test_next_op)
        input = tf.cast(tf.expand_dims(input=next_op[0]/255, axis=-1), dtype=tf.float32)
        # input = tf.cast(tf.reshape(tensor=next_op[0] / 255, shape=[128, 28, 28, 1]), dtype=tf.float32)

        labels = tf.one_hot(indices=next_op[1], depth=10)
        conv_output = self._conv_layer(inputs=input, out_filters=32, name_scope="conv_layer")
        conv_output = tf.layers.flatten(conv_output)
        hidden1_output = self._dense_layer(conv_output, num_units=512, dropout_rate=0.2, name_scope='hidden_layer1')
        hidden2_output = self._dense_layer(hidden1_output, num_units=512, dropout_rate=0.2, name_scope='hidden_layer2')
        hidden3_output = self._dense_layer(hidden2_output, num_units=512, dropout_rate=0.2, name_scope='hidden_layer3')
        hidden4_output = self._dense_layer(hidden3_output, num_units=512, dropout_rate=0.2, name_scope='hidden_layer4')
        logits = self._dense_layer(hidden4_output, num_units=10, dropout_rate=None, name_scope='logits')

        return logits, labels

    def train(self):
        learning_rate = self._learning_rate
        init_learning_rate = self._init_learning_rate

        logits, labels = self.model()

        equals = tf.cast(tf.equal(tf.argmax(labels, axis=1), tf.argmax(logits, axis=1)), dtype=tf.float32)
        accuracy = tf.reduce_mean(equals)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)

        with Session() as sess:
            import time
            sess.run(tf.global_variables_initializer())
            tmp_learning_rate = init_learning_rate

            feed_dict = {self._is_training: True, self._batch_size:128}
            sess.run([self._train_init_op.initializer, self._test_init_op.initializer], feed_dict=feed_dict)

            start_time = time.time()
            for iteration in range(100000):
                if iteration % 25000 == 0 or iteration % 50000 == 0 or iteration % 75000 == 0:
                    tmp_learning_rate *= 0.1

                feed_dict = {self._is_training: True, learning_rate: tmp_learning_rate, self._batch_size: 128}
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                train_accuracy_value = sess.run([accuracy], feed_dict=feed_dict)

                if iteration % 1000 == 0:
                    sess.run(self._test_init_op.initializer)

                    feed_dict = {self._is_training: False, learning_rate: tmp_learning_rate, self._batch_size: 10000}

                    test_accuracy_value = sess.run([accuracy], feed_dict=feed_dict)
                    print('iteration: %d, learning rate: %f, loss: %f, train accuracy: %f, test accuracy: %f' %
                          (iteration, tmp_learning_rate, loss_value, train_accuracy_value[0], test_accuracy_value[0]))

            print('total time: %d' % (time.time() - start_time))


if __name__=='__main__':
    model = MnistMlp(init_learning_rate=0.1)
    model.train()