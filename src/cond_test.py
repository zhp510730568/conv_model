import tensorflow as tf

from tensorflow.python.client.session import Session
from tensorflow.python.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

flag = tf.placeholder(tf.bool, shape=())

batch_size = 128

dataset_x = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset_x = dataset_x.shuffle(buffer_size=batch_size * 5)
dataset_x = dataset_x.repeat().batch(batch_size=batch_size)\
    .prefetch(buffer_size=batch_size * 5)

iter_x = dataset_x.make_initializable_iterator()
next_x = iter_x.get_next()

dataset_y = tf.data.Dataset.from_tensor_slices((x_test, y_test))
dataset_y = dataset_y.repeat().batch(10000)
iter_y = dataset_y.make_initializable_iterator()
next_y = iter_y.get_next()

output = tf.cond(flag, lambda: next_x, lambda: next_y)

y = output * 10
with Session() as sess:
    import time
    start_time = time.time()
    sess.run([iter_x.initializer, iter_y.initializer])
    for epoch in range(10000):
        feed_dict = {flag: True}
        sess.run(y, feed_dict=feed_dict)
        print('epoch: %d' % (epoch), y[0].shape)
        sess.run(iter_y.initializer)
        for _ in range(10):
            feed_dict = {flag: False}
            # test = sess.run(output, feed_dict=feed_dict)
            # print(test[1][0:100])
    print('total time: %d' % (time.time() - start_time))