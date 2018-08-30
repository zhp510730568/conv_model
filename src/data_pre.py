import numpy as np

path = '/home/zhangpengpeng/.keras/datasets/imdb.npz'
with np.load(path) as f:
    x_train, labels_train = f['x_train'], f['y_train']
    x_test, labels_test = f['x_test'], f['y_test']
    print(len(x_train[9]))
    print(labels_train[0])