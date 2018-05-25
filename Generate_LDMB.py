#!/usr/bin/env python3.5
# -*- coding: UTF-8 -*-

from ImageServer import ImageServer
import numpy as np
import os
import lmdb
from caffe.proto import caffe_pb2
import caffe


def loadData(trainSet, validationSet):
    Xtrain = trainSet.imgs
    Xvalid = validationSet.imgs
    Ytrain = getLabelsForDataset(trainSet)
    Yvalid = getLabelsForDataset(validationSet)
    generate_data_lmdb(Xtrain, train=True)
    generate_data_lmdb(Xvalid, train=False)
    generate_label_lmdb(Ytrain, train=True)
    generate_label_lmdb(Yvalid, train=False)

def getLabelsForDataset(imageServer):
    nSamples = imageServer.gtLandmarks.shape[0]
    nLandmarks = imageServer.gtLandmarks.shape[1]

    y = np.zeros((nSamples, nLandmarks*2), dtype=np.float32)
    for i in range(nSamples):
        y[i,:] = imageServer.gtLandmarks[i].flatten()

    return y

def generate_data_lmdb(imgs, train=True):
    lmdb_file = './data/data_val_lmdb'
    if train:
        lmdb_file = './data/data_train_lmdb'
    os.system('rm -rf '+ lmdb_file)
    # lmdb对于数据进行的是先缓存后一次性写入从而提高效率，因此定义一个batch_size控制每次写入的量。
    batch_size = 200
    # map_size定义最大空间
    lmdb_env = lmdb.open(lmdb_file, map_size=int(1e12))
    # 打开数据库的句柄
    lmdb_txn = lmdb_env.begin(write=True)
    # 这是caffe中定义数据的重要类型
    datum = caffe_pb2.Datum()
    nSamples = imgs.shape[0]
    label = 0
    for i in range(nSamples):
        data = imgs[i]
        datum = caffe.io.array_to_datum(data, label)
        keystr = '{:0>8d}'.format(i)
        lmdb_txn.put(keystr.encode('ascii'), datum.SerializeToString())

        if (i+1) % batch_size == 0:
            lmdb_txn.commit()
            lmdb_txn = lmdb_env.begin(write=True)
            print('data batch {} writen'.format(i+1))

    lmdb_txn.commit()
    lmdb_env.close()

def generate_label_lmdb(labels, train=True):
    all_labels = []
    key = 0
    lmdb_file = './data/label_val_lmdb'
    if train:
        lmdb_file = './data/label_train_lmdb'
    os.system('rm -rf '+ lmdb_file)
    batch_size = 200
    lmdb_env = lmdb.open(lmdb_file, map_size=int(1e12))
    lmdb_txn = lmdb_env.begin(write=True)
    datum = caffe_pb2.Datum()
    nSamples = labels.shape[0]
    label = 0
    for i in range(nSamples):
        datum.channels = labels[i].shape[0]
        datum.height = 1
        datum.width = 1
        data = labels[i].reshape(datum.channels, 1, 1)
        datum = caffe.io.array_to_datum(data, label)
        keystr = '{:0>8d}'.format(i)
        lmdb_txn.put(keystr.encode('ascii'), datum.SerializeToString())

        if (i+1) % batch_size == 0:
            lmdb_txn.commit()
            lmdb_txn = lmdb_env.begin(write=True)
            print('label batch {} writen'.format(i+1))

    lmdb_txn.commit()
    lmdb_env.close()


datasetDir = "./data/"

trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=44740_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz")
validationSet = ImageServer.Load(datasetDir + "dataset_nimgs=100_perturbations=[]_size=[112, 112].npz")
loadData(trainSet, validationSet)
