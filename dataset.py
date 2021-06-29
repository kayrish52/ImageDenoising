import cv2
import glob
import h5py
import numpy as np
import os
import os.path
import random
from skimage.color import rgb2gray
import torch
import torch.utils.data as udata


def normalize(data):
    return data/255.


def im2patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    total_pat_num = patch.shape[1] * patch.shape[2]
    y = np.zeros([endc, win*win, total_pat_num], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            y[:, k, :] = np.array(patch[:]).reshape(endc, total_pat_num)
            k = k + 1
    return y.reshape([endc, win, win, total_pat_num])


def prepare_data(data_path, trSamples=100, valSamples=20):

    # train
    print('process training data')
    scales = 1
    files = glob.glob(os.path.join(data_path, '*', '*.jpg'))

    train_files = []
    for i in range(trSamples):
        idx = random.randrange(0, len(files))
        train_files.append(files.pop(idx))

    val_files = []
    for i in range(valSamples):
        idx = random.randrange(0, len(files))
        val_files.append(files.pop(idx))

    train_files.sort()
    h5f = h5py.File('NWPU_train.h5', 'w')
    train_num = 0
    for i in range(len(train_files)):
        Img = cv2.imread(train_files[i])
        Img = np.clip(np.float64(normalize(Img)), 0.00001, 1.0)

        Img = rgb2gray(Img)

        print("file: %s scale %.1f" % (train_files[i], scales))

        h5f.create_dataset(str(train_num), data=Img)
        train_num += 1

    h5f.close()

    # validation
    print('\nprocess validation data')
    val_files.sort()
    h5f = h5py.File('NWPU_val.h5', 'w')
    val_num = 0
    for i in range(len(val_files)):
        Img = cv2.imread(val_files[i])
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        Img = np.clip(np.float64(normalize(Img)), 0.00001, 1.0)

        Img = rgb2gray(Img)

        print("file: %s scale %.1f" % (val_files[i], scales))

        h5f.create_dataset(str(val_num), data=Img)
        val_num += 1

    h5f.close()

    print('training set, # samples %d' % train_num)
    print('val set, # samples %d' % val_num)


class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('NWPU_train.h5', 'r')
        else:
            h5f = h5py.File('NWPU_val.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('NWPU_train.h5', 'r')
        else:
            h5f = h5py.File('NWPU_val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)
