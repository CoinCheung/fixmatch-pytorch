import os.path as osp
import pickle
import random
#  from queue import Queue
#  from threading import Thread
from multiprocessing import Process, Queue
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision
import transform as T



def load_data_train(L=250, dspth='./dataset'):
    datalist = [
        osp.join(dspth, 'cifar-10-batches-py', 'data_batch_{}'.format(i+1))
        for i in range(5)
    ]
    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    n_labels = L // 10
    data_x, label_x, data_u, label_u = [], [], [], []
    for i in range(10):
        indices = np.where(labels == i)[0]
        np.random.shuffle(indices)
        inds_x, inds_u = indices[:n_labels], indices[n_labels:]
        data_x += [
            data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            for i in inds_x
        ]
        label_x += [labels[i] for i in inds_x]
        data_u += [
            data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            for i in inds_u
        ]
        label_u += [labels[i] for i in inds_u]
    return data_x, label_x, data_u, label_u


def load_data_val(dspth='./dataset'):
    datalist = [
        osp.join(dspth, 'cifar-10-batches-py', 'test_batch')
    ]
    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data = [
        el.reshape(3, 32, 32).transpose(1, 2, 0)
        for el in data
    ]
    return data, labels


class Cifar10(Dataset):
    def __init__(self, data, labels, is_train=True):
        super(Cifar10, self).__init__()
        self.data, self.labels = data, labels
        assert len(self.data) == len(self.labels)
        mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        if is_train:
            self.trans_weak = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            self.trans_strong = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
        else:
            self.trans = T.Compose([
                T.Resize((32, 32)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        im, lb = self.data[idx], self.labels[idx]
        im = self.trans(im)
        return im, lb

    def __len__(self):
        return len(self.data)


class Cifar10Loader(object):
    def __init__(self, data, labels, batchsize, max_iter,
        is_train=True, num_worker=4
    ):
        self.data, self.labels = data, labels
        self.batchsize = batchsize
        self.max_iter = max_iter
        assert len(self.data) == len(self.labels)
        mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        if is_train:
            self.trans_weak = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                #  T.Normalize(mean, std),
                #  T.ToTensor(),
            ])
            self.trans_strong = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                #  T.Normalize(mean, std),
                #  T.ToTensor(),
            ])
        else:
            self.trans = T.Compose([
                T.Resize((32, 32)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
        self.to_tensor = T.Compose([
            T.Normalize(mean, std),
            T.ToTensor(),
        ])
        self.len = len(self.data)
        self.indices = list(range(self.len))
        random.shuffle(self.indices)

        all_num_imgs = self.batchsize * self.max_iter
        buffer_len = 640
        assert all_num_imgs % num_worker == 0
        self.idx_qs = [Queue(buffer_len) for el in range(num_worker)]
        self.loader_qs = [Queue(buffer_len) for el in range(num_worker)]
        gen_idx_thread = Process(target=self.gen_idx,
            args=(self.idx_qs, all_num_imgs))
        loader_threads = [
            Process(target=self.load_data,
                args=(self.idx_qs[el], self.loader_qs[el], all_num_imgs//num_worker))
            for el in range(num_worker)
        ]

        gen_idx_thread.start()
        for thread in loader_threads:
            thread.start()

    def fetch_batch(self):
        ims_weak, ims_strong, lbs = [], [], []
        count = 0
        finish = False
        while True:
            for lq in self.loader_qs:
                im_weak, im_strong, lb = lq.get()
                ims_weak.append(im_weak)
                ims_strong.append(im_strong)
                lbs.append(lb)
                count += 1
                if count >= self.batchsize:
                    finish = True
                    break
            if finish: break
        ims_weak = self.to_tensor(
            np.concatenate([el[None, ...] for el in ims_weak], axis=0))
        ims_strong = self.to_tensor(
            np.concatenate([el[None, ...] for el in ims_strong], axis=0))
        lbs = torch.tensor(lbs).long()
        return ims_weak, ims_strong, lbs


    def gen_idx(self, idx_qs, num_imgs):
        curr = 0
        for i in range(num_imgs):
            for q in idx_qs:
                q.put(self.indices[curr])
                curr += 1
                if curr >= self.len:
                    random.shuffle(self.indices)
                    curr = 0

    def load_data(self, idx_q, loader_q, all_num_imgs):
        for i in range(all_num_imgs):
            idx = idx_q.get()
            img_weak = self.trans_weak(self.data[idx])
            img_strong = self.trans_strong(self.data[idx])
            lb = self.labels[idx]
            loader_q.put((img_weak, img_strong, lb))



def get_train_loader(batch_size, mu, L, max_iter, root='dataset'):
    data_x, label_x, data_u, label_u = load_data_train(L=L, dspth=root)

    dl_x = Cifar10Loader(
        data=data_x,
        labels=label_x,
        batchsize=batch_size,
        max_iter=max_iter,
        num_worker=1,
        is_train=True
    )
    dl_u = Cifar10Loader(
        data=data_u,
        labels=label_u,
        batchsize=batch_size * mu,
        max_iter=max_iter,
        num_worker=2,
        is_train=True
    )
    return dl_x, dl_u


def get_val_loader(batch_size, num_workers, pin_memory=True, root='cifar10'):
    data, labels = load_data_val()
    ds = Cifar10(
        data=data,
        labels=labels,
        is_train=False
    )
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dl


class OneHot(object):
    def __init__(
            self,
            n_labels,
            lb_ignore=255,
        ):
        super(OneHot, self).__init__()
        self.n_labels = n_labels
        self.lb_ignore = lb_ignore

    def __call__(self, label):
        N, *S = label.size()
        size = [N, self.n_labels] + S
        lb_one_hot = torch.zeros(size)
        if label.is_cuda:
            lb_one_hot = lb_one_hot.cuda()
        ignore = label.data.cpu() == self.lb_ignore
        label[ignore] = 0
        lb_one_hot.scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(self.n_labels), *b]] = 0

        return lb_one_hot


if __name__ == "__main__":
    dl_x, dl_u = get_train_loader(64, 250, 2, 2)
    dl_x2 = iter(dl_x)
    dl_u2 = iter(dl_u)
    ims, lb = next(dl_u2)
    print(type(ims))
    print(len(ims))
    print(ims[0].size())
    print(len(dl_u2))
    for i in range(1024):
        try:
            ims_x, lbs_x = next(dl_x2)
            #  ims_u, lbs_u = next(dl_u2)
            print(i, ": ", ims_x[0].size())
        except StopIteration:
            dl_x2 = iter(dl_x)
            dl_u2 = iter(dl_u)
            ims_x, lbs_x = next(dl_x2)
            #  ims_u, lbs_u = next(dl_u2)
            print('recreate iterator')
            print(i, ": ", ims_x[0].size())
