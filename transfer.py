import argparse
import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.autograd import Variable
# from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from torch.cuda.amp import GradScaler, autocast

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# writer = SummaryWriter('./TensorBoardX/')


# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=64):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 32, (1, 17), (1, 1)),
            nn.Conv2d(32, 64, (1, 13), (1, 1)),
            # nn.Conv2d(64, 128, (1, 3), (1, 1)),
            nn.Conv2d(64, 64, (19, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 30), (1, 5)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(64, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=16,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(1856, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.5), # changed from 0.3..
            nn.Linear(32, 5) # i changed this to 5 because 5 classes .. ?
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=64, depth=8, n_classes=5, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class ExP():
    def __init__(self):
        super(ExP, self).__init__()
        self.batch_size = 4
        self.gradient_accumulations = 16        
        self.n_epochs = 200
        self.c_dim = 5 # ummm does this refer to classes ? who fking knows lol
        self.lr = 0.0001
        self.b1 = 0.5 # wtf do these mean lol :P
        self.b2 = 0.999
        self.dimension = (19, 200) # erm. awesum used to be (190, 50) i changed ? but dunno if that makes sense tbh
        # self.subj = subj # pass in the name of the subject (?) because each subject gets a diff model :D

        self.start_epoch = 0
        self.root = '/Preprocessing/'

        # self.log_write = open("./results/log_subject%d.txt" % self.subjList, "w") #ERMMMMMMM

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = Conformer().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()

        self.scaler = GradScaler()
        # summary(self.model, (1, 22, 1000))


    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):  
        aug_data = []
        aug_label = []
        for cls4aug in range(5):
            cls_idx = np.where(label == cls4aug)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 19, 200))
        
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    print(ri, rj)
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    wtf = tmp_data[rand_idx[rj], :, :, rj * 25:(rj + 1) * 25]
                    print(wtf.shape)
                    tmp_aug_data[ri, :, :, rj * 25:(rj + 1) * 25] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 25:(rj + 1) * 25]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label-1).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_source_data(self, features, labels, test_size=0.3): # reading the specific subject's data from mneList
        
        features = features.reshape(features.shape[0], 1, features.shape[1], features.shape[2])

        # split and shuffle data
        self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(features, labels, test_size=test_size)

        shuffle_num = np.random.permutation(len(self.train_data))
        self.train_data = self.train_data[shuffle_num, :, :, :]
        self.train_label = self.train_label[shuffle_num]

        # standardize #UMMM SHOULD I BE DOING THIS ?????
        target_mean = np.mean(self.train_data)
        target_std = np.std(self.train_data)
        self.train_data = (self.train_data- target_mean) / target_std
        self.test_data = (self.test_data - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.train_data, self.train_label, self.test_data, self.test_label


    def train(self, img, label):

        img = torch.from_numpy(img) 
        label = torch.from_numpy(label) 

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        # test_data = torch.from_numpy(test_data)
        # test_label = torch.from_numpy(test_label)
        # test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        # self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # test_data = Variable(test_data.type(self.Tensor))
        # test_label = Variable(test_label.type(self.LongTensor))

        # bestAcc = 0
        # averAcc = 0
        # acc = 0
        # num = 0
        # Y_true = 0
        # Y_pred = 0

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        for e in range(self.n_epochs):
            # in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # data augmentation # skipping this for now cuz i tbh dunno whats going on 
                # aug_data, aug_label = self.interaug(self.train_data, self.train_label)
                # img = torch.cat((img, aug_data))
                # label = torch.cat((label, aug_label))
                with autocast():
                    tok, outputs = self.model(img)
                    loss = self.criterion_cls(outputs, label) 

                self.optimizer.zero_grad()

                self.scaler.scale(loss/self.gradient_accumulations).backward()

                if (i + 1) % self.gradient_accumulations == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.model.zero_grad()
            print('Epoch:', e)
            
        del img, label
            # out_epoch = time.time()


            # test process
            # if (e + 1) % 1 == 0:
            #     self.model.eval()
            #     Tok, Cls = self.model(test_data)


            #     loss_test = self.criterion_cls(Cls, test_label)
            #     y_pred = torch.max(Cls, 1)[1]
            #     acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
            #     train_pred = torch.max(outputs, 1)[1]
            #     train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

            #     print('Epoch:', e,
            #           '  Train loss: %.6f' % loss.detach().cpu().numpy(),
            #           '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
            #           '  Train accuracy %.6f' % train_acc,
            #           '  Test accuracy is %.6f' % acc)

            #     # self.log_write.write(str(e) + "    " + str(acc) + "\n")
            #     if e > self.n_epochs * 0.75: # record the last few epochs.
            #         num = num + 1
            #         averAcc = averAcc + acc
            #     if acc > bestAcc:
            #         bestAcc = acc
            #         Y_true = test_label
            #         Y_pred = y_pred
                    
        # return (bestAcc, averAcc / num, acc)

    def test(self, test_data, test_label):
        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        acc = 0

        self.model.eval()
        with torch.no_grad():
            Tok, Cls = self.model(test_data)
            loss_test = self.criterion_cls(Cls, test_label)
            y_pred = torch.max(Cls, 1)[1]

            acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))

        print('  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
              '  Test accuracy is %.6f' % acc)

        return acc


def load_data(affix='_raw'):
    if affix == 'ecog':
        affix = '_raw'
        mneList = []
        for subj in ['bp', 'cc', 'ht', 'jp', 'mv', 'wc', 'zt']:
            features = np.load(f'pickles/{subj}_features{affix}.npy')[:, :40, :]
            labels = np.load(f'pickles/{subj}_labels{affix}.npy')
            mneList.append((subj, features, labels))
        return mneList
    else:
        mneList = []
        for letter in ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I']:
            features = np.load(f'pickles/subj_{letter}_features{affix}.npy')
            labels = np.load(f'pickles/subj_{letter}_labels{affix}.npy')
            features = features[:, [0, 1, 10, 2, 16, 3, 11, 12, 4, 17, 5, 13, 14, 6, 18, 7, 15, 8, 9]]
            mneList.append((f'subj_{letter}', features, labels))
        return mneList

def transfer():
    fileList = load_data('_raw')
    fileName = 'log_7_31.txt'
    # file = open('log.txt', 'w')

    facc = []

    for subj_num in range(7,8):
        torch.cuda.empty_cache()

        test_subj = fileList[subj_num]
        newFileList = fileList.copy()
        newFileList.remove(test_subj)
        newFileList = ([features for subj, features, labels in newFileList], [labels for subj, features, labels in newFileList])
        features = np.concatenate(newFileList[0])
        labels = np.concatenate(newFileList[1])
        idx = np.random.permutation(features.shape[0])
        features, labels = features[idx], labels[idx]
        test_data = test_subj[1]
        test_label = test_subj[2]
        
        # train on all subjects except one
        conformer = ExP()
        print(test_subj[0])

        features_reshape = features.reshape(features.shape[0], 1, features.shape[1], features.shape[2])
        test_reshape = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2])
        
        conformer.train(features_reshape, labels)
        acc = conformer.test(test_reshape, test_label)
        with open(fileName, 'a') as f:
            f.write(f'acc after training on all subjects {acc}\n')
        features, labels, test_data, test_label = conformer.get_source_data(test_data, test_label, 0.7)
        conformer.train(features, labels)
        acc = conformer.test(test_data, test_label)
        with open(fileName, 'a') as f:
            f.write(f'acc after fine tuning on 30% {acc}\n')
        facc.append(acc)
    with open(fileName, 'a') as f:
        f.write(f'final accuracies after fine tuning on 30%: {facc}\n')

transfer()