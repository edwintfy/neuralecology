#!/usr/bin/env python
# coding: utf-8

# Neural Dynamic Occupancy Models for Breeding Bird Survey Data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import itertools
import re
import multiprocessing

import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import os
import random
import csv

# custom modules
import dataset
import utils

# Set device to the gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.deterministic = True
cudnn.benchmark = True

# Set seed for reproducibility
def set_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(2147483647)


## Reading and parsing the Breeding Bird Survey Data

bbs_train = dataset.BBSData(dataset.bbs.query("group == 'train'"))
bbs_valid = dataset.BBSData(dataset.bbs.query("group == 'validation'"))

# loading one example
ex_sp, ex_gn, ex_fm, ex_or, ex_l1, ex_x, ex_x_p, ex_y = bbs_train[0]

# checking shapes
nx = ex_x.shape[0]
nt = ex_y.shape[0]
nsp = len(dataset.cat_ix["english"].keys())
ngn = len(dataset.cat_ix["genus"].keys())
nfm = len(dataset.cat_ix["family"].keys())
nor = len(dataset.cat_ix["order"].keys())
nx_p = ex_x_p.shape[-1]

print(f"Dimension of x is {nx}")
print(f"Number of time steps is {nt}")
print(f"Number of species is {nsp}")
print(f"Number of genera is {ngn}")
print(f"Number of families is {nfm}")
print(f"Number of orders is {nor}")
print(f"Dimension of detection covariate vector is {nx_p}")


class MultiNet(nn.Module):
    """ Multispecies neural dynamic occupancy model. """

    embed_dim = 2 ** 6
    h_dim = 2 ** 5
    activation = nn.LeakyReLU()

    def __init__(self, num_sp, num_gn, num_fm, num_or, num_l1, nx, nt, nx_p):
        """Initialize a network object. 
        
        Args
        ----
        num_sp (int): number of species
        num_gn (int): number of genera
        num_fm (int): number of families
        num_or (int): number of orders
        num_l1 (int): number of level 1 ecoregions
        nx (int): number of route-level input features
        nt (int): number of timesteps (years)
        nx_p (int): number of detection/survey-level covariates
        """
        # Define embeddings and fully connected layers
        super(MultiNet, self).__init__()
        self.nt = nt
        self.nx_p = nx_p

        # shared layers
        self.l1_embed = nn.Embedding(num_l1, self.embed_dim)
        self.sp_embed = nn.Embedding(num_sp, self.embed_dim)
        self.gn_embed = nn.Embedding(num_gn, self.embed_dim)
        self.fm_embed = nn.Embedding(num_fm, self.embed_dim)
        self.or_embed = nn.Embedding(num_or, self.embed_dim)

        self.to_h = nn.Sequential(
            nn.Linear(self.embed_dim + nx, self.h_dim),
            self.activation,
            nn.Linear(self.h_dim, self.h_dim),
            self.activation,
            nn.Linear(self.h_dim, self.h_dim),
            self.activation,
        )

        # component hidden layers
        self.to_h_psi0 = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim), self.activation
        )

        self.to_h_t = nn.RNN(
            input_size=self.h_dim,
            hidden_size=self.h_dim * 3,
            num_layers=2,
            batch_first=True,
        )

        # weight layers
        self.to_psi0_weights = self.sp_embed_to_w(self.h_dim)
        self.to_p_weights = self.sp_embed_to_w(self.h_dim + nx_p)
        self.to_phi_weights = self.sp_embed_to_w(self.h_dim)
        self.to_gamma_weights = self.sp_embed_to_w(self.h_dim)

    def sp_embed_to_w(self, dim_out):
        """ Mapping from an embedding to a weight vector. 
        
        Args
        ----
        dim_out (int): the dimension of the weight vector
        """
        layers = nn.Sequential(
            nn.Linear(self.embed_dim * 4, self.h_dim),
            self.activation,
            nn.Linear(self.h_dim, dim_out),
        )
        return layers

    def forward(self, sp, gn, fm, order, l1, x, x_p):
        """ Make a forward pass for the model.
        
        Args
        ----
        sp (tensor): integer indices for species
        gn (tensor): integer indices for genera
        fm (tensor): integer indices for families
        order (tensor): integer indices for orders
        l1 (tensor): integer indices for level 1 ecoregion
        x (tensor): route-level features
        x_p (tensor): survey-level features
        
        Return
        ------
        A dictionary of parameters
        """

        l1_vec = self.l1_embed(l1)
        sp_vec = torch.cat(
            (
                self.sp_embed(sp),
                self.gn_embed(gn),
                self.fm_embed(fm),
                self.or_embed(order),
            ),
            -1,
        )

        h = self.to_h(torch.cat((l1_vec, x), -1))
        h_psi0 = self.to_h_psi0(h)
        h_t, _ = self.to_h_t(h.view(-1, 1, self.h_dim).repeat(1, self.nt, 1))
        h_t = self.activation(h_t.view(-1, self.nt, self.h_dim, 3))
        h_phi = h_t[:, :-1, :, 0]
        h_gamma = h_t[:, :-1, :, 1]
        h_p = h_t[:, :, :, 2]

        phi_weights = self.to_phi_weights(sp_vec)
        phi = torch.sigmoid(
            torch.bmm(h_phi, phi_weights.view(-1, self.h_dim, 1)).view(
                -1, self.nt - 1
            )
        )

        gamma_weights = self.to_gamma_weights(sp_vec)
        gamma = torch.sigmoid(
            torch.bmm(h_gamma, gamma_weights.view(-1, self.h_dim, 1)).view(
                -1, self.nt - 1
            )
        )

        # for each year, concatenate the route embedding with survey condtion data
        p_weights = self.to_p_weights(sp_vec)
        p_t = list()
        for t in range(self.nt):
            # stack hidden state with survey data
            p_t.append(
                torch.bmm(
                    torch.cat((h_p[:, t, :], x_p[:, t, :]), -1).view(
                        -1, 1, self.h_dim + self.nx_p
                    ),
                    p_weights.view(-1, self.h_dim + self.nx_p, 1),
                ).squeeze()
            )
        logit_p = torch.stack(p_t, dim=-1)

        psi0_weights = self.to_psi0_weights(sp_vec)  # (batch, h_dim)
        psi0 = torch.sigmoid(
            # take the row-wise dot product (route vector * species weights)
            torch.bmm(
                h_psi0.view(-1, 1, self.h_dim),
                psi0_weights.view(-1, self.h_dim, 1),
            )
        ).view(-1, 1)

        out_dict = {
            "phi": phi,
            "gamma": gamma,
            "psi0": psi0,
            "logit_p": logit_p,
            "phi_weights": phi_weights,
            "gamma_weights": gamma_weights,
            "psi0_weights": psi0_weights,
            "p_weights": p_weights,
            "h_p": h_p,
            "h_psi0": h_psi0,
            "h_gamma": h_gamma,
            "h_phi": h_phi,
        }
        return out_dict


# Training the neural network
idlist = []
train = []
valid = []
eplist = []

# Epoch changed from 5 to 100
n_epoch = 100
batch_size = 2 ** 11
print(f"Batch size is {batch_size}")

train_loader = DataLoader(
    bbs_train, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()
)
valid_loader = DataLoader(
    bbs_valid, batch_size=batch_size * 4, num_workers=multiprocessing.cpu_count(), shuffle=True
)

net = MultiNet(
    num_sp=nsp,
    num_gn=ngn,
    num_fm=nfm,
    num_or=nor,
    num_l1=len(dataset.cat_ix["L1_KEY"]),
    nx=nx,
    nt=nt,
    nx_p=nx_p,
)
net.to(device)

optimizer = optim.Adam(net.parameters(), weight_decay=1e-3)

train_loss = list()
valid_loss = list()

print("Training the multispecies model...")
i = 0
for epoch in range(n_epoch):
    print("Starting epoch " + str(epoch + 1) + " of " + str(n_epoch) + "...")
    eplist.append(epoch)
    i += 1
    idlist.append(i)
    with torch.enable_grad():
        net.train()
        train_loss.append(
            utils.fit_epoch(
                net, train_loader, training=True, optimizer=optimizer, pb=True
            )
        )
    train.append(np.mean(train_loss[-1]))
    with torch.no_grad():
        net.eval()
        valid_loss.append(
            utils.fit_epoch(net, valid_loader, training=False, pb=False)
        )
        print(np.mean(valid_loss[-1]))
    valid.append(np.mean(valid_loss[-1]))

print("Validation loss for the multispecies model:")
print([np.mean(l) for l in valid_loss])

print('Length idlist: ', len(idlist))            
print('Length eplist: ', len(eplist))
print('Length train: ', len(train))
print('Length valid: ', len(valid))

print('Write file')
with open('out_rnn_epoch_100.csv', 'w+', newline='') as file:
    writer= csv.writer(file)
    writer.writerows(zip(idlist,eplist,train,valid))

