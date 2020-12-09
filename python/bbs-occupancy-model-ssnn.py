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



# Training the neural network

n_epoch = 5
batch_size = 2 ** 11
print(f"Batch size is {batch_size}")

train_loader = DataLoader(
    bbs_train, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()
)
valid_loader = DataLoader(
    bbs_valid, batch_size=batch_size * 4, num_workers=multiprocessing.cpu_count(), shuffle=True
)



# Training species-specific models
class OneSpeciesNet(nn.Module):
    """ Single species neural dynamic occupancy model. """

    embed_dim = 2 ** 3
    h_dim = 2 ** 2
    activation = nn.LeakyReLU()

    def __init__(self, num_l1, nx, nt, nx_p):
        """Initialize a network object. 
        
        Args
        ----
        num_l1 (int): number of level 1 ecoregions
        nx (int): number of route-level input features
        nt (int): number of timesteps (years)
        nx_p (int): number of detection/survey-level covariates
        """

        # Define embeddings and fully connected layers
        super(OneSpeciesNet, self).__init__()
        self.nt = nt
        self.nx_p = nx_p

        # shared layers
        self.l1_embed = nn.Embedding(num_l1, self.embed_dim)

        self.to_h = nn.Sequential(
            nn.Linear(self.embed_dim + nx, self.h_dim), self.activation
        )

        # component output layers
        self.to_psi0 = nn.Linear(self.h_dim, 1)
        self.to_p = nn.Linear(self.h_dim + nx_p, 1)
        self.to_phi = nn.Linear(self.h_dim, 1)
        self.to_gamma = nn.Linear(self.h_dim, 1)

    def forward(self, sp, gn, fm, order, l1, x, x_p):
        """ Make a forward pass for the model.

        Note that while species, genus, family, and order are provided as 
        inputs, these are ignored. (Keeping them allows the re-use of the 
        same dataset definition.)
        
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
        h = self.to_h(torch.cat((l1_vec, x), -1))

        phi = torch.sigmoid(self.to_phi(h)).repeat(1, self.nt - 1)
        gamma = torch.sigmoid(self.to_gamma(h)).repeat(1, self.nt - 1)
        psi0 = torch.sigmoid(self.to_psi0(h))

        # for each year, concatenate the route embedding with survey condtion data
        p_t = list()
        for t in range(self.nt):
            # stack hidden state with survey data
            p_t.append(self.to_p(torch.cat((h, x_p[:, t, :]), -1)))
        logit_p = torch.stack(p_t, dim=-1).squeeze()  # batch_size X nt
        out_dict = {
            "phi": phi,
            "gamma": gamma,
            "psi0": psi0,
            "logit_p": logit_p,
        }
        return out_dict


def fit_single_species_model(sp, model):
    """ Fit a single species model. 
    
    This function trains a single species model and writes the 
    results to a csv file.
    
    Args: 
    - sp (string): english species name
    - model (nn.Module): the model to use
    """
    sp_name_lower = re.sub(" ", "_", sp.lower())
    out_path = f"out/{sp_name_lower}_ssnet.csv"
    if os.path.isfile(out_path):
        return f"File already exists for {sp}. Moving on..."

    sp_train = dataset.BBSData(
        dataset.bbs.query("group == 'train' & english == @sp")
    )
    sp_valid = dataset.BBSData(
        dataset.bbs.query("group == 'validation' & english == @sp")
    )
    sp_train_loader = DataLoader(
        sp_train, batch_size=batch_size // 4, shuffle=True, num_workers=multiprocessing.cpu_count()
    )
    sp_valid_loader = DataLoader(
        sp_valid, batch_size=batch_size * 4, num_workers=multiprocessing.cpu_count(), shuffle=True
    )
    sp_net = model(
        num_l1=len(dataset.cat_ix["L1_KEY"]), nx=nx, nt=nt, nx_p=nx_p
    )
    sp_net.to(device)

    optimizer = optim.Adam(sp_net.parameters(), weight_decay=1e-3, lr=0.01)
    max_n_epoch = 100

    train_loss = list()
    valid_loss = list()
    for epoch in range(max_n_epoch):
        with torch.enable_grad():
            sp_net.train()
            train_loss.append(
                utils.fit_epoch(
                    sp_net,
                    sp_train_loader,
                    training=True,
                    optimizer=optimizer,
                    pb=False,
                )
            )
        with torch.no_grad():
            sp_net.eval()
            valid_loss.append(
                utils.fit_epoch(
                    sp_net, sp_valid_loader, training=False, pb=False
                )
            )
            if epoch > 2:
                valid_loss_diff = np.mean(valid_loss[-1]) - np.mean(
                    valid_loss[-2]
                )
                if valid_loss_diff >= -0.2:
                    break

    valid_arr = np.concatenate(valid_loss)
    train_arr = np.concatenate(train_loss)

    sp_df = dataset.bbs.query("english == @sp & group != 'test'")
    years = list(sp_df.filter(regex="^[0-9]{4}", axis=1))
    sp_ds = dataset.BBSData(sp_df)
    sp_loader = DataLoader(
        sp_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )
    sp_p = []
    sp_gamma = []
    sp_phi = []
    sp_psi0 = []
    for i_batch, xy in enumerate(sp_loader):
        _, out = utils.bbs_nll(xy, model=sp_net)
        sp_p.append(torch.sigmoid(out["logit_p"]).cpu().detach().numpy())
        sp_gamma.append(out["gamma"].cpu().detach().numpy())
        sp_phi.append(out["phi"].cpu().detach().numpy())
        sp_psi0.append(out["psi0"].cpu().detach().numpy())

    p_df = pd.DataFrame(np.concatenate(sp_p), columns=["p_" + y for y in years])
    gamma_df = pd.DataFrame(
        np.concatenate(sp_gamma), columns=["gamma_" + y for y in years[:-1]]
    )
    phi_df = pd.DataFrame(
        np.concatenate(sp_phi), columns=["phi_" + y for y in years[:-1]]
    )
    psi0_df = pd.DataFrame(np.concatenate(sp_psi0), columns=["psi0"])

    res = pd.concat(
        (
            sp_df.filter(
                regex="^[0-9]{4}$|sp.bbs|route_id|Lat|Lon", axis=1
            ).reset_index(),
            p_df,
            gamma_df,
            phi_df,
            psi0_df,
        ),
        axis=1,
    )
    res.to_csv(out_path, index=False, na_rep="NA")

    del sp_net
    del sp_train
    del sp_valid
    del sp_train_loader
    del sp_valid_loader


# iterate over species and fit a model for each
print("Training and saving single-species results")
for sp in tqdm(dataset.cat_ix["english"].keys()):
    fit_single_species_model(sp, model=OneSpeciesNet)

