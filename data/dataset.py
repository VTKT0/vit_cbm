## This file declares the PyTorch dataloader we use


import logging
logger = logging.getLogger(__name__)

import numpy as np
import json
import openslide
import time
import pickle
import os
import glob
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms

class WSITileDataset(Dataset):


    def __init__(self,
    dataset # an Datframe of faetures
    ,config # a config file

    ):

        self.cpt_ids = config['concepts']['cpt_ids']
        self.bag_size = config['data']['bag_size']
        self.paitent_limit = config['data']['paitent_limit']
        self.bag_size = config['data']['bag_size']
        self.resample_small_bags = config['data']['resample_small_bags']
        self.has_static_bags = config['data']['static_bags']
        self.dataset = dataset
        self.np_bags_dir = config['data']['np_bags_dir']
        self.id_list = self.dataset['ID'].to_list()[:self.paitent_limit]

        # check all bags exist
        for id in self.id_list:
            if not os.path.exists(os.path.join(self.np_bags_dir, f'{id}.npy')):
                raise ValueError(f"Bag for {id} does not exist")


        self.rng = np.random.default_rng(seed=42) # for selecting tiles

        # if we want static bags, then calcualte upfront
        if self.has_static_bags:
            self.static_bags ={}
            for id in self.id_list:

                with open(os.path.join(self.np_bags_dir, 'manifest.json'), 'r') as f:
                    manifest = json.load(f)

                if id not in manifest.keys():
                    raise ValueError(f"Bag for {id} does not exist")

                num_tiles = manifest[id]['total_tiles']

                if num_tiles >= self.bag_size:
                    indices = self.rng.choice(num_tiles, self.bag_size, replace=False)
                else:
                    indices = self.rng.choice(num_tiles, self.bag_size, replace=True)

                self.static_bags[id] = indices

                
    def _get_concepts(self, ws_id):
        """Get concept values for a given WSI ID."""
        cpts = []
        for c in self.cpt_ids:
            cpts.append(
                self.dataset.loc[
                    self.dataset['ID'] == ws_id, c
                ].iloc[0]
            )
        return cpts

    def _get_survival(self, ws_id):
        """ Get survival data for Cox regression (event,survtime) """
        
        row = self.dataset.loc[self.dataset['ID'] == ws_id].iloc[0]

        event = row['event']
        survtime = row['time']

        if pd.isna(event):
            raise ValueError(f"Missing event value for {ws_id}")
        if pd.isna(survtime) or survtime <= 0:
            raise ValueError(f"Invalid survival time for {ws_id}: {survtime}")

        return int(event), float(survtime)

    def _get_tiles_from_np_bags(self, id):
        """Get tiles for a given WSI ID from prestored np bags"""

        bag_np = np.load(os.path.join(self.np_bags_dir, f'{id}.npy'))
        
        if self.has_static_bags:
            indices = self.static_bags[id]
        else:
            n_tiles = bag_np.shape[0]

            if n_tiles >= self.bag_size:

                #sample without replacement
                indices = self.rng.choice(n_tiles, self.bag_size, replace=False)

            elif n_tiles < self.bag_size:

                # sample with replacement
                indices = self.rng.choice(n_tiles, self.bag_size, replace=True)

        bag_tensor = torch.tensor(bag_np[indices], dtype=torch.float32)

        return bag_tensor

        
    def __len__(self):
        return len(self.id_list)


    def __getitem__(self,idx):

        id = self.id_list[idx]
        event, survtime = self._get_survival(id)
        
        cpts = self._get_concepts(id)

        event = torch.tensor(event, dtype=torch.float32)
        survtime = torch.tensor(survtime, dtype=torch.float32)
        cpts = torch.tensor(cpts, dtype=torch.float32)

        tiles = self._get_tiles_from_np_bags(id)
        
        return tiles, (event, survtime), cpts, id


        