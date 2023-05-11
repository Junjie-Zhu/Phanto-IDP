import csv
import json
import os
import time
from os.path import isfile, join
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def splitDataset(dataset, train_dirs, val_dirs, test_dirs,
                 collate_fn=default_collate,
                 batch_size=64,
                 num_workers=1,
                 pin_memory=False):
    train_indices = [i for i, row in enumerate(dataset.id_prop_data) if row in train_dirs]
    val_indices = [i for i, row in enumerate(dataset.id_prop_data) if row in val_dirs]
    test_indices = [i for i, row in enumerate(dataset.id_prop_data) if row in test_dirs]

    # Sample elements randomly from a given list of indices, without replacement.
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              collate_fn=collate_fn, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            collate_fn=collate_fn, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(dataset, batch_size=batch_size,
                             collate_fn=collate_fn, sampler=test_sampler,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


def collate_pool(dataset_list):
    N = max([x[0][0].size(0) for x in dataset_list])  # max atoms
    A = max([len(x[1][0]) for x in dataset_list])  # max amino in protein
    M = dataset_list[0][0][1].size(1)  # num neighbors are same for all so take the first value
    B = len(dataset_list)  # Batch size
    h_b = dataset_list[0][0][1].size(2)  # Edge feature length

    final_protein_atom_fea = torch.zeros(B, N)
    final_nbr_fea = torch.zeros(B, N, M, h_b)
    final_nbr_fea_idx = torch.zeros(B, N, M, dtype=torch.long)
    
    final_target_n, final_target_ca, final_target_c = torch.zeros(B, A, 3), torch.zeros(B, A, 3), torch.zeros(B, A, 3)

    for i, ((protein_atom_fea, nbr_fea, nbr_fea_idx, atom_amino_idx), (target_n, target_ca, target_c), protein_id) in enumerate(
            dataset_list):
        final_protein_atom_fea[i] = protein_atom_fea.squeeze()
        final_nbr_fea[i] = nbr_fea
        final_nbr_fea_idx[i] = nbr_fea_idx
        
        final_target_n[i], final_target_ca[i], final_target_c[i] = target_n, target_ca, target_c

    return (final_protein_atom_fea, final_nbr_fea, final_nbr_fea_idx), \
           (final_target_n, final_target_ca, final_target_c)


class AtomInitializer(object):

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {key: value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        counter = 0
        for key, _ in elem_embedding.items():
            self._embedding[key] = counter
            counter += 1


class ProteinDataset(Dataset):

    def __init__(self, pkl_dir, protein_dir, atom_init_filename, random_seed=123):
        assert os.path.exists(pkl_dir), '{} does not exist!'.format(pkl_dir)
        assert os.path.exists(protein_dir), '{} does not exist!'.format(protein_dir)

        self.pkl_dir = pkl_dir
        self.protein_dir = protein_dir

        protein_atom_init_file = os.path.join(self.pkl_dir, atom_init_filename)
        assert os.path.exists(protein_atom_init_file), '{} does not exist!'.format(protein_atom_init_file)
        self.ari = AtomCustomJSONInitializer(protein_atom_init_file)
        self.gdf = GaussianDistance(dmin=0, dmax=16, step=8)

        all_pdb_files = [file for file in os.listdir(self.protein_dir)
                         if isfile(join(self.protein_dir, file)) and file.endswith('.pdb')]
        random.seed(random_seed)
        random.shuffle(all_pdb_files)

        self.id_prop_data = []
        self.ca_crd, self.c_crd, self.n_crd = [], [], []

        print("Getting target coordinates")
        time_start = time.time()
        i = 0
        for files in all_pdb_files:
            if files != 'PaaA2.md.1.pdb':
                self.ca_crd.append([])
                self.c_crd.append([])
                self.n_crd.append([])
                fopen = open(os.path.join(self.protein_dir, files), "r")
                for lines in fopen.readlines():
                    if len(lines) >= 60:
                        if lines[13] == 'C' and lines[14] == 'A':
                            self.ca_crd[i].append([float(lines[30:38]), float(lines[38:46]), float(lines[46:54])])
                        elif lines[13] == 'C' and lines[14] == ' ':
                            self.c_crd[i].append([float(lines[30:38]), float(lines[38:46]), float(lines[46:54])])
                        elif lines[13] == 'N' and lines[14] == ' ':
                            self.n_crd[i].append([float(lines[30:38]), float(lines[38:46]), float(lines[46:54])])
                fopen.close()
                self.id_prop_data.append(files)
                i += 1
        print("Processed successfully, using time: ", time.time() - time_start)

    def __len__(self):
        return len(self.id_prop_data)

    def __getitem__(self, idx):
        return self.get_idx(idx)

    def get_idx(self, idx):
        protein_id = self.id_prop_data[idx]

        with open(self.pkl_dir + '_' + protein_id.replace('.pdb', '') + '.pkl', 'rb') as f:
            protein_atom_fea = torch.Tensor(np.vstack([self.ari.get_atom_fea(atom) for atom in pickle.load(
                f)]))  # Atom features (here one-hot encoding is used)
            nbr_info = pickle.load(f)  # Edge features for each atom in the graph
            nbr_fea_idx = torch.LongTensor(pickle.load(f))  # Edge connections that define the graph

            atom_amino_idx = torch.LongTensor(pickle.load(
                f))  # Mapping that denotes which atom corresponds to which amino residue in the protein graph 
            protein_id = pickle.load(f)

            nbr_fea = torch.Tensor(nbr_info)

            target_n = torch.FloatTensor(self.n_crd[idx])
            target_ca = torch.FloatTensor(self.ca_crd[idx])
            target_c = torch.FloatTensor(self.c_crd[idx])
 
        return (protein_atom_fea, nbr_fea, nbr_fea_idx, atom_amino_idx), (target_n, target_ca, target_c), protein_id
