import csv, pickle, json, os
import argparse
import numpy as np
from os.path import isfile, join
from subprocess import call
from joblib import Parallel, delayed
from collections import defaultdict as ddict

parser = argparse.ArgumentParser(description='Argument Parser')

parser.add_argument('-datapath', default='../Traj/processed/', help='Directory where all protein pdb files exist')
parser.add_argument('-savepath', default='./data/pkl/', help='Directory where result is saved')
parser.add_argument('-cpp_executable', default='./preprocess/get_features',
                    help='Directory where cpp cpp_executable is located')
parser.add_argument('-groups20_filepath', default='./preprocess/groups20.txt',
                    help='Directory where groups20.txt is located')
parser.add_argument('-parallel_jobs', default=5, help='Number of threads to use for parallel jobs')
parser.add_argument('-get_json_files', default=True, help='Whether to fetch json files or not', action='store_true')
parser.add_argument('-get_pkl_flies', default=True, help='Whether to fetch pkl files or not', action='store_true')

args = parser.parse_args()

datapath = args.datapath
savepath = args.savepath
cpp_executable = args.cpp_executable
groups20_filepath = args.groups20_filepath
parallel_jobs = args.parallel_jobs
get_json_files = args.get_json_files
get_pkl_flies = args.get_pkl_flies
protein_id_prop_file = savepath + 'gsk/protein_id_prop.csv'
protein_atom_init_file = savepath + 'gsk/protein_atom_init.json'

protein_dirs = os.listdir(datapath)
all_dirs = []
#for directory in protein_dirs:
#    all_dirs.append(directory)
all_dirs.append("gsk/")


def commandRunner(command):
    call(command, shell=True)


def runCommands(directory):
    path = datapath + directory + "/"
    pdb_files = [f for f in os.listdir(path) if isfile(join(path, f)) and f.endswith('.pdb')]
    for filename in pdb_files:
        pdb_filepath = path + filename
        json_filepath = path + filename.strip('.pdb') + '.json'
        command = cpp_executable + ' -i ' + pdb_filepath + ' -j ' + json_filepath
        commandRunner(command)


if get_json_files:
    Parallel(n_jobs=parallel_jobs)(delayed(runCommands)(directory) for directory in all_dirs)

if get_pkl_flies:
    # Generate pickle files for each pdb file - naming convention is <protein name><pdb name>.pkl
    max_neighbors = 50

    if not os.path.exists(savepath):
        os.makedirs(savepath)

# Create a one hot encoded feature map for each protein atom
feature_map = {}
with open(groups20_filepath, 'r') as f:
    data = f.readlines()
    len_amino = sum(1 for row in data)
    for idx, line in enumerate(data):
        a = [0] * len_amino
        a[idx] = 1
        name, _ = line.split(" ")
        feature_map[name] = a

with open(protein_atom_init_file, 'w') as f:
    json.dump(feature_map, f)


def createSortedNeighbors(contacts, bonds, max_neighbors):
    bond_true = 1
    bond_false = 0
    neighbor_map = ddict(list)
    dtype = [('index2', int), ('distance', float), ('x1', float), ('y1', float), ('z1', float), ('bool_bond', int)]
    idx = 0

    for contact in contacts:
        if ([contact[0], contact[1]] or [contact[1], contact[0]]) in bonds:
            neighbor_map[contact[0]].append((contact[1], contact[2], contact[3], contact[4], contact[5], bond_true))
            neighbor_map[contact[1]].append((contact[0], contact[2], contact[6], contact[7], contact[8], bond_true))
        else:
            neighbor_map[contact[0]].append((contact[1], contact[2], contact[3], contact[4], contact[5], bond_false))
            neighbor_map[contact[1]].append((contact[0], contact[2], contact[6], contact[7], contact[8], bond_false))
        idx += 1

    for k, v in neighbor_map.items():
        if len(v) < max_neighbors:
            true_nbrs = np.sort(np.array(v, dtype=dtype), order='distance', kind='mergesort').tolist()[0:len(v)]
            true_nbrs.extend([(0, 0, 0, 0, 0, 0) for _ in range(max_neighbors - len(v))])
            neighbor_map[k] = true_nbrs
        else:
            neighbor_map[k] = np.sort(np.array(v, dtype=dtype), order='distance', kind='mergesort').tolist()[
                              0:max_neighbors]

    return neighbor_map


def processDirectory(directory, max_neighbors, savepath):
    path = datapath + directory + '/'
    all_json_files = [file for file in os.listdir(path) if isfile(join(path, file)) and file.endswith('.json')]

    print('Processing ', directory)
    for filename in all_json_files:
        save_filename = (directory + '_' + filename).replace('.json', '')
        json_filepath = path + filename
        with open(json_filepath, 'r') as file:
            json_data = json.load(file)

        neighbor_map = createSortedNeighbors(json_data['contacts'], json_data['bonds'], max_neighbors)
        amino_atom_idx = json_data['res_idx']
        atom_fea = json_data['atoms']
        nbr_fea_idx = np.array([list(map(lambda x: x[0], neighbor_map[idx])) for idx in range(len(json_data['atoms']))])
        nbr_fea = np.array([list(map(lambda x: x[1:], neighbor_map[idx])) for idx in range(len(json_data['atoms']))])

        with open(savepath + save_filename + '.pkl', 'wb') as file:
            pickle.dump(atom_fea, file)
            pickle.dump(nbr_fea, file)
            pickle.dump(nbr_fea_idx, file)
            pickle.dump(amino_atom_idx, file)
            pickle.dump(save_filename, file)


if get_pkl_flies:
    # create a new file
    with open(protein_id_prop_file, 'w') as f:
        pass
    Parallel(n_jobs=parallel_jobs)(
        delayed(processDirectory)(directory, max_neighbors, savepath) for directory in all_dirs)
