import os
import numpy as np

rg_dict = {'RS1': [], 'PaaA2': [], 'synuclein': []}
for files in os.listdir('./'):
    fsplit = files.split('_')
    if fsplit[0] == 'Rg':
        with open(files, 'r') as f:
            for lines in f.readlines()[1:]:
                rg_dict[fsplit[1]].append(float(lines.split()[-1]))

_mean = [np.mean(rg_dict['RS1']),
         np.mean(rg_dict['PaaA2']),
         np.mean(rg_dict['synuclein'])]
_std = [np.std(rg_dict['RS1']),
        np.std(rg_dict['PaaA2']),
        np.std(rg_dict['synuclein'])]

print(_mean, _std)