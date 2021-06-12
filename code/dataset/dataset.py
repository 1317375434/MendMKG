# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset

import numpy as np
import config


class mimicDataset(Dataset):
    def __init__(self, data, level_parents):
        super(mimicDataset, self).__init__()
        self.data = data
        self.level_parents = level_parents
        self.SUBJECT_ID_list = self.data.SUBJECT_ID.unique()
        self.diag_len = config.Data_CONFIG['diag_len']
        self.med_len = config.Data_CONFIG['med_len']

    def __len__(self):
        return len(self.SUBJECT_ID_list)

    def __getitem__(self, idx):
        value = self.data[self.data.SUBJECT_ID.isin([self.SUBJECT_ID_list[idx]])].drop(['SUBJECT_ID', 'HADM_ID'],
                                                                                       axis=1)
        label = self.level_parents[self.level_parents.SUBJECT_ID.isin([self.SUBJECT_ID_list[idx]])].drop(
            ['SUBJECT_ID', 'HADM_ID'], axis=1).shift(-1).fillna(0)

        value = torch.tensor(np.array(value)[:, :(self.diag_len + self.med_len)], dtype=torch.float)
        label = torch.tensor(np.array(label), dtype=torch.float)

        return torch.cat([value, label], dim=1)


class PreTrainTransformerDataset(Dataset):
    def __init__(self, data, level_parents):
        super(PreTrainTransformerDataset, self).__init__()
        self.data = data
        self.level_parents = level_parents
        self.diag_len = config.Data_CONFIG['diag_len']
        self.med_len = config.Data_CONFIG['med_len']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        value = self.data[idx:(idx + 1)]
        subject_id = value['SUBJECT_ID'].values[0]
        hadm_id = value['HADM_ID'].values[0]
        value = value.drop(['SUBJECT_ID', 'HADM_ID'], axis=1)
        label = self.level_parents[(self.level_parents['SUBJECT_ID'] == subject_id) &
                                   (self.level_parents['HADM_ID'] == hadm_id)].drop(['SUBJECT_ID', 'HADM_ID'], axis=1)

        value = torch.tensor(np.array(value)[:, :(self.diag_len + self.med_len)], dtype=torch.float)
        label = torch.tensor(np.array(label), dtype=torch.float)

        return value, label
