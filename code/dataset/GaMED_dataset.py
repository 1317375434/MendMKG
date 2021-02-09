# -*- coding: utf-8 -*-

from torch.utils.data import Dataset


class KGCDataset(Dataset):
    def __init__(self, edge_index):
        super(KGCDataset, self).__init__()
        self.edge_index = edge_index

    def __len__(self):
        return len(self.edge_index[0])

    def __getitem__(self, idx):
        value = self.edge_index[:, idx]
        return value
