# -*- coding: utf-8 -*-

from dataset.dataset import mimicDataset
from model import *
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
import config
from sklearn import metrics

import warnings

warnings.filterwarnings("ignore")

torch.cuda.set_device(0)
SEED = 2020
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    data_length = [len(sq) for sq in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    return data, data_length


def out_label(data, data_len):
    result = []
    data = data.cpu().detach().numpy()

    for i, l in enumerate(data_len):
        result.extend(data[i, :l - 1].reshape(-1, data.shape[-1]).tolist())
    return result


def out_mask(data, data_len, device='cuda'):
    """
    mask
    :param data: tensor
    :param data_len:
    :param device: gpu
    :return: mask matrix
    """
    mask = torch.zeros_like(data, device=device)
    if mask.size(0) == len(data_len):
        for i, l in enumerate(data_len):
            mask[i, :l - 1] = 1
    else:
        for i, l in enumerate(data_len):
            mask[:l - 1, i] = 1
    return mask


def train(model, data, test_data, level_parents, edge_index=None, batch_size=8, n_epoch=10, diag_len=None,
          med_len=None, device='cuda'):
    train_dataset = mimicDataset(data=data, level_parents=level_parents)
    test_dataset = mimicDataset(data=test_data, level_parents=level_parents)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=12)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(n_epoch):
        # train
        model.train()
        t0 = time.time()
        tr_loss, nb_tr_steps = 0, 0
        for idx, (batch_data, batch_data_len) in enumerate(data_loader):
            x = batch_data[:, :, :(diag_len + med_len)].to(device)
            label = batch_data[:, :, (diag_len + med_len):].to(device)

            mask = out_mask(label, batch_data_len)
            if edge_index is not None:
                out, _ = model.forward(x, edge_index)
            else:
                out = model.forward(x)

            optimizer.zero_grad()
            loss = F.binary_cross_entropy(out * mask, label)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_steps += 1
        print('epoch{},train_loss:{},train_time:{}'.format(epoch, tr_loss / nb_tr_steps, time.time() - t0))

        # test
        model.eval()
        with torch.no_grad():
            tr_loss, nb_tr_steps = 0, 0
            test_label, res = [], []
            for idx, (batch_data, batch_data_len) in enumerate(test_loader):
                x = batch_data[:, :, :(diag_len + med_len)].to(device)
                label = batch_data[:, :, (diag_len + med_len):].to(device)

                mask = out_mask(label, batch_data_len)
                if edge_index is not None:
                    out, embedding = model.forward(x, edge_index)
                else:
                    out = model.forward(x)
                loss = F.binary_cross_entropy(out * mask, label)

                tr_loss += loss.item()
                nb_tr_steps += 1
                test_label.extend(out_label(label, batch_data_len))
                res.extend(out_label(out, batch_data_len))
            precision, recall, _thresholds = metrics.precision_recall_curve(np.array(test_label).ravel(),
                                                                            np.array(res).ravel())
            prauc = metrics.auc(recall, precision)

            print('prauc:', prauc)


if __name__ == '__main__':

    data = pd.read_csv('../data/train_data.csv')
    test_data = pd.read_csv('../data/test_data.csv')

    edge_index = np.load('../saved/graph_edge_coo_completion.npz')
    edge_index_dict = {}
    for id in edge_index.keys():
        edge_index_dict[id] = torch.tensor(edge_index[id], dtype=torch.long, device=config.Dipole_CONFIG['device'])

    diag_len = config.Data_CONFIG['diag_len']
    med_len = config.Data_CONFIG['med_len']
    out_dim = config.Data_CONFIG['out_dim']
    num_modes = config.Data_CONFIG['num_nodes']
    n_epoch = config.Dipole_CONFIG['n_epoch']

    model = GAHeM(num_modes, GAT_output_size=64)

    print(model)

    model.load_state_dict(torch.load('../saved/pretrain_model.tar'),
                          strict=False)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    level_parents = pd.read_csv('../data/data_label.csv')
    print(level_parents.shape)
    train(model, data, test_data, level_parents, edge_index_dict, diag_len=diag_len, med_len=med_len)  # GAT*
