# -*- coding: utf-8 -*-

import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import config
from model.GaMED_AE_KGC import GraphAutoEncoder, GraphCompletion
from dataset.dataset import PreTrainTransformerDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import warnings

warnings.filterwarnings("ignore")

torch.cuda.set_device(0)
SEED = 666
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

diag_len = config.Data_CONFIG['diag_len']
med_len = config.Data_CONFIG['med_len']
heads = config.GAT_CONFIG['heads']
output_dim = config.Data_CONFIG['out_dim']


def train(model_GCM, data, test_data, level_parents, edge_index=None, batch_size=64, n_epoch=50, device='cuda', **kwargs):
    state_dict_GAE = None
    edge_index_dict = {}
    for id in edge_index.keys():
        edge_index_dict[id] = torch.tensor(edge_index[id], dtype=torch.long, device=config.Dipole_CONFIG['device'])

    soft_IND = torch.tensor(kwargs.get('soft_IND'), dtype=torch.long, device=config.Dipole_CONFIG['device'])
    soft_ADR = torch.tensor(kwargs.get('soft_ADR'), dtype=torch.long, device=config.Dipole_CONFIG['device'])
    soft_DDI = torch.tensor(kwargs.get('soft_DDI'), dtype=torch.long, device=config.Dipole_CONFIG['device'])
    for i in range(1):
        optimizer = optim.Adam(model_GCM.parameters(), lr=0.001)
        edge_index_pos_IND = edge_index_dict['disease_drug_indication']
        edge_index_pos_ADR = edge_index_dict['disease_drug_side']
        edge_index_pos_DDI = edge_index_dict['drug_drug_edge']
        best_loss = 10
        AE_loss_best = 10
        for epoch in range(50):
            # train
            model_GCM.train()
            t0 = time.time()
            out_embedding_1, out_embedding_2, out_embedding_3 = model_GCM.forward(edge_index_dict)
            optimizer.zero_grad()
            loss = model_GCM.recon_loss(
                out_embedding_1, edge_index_pos_IND) + model_GCM.recon_loss(
                out_embedding_2, edge_index_pos_ADR) + model_GCM.recon_loss(
                out_embedding_3, edge_index_pos_DDI)
            loss.backward()
            optimizer.step()
            print('epoch{},train_loss:{},train_time:{}'.format(epoch, loss.item(), time.time() - t0))
            if loss.item() < best_loss:
                state_dict = model_GCM.state_dict()
                best_loss = loss.item()
        with torch.no_grad():
            _, indices_1 = model_GCM.decoder(out_embedding_1, soft_IND).topk(1000, largest=True, sorted=True)
            _, indices_2 = model_GCM.decoder(out_embedding_2, soft_ADR).topk(2000, largest=True, sorted=True)
            _, indices_3 = model_GCM.decoder(out_embedding_3, soft_DDI).topk(4000, largest=True, sorted=True)

            soft_edge_list = {'disease_drug_indication': soft_IND[:, indices_1],
                              'disease_drug_side': soft_ADR[:, indices_2],
                              'drug_drug_edge': soft_DDI[:, indices_3]}
        GAE = GraphAutoEncoder(num_modes, GAT_output_size=64, edge_list=edge_index_dict, soft_edge_list=soft_edge_list)

        if torch.cuda.is_available():
            GAE = GAE.cuda()
        if state_dict_GAE is not None:
            GAE.load_state_dict(state_dict_GAE, strict=False)
        GAE.load_state_dict(state_dict, strict=False)
        for p in GAE.parameters():
            p.requires_grad = False
        for p in GAE.outputLayer.parameters():
            p.requires_grad = True

        GAE.soft_edge_weight_DMF.requires_grad = True
        GAE.soft_edge_weight_DMS.requires_grad = True
        GAE.soft_edge_weight_MM.requires_grad = True

        train_dataset = PreTrainTransformerDataset(data=data, level_parents=level_parents)
        test_dataset = PreTrainTransformerDataset(data=test_data, level_parents=level_parents)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=12)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, GAE.parameters()), lr=0.0001)
        for epoch in range(10):
            # train
            GAE.train()
            t0 = time.time()
            tr_loss, nb_tr_steps = 0, 0
            for idx, (batch_data, batch_label) in enumerate(train_data_loader):
                x = batch_data[:, :, :(diag_len + med_len)].to(device)
                label = batch_label.to(device)

                out = GAE.forward(x, edge_index_dict, soft_edge_list=soft_edge_list)

                optimizer.zero_grad()
                loss = F.binary_cross_entropy(out, label)
                loss.backward()
                optimizer.step()

                tr_loss += loss.item()
                nb_tr_steps += 1
            print('epoch{},train_loss:{},train_time:{}'.format(epoch, tr_loss / nb_tr_steps, time.time() - t0))

            # test
            GAE.eval()
            with torch.no_grad():
                t0 = time.time()
                tr_loss, nb_tr_steps = 0, 0
                for idx, (batch_data, batch_label) in enumerate(test_data_loader):
                    x = batch_data[:, :, :(diag_len + med_len)].to(device)
                    label = batch_label.to(device)

                    out = GAE.forward(x, edge_index_dict, soft_edge_list=soft_edge_list)

                    loss = F.binary_cross_entropy(out, label)
                    tr_loss += loss.item()
                    nb_tr_steps += 1
                print('epoch{},test_loss:{},train_time:{}'.format(epoch, tr_loss / nb_tr_steps, time.time() - t0))
                if (tr_loss / nb_tr_steps) < AE_loss_best:
                    AE_loss_best = tr_loss / nb_tr_steps
                    state_dict_GAE = {k: v for k, v in GAE.state_dict().items() if k.split('.')[0] == 'outputLayer'}

        _, indices_add_edge_DMF = GAE.soft_edge_weight_DMF.topk(200, largest=True, sorted=True)
        _, indices_add_edge_DMS = GAE.soft_edge_weight_DMS.topk(400, largest=True, sorted=True)
        _, indices_add_edge_DDI = GAE.soft_edge_weight_MM.topk(800, largest=True, sorted=True)
        edge_index_dict['disease_drug_indication'] = torch.cat([edge_index_dict['disease_drug_indication'],
                                                                soft_edge_list['disease_drug_indication'][:, indices_add_edge_DMF]], dim=-1)
        edge_index_dict['disease_drug_side'] = torch.cat([edge_index_dict['disease_drug_side'],
                                                          soft_edge_list['disease_drug_side'][:, indices_add_edge_DMS]], dim=-1)
        edge_index_dict['drug_drug_edge'] = torch.cat([edge_index_dict['drug_drug_edge'],
                                                       soft_edge_list['drug_drug_edge'][:, indices_add_edge_DDI]], dim=-1)
        print()
    np.savez('../saved/graph_edge_coo_completion.npz',
             icd_tree=edge_index_dict['icd_tree'].cpu().detach().numpy(),
             atc_tree=edge_index_dict['atc_tree'].cpu().detach().numpy(),
             disease_drug_indication=edge_index_dict['disease_drug_indication'].cpu().detach().numpy(),
             disease_drug_side=edge_index_dict['disease_drug_side'].cpu().detach().numpy(),
             drug_drug_edge=edge_index_dict['drug_drug_edge'].cpu().detach().numpy())
    torch.save(state_dict, '../saved/pretrain_model.tar')


def candidate_edge_index(edge_index):
    diag_index = [i for i in range(0, 3071)]
    med_index = [i for i in range(3071, 3353)]
    IND = [tuple(edge) for edge in edge_index['disease_drug_indication'].T.tolist()]
    IDR = [tuple(edge) for edge in edge_index['disease_drug_side'].T.tolist()]
    DDI = [tuple(edge) for edge in edge_index['drug_drug_edge'].T.tolist()]
    IND_IDR_Completion_index = set()
    for i in diag_index:
        for j in med_index:
            if i != j:
                IND_IDR_Completion_index.add((i, j))
    DDI_Completion_index = set()
    for i in med_index:
        for j in med_index:
            if i != j:
                DDI_Completion_index.add((i, j))
    IND_candidate = IND_IDR_Completion_index - set(IND)
    IDR_candidate = IND_IDR_Completion_index - set(IDR)
    DDI_candidate = DDI_Completion_index - set(DDI)
    return IND_candidate, IDR_candidate, DDI_candidate


if __name__ == '__main__':
    num_modes = config.Data_CONFIG['num_nodes']
    edge_index_dict = np.load('../data/graph_edge_coo.npz')

    IND_candidate, IDR_candidate, DDI_candidate = candidate_edge_index(edge_index_dict)
    GCM = GraphCompletion(num_modes, GAT_output_size=64)

    print(GCM)
    total_params = sum(p.numel() for p in GCM.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        GCM = GCM.cuda()
    data = pd.read_csv('../data/train_data.csv')
    test_data = pd.read_csv('../data/test_data.csv')
    level_parents = pd.read_csv('../data/data_label.csv')
    train(GCM, data, test_data, level_parents, edge_index=edge_index_dict, soft_DDI=DDI_candidate, soft_IND=IND_candidate,
          soft_ADR=IDR_candidate)
