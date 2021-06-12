# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model.GaMED_gat_conv import GATConv
# from torch_geometric.nn import GATConv
import config

diag_len = config.Data_CONFIG['diag_len']
med_len = config.Data_CONFIG['med_len']
heads = config.GAT_CONFIG['heads']
output_dim = config.Data_CONFIG['out_dim']


class GAHeM(nn.Module):
    def __init__(self, num_nodes, GAT_output_size):
        super(GAHeM, self).__init__()

        self.num_nodes = num_nodes
        self.GAT_output_size = GAT_output_size
        self.in_channels = 64

        self.embedding_0 = nn.Parameter(torch.Tensor(self.num_nodes, self.in_channels))
        nn.init.xavier_normal_(self.embedding_0)
        self.GAT_DMF = GAT(self.in_channels, self.GAT_output_size, heads=heads, num_nodes=self.num_nodes)
        self.GAT_MM = GAT(self.in_channels, self.GAT_output_size, heads=heads, num_nodes=self.num_nodes)
        self.GAT_ICD10 = GAT(self.in_channels, self.GAT_output_size, heads=heads, num_nodes=self.num_nodes)
        self.GAT_ATC_tree = GAT(self.in_channels, self.GAT_output_size, heads=heads, num_nodes=self.num_nodes)

        self.GAT_DMS = GAT(self.in_channels, self.GAT_output_size, heads=heads, num_nodes=self.num_nodes)

        self.att_semantic_PD = nn.Linear(self.GAT_output_size * heads, 1)
        self.att_semantic_PM = nn.Linear(self.GAT_output_size * heads, 1)
        self.att_semantic_ND = nn.Linear(self.GAT_output_size * heads, 1)
        self.att_semantic_NM = nn.Linear(self.GAT_output_size * heads, 1)

        self.Dipole = Dipole(day_dim=self.GAT_output_size * heads * 2, output_dim=output_dim, rnn_hiddendim=300)

    def forward(self, x, meta_path_list):
        embedding_list_positive = []
        embedding_list_negitive = []

        embedding_list_positive.append(self.GAT_DMF(self.embedding_0, meta_path_list['disease_drug_indication']).unsqueeze(dim=1))

        embedding_list_positive.append(self.GAT_ICD10(self.embedding_0, meta_path_list['icd_tree']).unsqueeze(dim=1))
        embedding_list_positive.append(self.GAT_ATC_tree(self.embedding_0, meta_path_list['atc_tree']).unsqueeze(dim=1))

        embedding_list_negitive.append(self.GAT_DMS(self.embedding_0, meta_path_list['disease_drug_side']).unsqueeze(dim=1))
        embedding_list_negitive.append(self.GAT_MM(self.embedding_0, meta_path_list['drug_drug_edge']).unsqueeze(dim=1))

        multi_embedding_positive = torch.cat(embedding_list_positive,
                                             dim=1)  # [nodes_num,meta_path_num,heads*out_features]
        multi_embedding_negitive = torch.cat(embedding_list_negitive,
                                             dim=1)  # [nodes_num,meta_path_num,heads*out_features]

        # semantic attention
        beta_PM = self.att_semantic_PM(multi_embedding_positive)
        beta_PM = F.relu(torch.squeeze(beta_PM, dim=2))
        beta_PM = F.softmax(beta_PM, dim=1).unsqueeze(dim=2)

        beta_NM = self.att_semantic_NM(multi_embedding_negitive)
        beta_NM = F.relu(torch.squeeze(beta_NM, dim=2))
        beta_NM = F.softmax(beta_NM, dim=1).unsqueeze(dim=2)

        attn_applied_positive = (beta_PM * multi_embedding_positive).mean(dim=1)
        attn_applied_negitive = (beta_NM * multi_embedding_negitive).mean(dim=1)
        attn_applied = torch.cat([attn_applied_positive, attn_applied_negitive], dim=-1)  # 拼接embedding
        x = torch.matmul(x.view(-1, x.shape[2]), attn_applied[:(diag_len + med_len), :]).view(x.shape[0], x.shape[1],
                                                                                              -1)
        output = self.Dipole(x.transpose(0, 1)).transpose(0, 1)
        return output, attn_applied


class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads, num_nodes):
        super(GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.GATConv = GATConv(in_channels=self.in_channels, out_channels=self.out_channels, heads=self.heads,
                               dropout=0.2)
        self.GATConv2 = GATConv(in_channels=self.out_channels * self.heads, out_channels=self.out_channels,
                                heads=self.heads,
                                dropout=0.2)
        self.num_nodes = num_nodes

    def forward(self, x, edge_index, soft_edge_index=None, edge_weight=None, soft_edge_weight=None):
        if soft_edge_index is None:
            graph_embedding = F.relu(self.GATConv(x, edge_index))
            graph_embedding = F.relu(self.GATConv2(graph_embedding, edge_index))
        else:
            graph_embedding = F.relu(self.GATConv(x, edge_index, soft_edge_index, edge_weight, soft_edge_weight))
            graph_embedding = F.relu(self.GATConv2(graph_embedding, edge_index, soft_edge_index, edge_weight, soft_edge_weight))
        return graph_embedding


class Dipole(nn.Module):
    def __init__(self, day_dim, output_dim, rnn_hiddendim, keep_prob=1.0):
        super(Dipole, self).__init__()
        self.day_dim = day_dim
        self.output_dim = output_dim
        self.rnn_hiddendim = rnn_hiddendim
        self.keep_prob = keep_prob

        self.dropout = nn.Dropout(1 - self.keep_prob)

        self.gru = nn.GRU(self.day_dim, self.rnn_hiddendim)
        self.gru_reverse = nn.GRU(self.day_dim, self.rnn_hiddendim)
        self.attn = nn.Linear(self.rnn_hiddendim * 2, 1)
        self.attn_out = nn.Linear(self.rnn_hiddendim * 4, self.day_dim)

        self.out = nn.Linear(self.day_dim, self.output_dim)

    def attentionStep(self, h_0, att_timesteps):
        day_emb = self.day_emb[:att_timesteps]
        rnn_h = self.gru(day_emb, h_0)[0]
        day_emb_reverse = self.day_emb[:att_timesteps].flip(dims=[0])
        rnn_h_reverse = self.gru_reverse(day_emb_reverse, h_0)[0].flip(dims=[0])

        rnn_h = torch.cat((rnn_h, rnn_h_reverse), 2)

        Alpha = self.attn(rnn_h)
        Alpha = torch.squeeze(Alpha, dim=2)
        Alpha = torch.transpose(F.softmax(torch.transpose(Alpha, 0, 1)), 0, 1)

        attn_applied = Alpha.unsqueeze(2) * rnn_h
        c_t = torch.mean(attn_applied, 0)
        h_t = torch.cat((c_t, rnn_h[-1]), dim=1)

        h_t_out = self.attn_out(h_t)
        return h_t_out

    def forward(self, x):
        # x = torch.tensor(x)
        h_0 = self.initHidden(x.shape[1])
        self.day_emb = x

        # LSTM layer
        if self.keep_prob < 1.0:
            self.day_emb = self.dropout(self.day_emb)

        count = np.arange(x.shape[0]) + 1
        h_t_out = torch.zeros_like(self.day_emb)
        for i, att_timesteps in enumerate(count):
            h_t_out[i] = self.attentionStep(h_0, att_timesteps)

        # output layer
        y_hat = self.out(h_t_out)
        y_hat = torch.sigmoid(y_hat)
        return y_hat

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.rnn_hiddendim).cuda()
