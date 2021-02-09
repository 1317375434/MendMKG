# -*- coding: utf-8 -*-

import torch
from torch import nn
from model.gat_conv import GATConv
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
import config

diag_len = config.Data_CONFIG['diag_len']
med_len = config.Data_CONFIG['med_len']
heads = config.GAT_CONFIG['heads']
output_dim = config.Data_CONFIG['out_dim']
EPS = 1e-15
MAX_LOGSTD = 10


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


class GraphCompletion(nn.Module):
    def __init__(self, num_nodes, GAT_output_size):
        super(GraphCompletion, self).__init__()
        self.num_nodes = num_nodes
        self.GAT_output_size = GAT_output_size
        self.in_channels = 64
        self.MLP_hidden_size = 128

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


        self.decoder_linear_1 = nn.Linear(self.GAT_output_size * heads * 2, self.GAT_output_size * heads * 2)
        self.decoder_linear_2 = nn.Linear(self.GAT_output_size * heads * 2, self.GAT_output_size * heads * 2)
        self.decoder_linear_3 = nn.Linear(self.GAT_output_size * heads * 2, self.GAT_output_size * heads * 2)

    def decoder(self, z, edge_index):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value)

    def forward(self, meta_path_list):
        """
        :param x: tensor [batch_size,seq_len,n_features] input embedding
        :param meta_path_list: dict {meta_path:tensor([2,n_edge])}
        :return:
        """
        embedding_list_positive = []
        embedding_list_negitive = []

        embedding_list_positive.append(self.GAT_DMF(self.embedding_0, meta_path_list['disease_drug_indication']).unsqueeze(dim=1))
        embedding_list_positive.append(self.GAT_ICD10(self.embedding_0, meta_path_list['icd_tree']).unsqueeze(dim=1))
        embedding_list_positive.append(self.GAT_ATC_tree(self.embedding_0, meta_path_list['atc_tree']).unsqueeze(dim=1))

        embedding_list_negitive.append(
            self.GAT_DMS(self.embedding_0, meta_path_list['disease_drug_side']).unsqueeze(dim=1))
        embedding_list_negitive.append(
            self.GAT_MM(self.embedding_0, meta_path_list['drug_drug_edge']).unsqueeze(dim=1))

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
        attn_applied = torch.cat([attn_applied_positive, attn_applied_negitive], dim=-1)

        return self.decoder_linear_1(attn_applied), self.decoder_linear_2(attn_applied), self.decoder_linear_3(attn_applied)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        pos_loss = -torch.log(self.decoder(z, pos_edge_index) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index) + EPS).mean()

        return pos_loss + neg_loss


class GraphAutoEncoder(nn.Module):
    def __init__(self, num_nodes, GAT_output_size, edge_list, soft_edge_list):
        super(GraphAutoEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.GAT_output_size = GAT_output_size
        self.in_channels = 64
        self.MLP_hidden_size = 128
        self.n_class = output_dim

        self.embedding_0 = nn.Parameter(torch.Tensor(self.num_nodes, self.in_channels))
        nn.init.xavier_normal_(self.embedding_0)

        self.edge_weight_DMF = nn.Parameter(torch.ones(edge_list['disease_drug_indication'].shape[1] + self.num_nodes))
        self.soft_edge_weight_DMF = nn.Parameter(torch.ones(soft_edge_list['disease_drug_indication'].shape[1]))
        self.edge_weight_DMS = nn.Parameter(torch.ones(edge_list['disease_drug_side'].shape[1] + self.num_nodes))
        self.soft_edge_weight_DMS = nn.Parameter(torch.ones(soft_edge_list['disease_drug_side'].shape[1]))
        self.edge_weight_MM = nn.Parameter(torch.ones(edge_list['drug_drug_edge'].shape[1] + self.num_nodes))
        self.soft_edge_weight_MM = nn.Parameter(torch.rand(soft_edge_list['drug_drug_edge'].shape[1]))

        self.GAT_DMF = GAT(self.in_channels, self.GAT_output_size, heads=heads, num_nodes=self.num_nodes)
        self.GAT_MM = GAT(self.in_channels, self.GAT_output_size, heads=heads, num_nodes=self.num_nodes)
        self.GAT_ICD10 = GAT(self.in_channels, self.GAT_output_size, heads=heads, num_nodes=self.num_nodes)
        self.GAT_ATC_tree = GAT(self.in_channels, self.GAT_output_size, heads=heads, num_nodes=self.num_nodes)

        self.GAT_DMS = GAT(self.in_channels, self.GAT_output_size, heads=heads, num_nodes=self.num_nodes)

        self.att_semantic_PD = nn.Linear(self.GAT_output_size * heads, 1)
        self.att_semantic_PM = nn.Linear(self.GAT_output_size * heads, 1)
        self.att_semantic_ND = nn.Linear(self.GAT_output_size * heads, 1)
        self.att_semantic_NM = nn.Linear(self.GAT_output_size * heads, 1)

        self.outputLayer = nn.Sequential(nn.Linear(self.GAT_output_size * heads * 2, self.MLP_hidden_size),
                                         nn.ReLU(), nn.Linear(self.MLP_hidden_size, self.n_class),
                                         nn.Sigmoid())

    def forward(self, x, meta_path_list, soft_edge_list):
        """
        :param x: tensor [batch_size,seq_len,n_features]
        :param meta_path_list: dict {meta_path:tensor([2,n_edge])}
        :return:
        """
        embedding_list_positive = []
        embedding_list_negitive = []

        embedding_list_positive.append(self.GAT_DMF(self.embedding_0, meta_path_list['disease_drug_indication'],
                                                    soft_edge_list['disease_drug_indication'], self.edge_weight_DMF,
                                                    self.soft_edge_weight_DMF).unsqueeze(dim=1))

        embedding_list_positive.append(self.GAT_ICD10(self.embedding_0, meta_path_list['icd_tree']).unsqueeze(dim=1))
        embedding_list_positive.append(self.GAT_ATC_tree(self.embedding_0, meta_path_list['atc_tree']).unsqueeze(dim=1))

        embedding_list_negitive.append(
            self.GAT_DMS(self.embedding_0, meta_path_list['disease_drug_side'], soft_edge_list['disease_drug_side'], self.edge_weight_DMS,
                         self.soft_edge_weight_DMS).unsqueeze(dim=1))
        embedding_list_negitive.append(
            self.GAT_MM(self.embedding_0, meta_path_list['drug_drug_edge'], soft_edge_list['drug_drug_edge'], self.edge_weight_MM,
                        self.soft_edge_weight_MM).unsqueeze(dim=1))

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
        attn_applied = torch.cat([attn_applied_positive, attn_applied_negitive], dim=-1)

        x = torch.matmul(x.view(-1, x.shape[2]), attn_applied[:(diag_len + med_len), :]).view(x.shape[0], x.shape[1],
                                                                                              -1)
        output = self.outputLayer(x)
        return output
