import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math
import os 
import numpy as np
import dgl
import networkx as nx
import sys
from dgl import ops 

from torch_geometric.nn import GCNConv, GATConv, APPNP, GCN2Conv, MessagePassing, JumpingKnowledge
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul
import scipy.sparse
from torch_geometric.utils import degree, add_self_loops, remove_self_loops, from_scipy_sparse_matrix, to_undirected, get_laplacian
# from gcnconv import GCNConv
# from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import scipy.sparse as sp



class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLP, self).__init__()
        
        self.lins = nn.ModuleList()
        self.lns = nn.ModuleList()
        
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, hidden_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.lns.append(nn.LayerNorm(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.gelu(x)
            x = self.lns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    

# GCN
class GCN(nn.Module):

    def __init__(self, dataset, num_layers, mlp_layers, input_dim, hidden_dim, dropout, device, latent_dim, with_latent):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device
        self.gcn_convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.mlp_layers = mlp_layers
        
        if with_latent == 'True':
            self.init_layer = nn.Linear(self.dataset.num_features + latent_dim, self.hidden_dim)
        else:
            self.init_layer = nn.Linear(self.dataset.num_features, self.hidden_dim)
        
        self.feed_forward = MLP(self.hidden_dim, self.hidden_dim, dataset.num_classes, self.mlp_layers, dropout=self.dropout).to(self.device) 
                
        for i in range(self.num_layers):   
            conv = GCNConv(hidden_dim, hidden_dim)
            self.gcn_convs.append(conv.to(self.device))
            self.lns.append(nn.LayerNorm(hidden_dim))


    def forward(self, x, edge_index):
        x = x.to(self.device)
        num_nodes = x.shape[0]
        # num_edges = edge_index.shape[1]
        
        x = self.init_layer(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = F.gelu(x) 
        
        # message propagation through hidden layers
        for i in range(self.num_layers):
         
            x_res = self.lns[i](x)
            x = self.gcn_convs[i](x, edge_index)
            x = x + x_res
            
        x = self.feed_forward(x)
                     
        embedding = x
        x = F.log_softmax(x, dim = 1)
        return x



class GAT(nn.Module):
    def __init__(self, dataset, in_channels, hidden_channels, out_channels, latent_dim, with_latent, num_layers=2,
                 dropout=0.5, heads=2, sampling=False, add_self_loops=True):
        super(GAT, self).__init__()

        self.dropout = dropout
        self.feed_forward = MLP(hidden_channels, hidden_channels, dataset.num_classes, num_layers = 1, dropout=self.dropout)

        # if with_latent == 'True':
        #     self.init_layer = nn.Linear(dataset.num_features + latent_dim, hidden_dim)
        # else:
        #     self.init_layer = nn.Linear(dataset.num_features, hidden_dim)

        self.convs = nn.ModuleList()
        
        # for i in range(num_layers - 1):
        #     if i == 0:
        #         self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=True, add_self_loops=add_self_loops))
        #     else:
        #         self.convs.append(GATConv(hidden_dim*heads, hidden_dim, heads=heads, concat=True, add_self_loops=add_self_loops))
        #     self.bns.append(nn.BatchNorm1d(hidden_dim*heads))

        # self.convs.append(GATConv(hidden_dim*heads, hidden_dim, heads=heads, concat=False, add_self_loops=add_self_loops))
        
        if with_latent == 'True':
            self.convs.append(GATConv(in_channels+latent_dim, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops))
        else:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(in_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(GATConv(hidden_channels*heads, out_channels, heads=heads, concat=False, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.elu
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):

        # x = self.init_layer(x)
        # x = F.dropout(x, p=self.dropout, training=True)
        # x = F.gelu(x) 

        for i, conv in enumerate(self.convs[:-1]):
            # x_res = self.bns[i](x)
            x = conv(x, edge_index)
            # print(i, "   ", x.shape)
            # x = x + x_res
            # x = self.activation(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        # x = self.feed_forward(x)
        return x



class APPNP_Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, latent_dim, with_latent, dprate=.0, dropout=.5, K=10, alpha=.1, num_layers=3):
        super(APPNP_Net, self).__init__()

        if with_latent == 'True':
            self.mlp = MLP(in_channels+latent_dim, hidden_channels, out_channels,
                       num_layers=num_layers, dropout=dropout)
        else:
            self.mlp = MLP(in_channels, hidden_channels, out_channels,
                       num_layers=num_layers, dropout=dropout)
        self.prop1 = APPNP(K, alpha)

        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):
        # edge_index = data.graph['edge_index']
        x = self.mlp(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x
        
        
class GCNII(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, latent_dim, with_latent, num_layers, alpha, theta, shared_weights=True, dropout=0.5):
        super(GCNII, self).__init__()

        self.lins = nn.ModuleList()
        if with_latent == 'True':
            self.lins.append(nn.Linear(in_channels+latent_dim, hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                        shared_weights, normalize=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        # x = data.graph['node_feat']
        n = x.shape[0]
        # edge_index = data.graph['edge_index']
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(
                row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype)
            edge_weight = None
            adj_t = edge_index

        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = self.bns[i](x)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x
    
    
    

class GPR_prop(MessagePassing):
    '''
    GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, latent_dim, with_latent, Init='Random', dprate=.0, dropout=.5, K=10, alpha=.1, Gamma=None, num_layers=3):
        super(GPRGNN, self).__init__()

        if with_latent == 'True':
            self.mlp = MLP(in_channels+latent_dim, hidden_channels, out_channels,
                        num_layers=num_layers, dropout=dropout)
        else:
            self.mlp = MLP(in_channels, hidden_channels, out_channels,
                        num_layers=num_layers, dropout=dropout)
        self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):
        # edge_index = data.graph['edge_index']
        x = self.mlp(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x



class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """

    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        device = x.device
        # print("type ", type(x), "  ", type(adj_t), "   ", type(adj_t2))
        # print("shape ", x.shape, "  ", adj_t, "  ", adj_t2)
        x1 = torch.sparse.mm(adj_t.to_dense().to(device), x)
        x2 = torch.sparse.mm(adj_t2.to_dense().to(device), x)
        return torch.cat([x1, x2], dim=1)


class H2GCN(nn.Module):
    """ our implementation """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes, latent_dim, with_latent,
                 num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                 use_bn=True, conv_dropout=True):
        super(H2GCN, self).__init__()

        if with_latent == 'True':
            self.feature_embed = MLP(in_channels + latent_dim, hidden_channels,
                                    hidden_channels, num_layers=num_mlp_layers, dropout=dropout)
        else:
            self.feature_embed = MLP(in_channels, hidden_channels,
                                    hidden_channels, num_layers=num_mlp_layers, dropout=dropout)

        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs)))

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers-2:
                self.bns.append(nn.BatchNorm1d(
                    hidden_channels*2*len(self.convs)))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout  # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels*(2**(num_layers+1)-1)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes

        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(
                row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)

        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)


    def forward(self, x, edge_index):
        # x = data.graph['node_feat']
        # n = data.graph['num_nodes']

        adj_t = self.adj_t
        adj_t2 = self.adj_t2

        x = self.feature_embed(x)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2)
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_project(x)
        return x
