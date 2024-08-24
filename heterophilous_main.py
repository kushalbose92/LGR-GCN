import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
import os 
import time

from torch_geometric.loader import GraphSAINTRandomWalkSampler, RandomNodeLoader
from torch_geometric.utils import degree, remove_self_loops, add_self_loops, to_dense_adj, homophily, degree
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_sparse import SparseTensor

from models import MLP
# from expt_models import GAT, APPNP_Net, GGCN, GCNII, GPRGNN, H2GCN, LINKX
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from torch_geometric.utils import homophily
from train import generate_latent_feature, train, mlp_training
from test import test


import sdrf 
import fosr 
import borf 

import argparse
from tqdm import tqdm 
import matplotlib.pyplot as plt

# from custom_parser import argument_parser
from model_for_new_heterophily import GCN, GAT, APPNP_Net, GCNII, GPRGNN, H2GCN
from utils import *

from sklearn.metrics import roc_auc_score


# loss function
def loss_fn(pred, label):
    return F.nll_loss(pred, label)

# train function
def train_new(dataset, model, criterion, opti_train, latent_x, new_edge_index, device, with_latent):   
    model.train()
    opti_train.zero_grad()
    # dataset.x = dataset.x.to(device)
    dataset.edge_index = dataset.edge_index.to(device)
    if with_latent == 'True':
        out = model(latent_x, new_edge_index) 
    else:
        out = model(dataset.x, new_edge_index) 
    # emb, pred = model(dataset.x, dataset.edge_index)
    label = dataset.y.to(device)
    pred_train = out[dataset.train_mask].to(device)
    label_train = label[dataset.train_mask]
    loss_train = criterion(pred_train, label_train) 
    loss_train.backward()
    opti_train.step()

    train_acc, val_acc, test_acc = test_new(dataset, model, latent_x, new_edge_index, with_latent)
    
    return loss_train, val_acc, train_acc

@torch.no_grad()
def test_new(dataset, model, latent_x, new_edge_index, with_latent):

    model.eval()
    if with_latent:
        pred = model(latent_x, new_edge_index) 
    else:
        pred = model(dataset.x, new_edge_index) 
    # dataset.x = dataset.x.to(device)
    # dataset.edge_index = dataset.edge_index.to(device)
    dataset.y = dataset.y.to(device)
    # emb, pred = model(dataset.x, dataset.edge_index)
    # print("pred ", pred.shape)

    # print(y_true['train'].shape, "  ", y_true['valid'].shape, "  ", y_true['test'].shape)
    
    if dataname == 'Minesweeper' or dataname == 'Tolokers' or dataname == 'Questions':
        pred = pred[:, 1]
        train_acc = roc_auc_score(y_true = dataset.y[dataset.train_mask].cpu().numpy(), y_score = pred[dataset.train_mask].cpu().numpy()).item()
        valid_acc = roc_auc_score(y_true = dataset.y[dataset.valid_mask].cpu().numpy(), y_score = pred[dataset.valid_mask].cpu().numpy()).item()
        test_acc = roc_auc_score(y_true = dataset.y[dataset.test_mask].cpu().numpy(), y_score = pred[dataset.test_mask].cpu().numpy()).item()
    else:
        pred = pred.argmax(dim=1)
        train_acc = (pred[dataset.train_mask] == dataset.y[dataset.train_mask]).float().mean().item()
        valid_acc = (pred[dataset.valid_mask] == dataset.y[dataset.valid_mask]).float().mean().item()
        test_acc = (pred[dataset.test_mask] == dataset.y[dataset.test_mask]).float().mean().item()
            

    return train_acc, valid_acc, test_acc


parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str)
parser.add_argument('--seed', type = int)

parser.add_argument('--MLP_epoch', type=int)
parser.add_argument('--MLP_hidden', type=int)
parser.add_argument('--MLP_weight_decay', type=float, default=5e-4)
parser.add_argument('--MLP_learning_rate', type=float, default = 0.01)
parser.add_argument('--MLP_dropout', type=float)

parser.add_argument('--AE_epoch', type=int)
parser.add_argument('--AE_latent_dim', type=int)
parser.add_argument('--AE_weight_decay', type=float)
parser.add_argument('--AE_learning_rate', type=float, default = 0.01)

parser.add_argument('--GCN_epoch', type=int)
parser.add_argument('--GCN_hidden', type=int)
parser.add_argument('--GCN_weight_decay', type=float)
parser.add_argument('--GCN_learning_rate', type=float, default = 0.01)
parser.add_argument('--GCN_dropout', type=float)

parser.add_argument('--top_k', type=int)
parser.add_argument('--bottom_k', type=int)

parser.add_argument('--model', type=None)
parser.add_argument('--vanilla', type=None)
parser.add_argument('--with_latent', type=None)
parser.add_argument('--rewiring', type=None)
parser.add_argument('--device', type=str)

args = parser.parse_args()

dataname = args.dataname
seed = args.seed

mlp_epoch = args.MLP_epoch
mlp_hidden = args.MLP_hidden
mlp_weight_decay = args.MLP_weight_decay
mlp_learning_rate = args.MLP_learning_rate
mlp_dropout = args.MLP_dropout

ae_epoch = args.AE_epoch
ae_hidden = args.AE_latent_dim
ae_weight_decay = args.AE_weight_decay
ae_learning_rate = args.AE_learning_rate

gcn_epoch = args.GCN_epoch
gcn_hidden = args.GCN_hidden
gcn_weight_decay = args.GCN_weight_decay
gcn_learning_rate = args.GCN_learning_rate
gcn_dropout = args.GCN_dropout

top_k = args.top_k
bottom_k = args.bottom_k

model_name = args.model
vanilla = args.vanilla
with_latent = args.with_latent
rewiring = args.rewiring
device = args.device


print(args)

# setting seeds
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device == 'cuda:0':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# "Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"

dataset = HeterophilousGraphDataset(root='./data', name=dataname)
print(dataset)

print(dataset.x.sum(dim=1))

print("----------------------------------------------")
print("Dataset: ", dataname)
print("Label-guided Graph Rewiring")
# print("number of hidden layers:", num_layers)
print("-------------------------------------------------------")
print("features ", dataset.x.shape)
print("node labels ", dataset.y.shape)
print("edge index ", dataset.edge_index.shape)
print("mask ", dataset.train_mask.shape, "   ", dataset.val_mask.shape, "   ", dataset.test_mask.shape)
print("----------------------------------------------")


dataset.num_nodes = dataset.x.shape[0]
dataset.num_edges = dataset.edge_index.shape[1]
# dataset.num_classes = max(dataset.y) + 1


node_degrees = degree(dataset.edge_index[0], num_nodes = dataset.num_nodes)
isolated_nodes = torch.sum(torch.eq(node_degrees, 0)).item()
print(f"Isolated nodes: {isolated_nodes} || Total nodes: {dataset.num_nodes}")


# dataset.edge_index = remove_self_loops(dataset.edge_index)[0]
# dataset.edge_index = add_self_loops(dataset.edge_index)[0]
# print("edge index ", dataset.edge_index.shape)

homo_ratio = homophily(dataset.edge_index, dataset.y, method = 'edge')
print("Homophily ratio ", homo_ratio)

# import sys
# sys.exit()

# dataset.edge_index = add_self_loops(dataset.edge_index)[0]
# print(dataset.edge_index.shape)
# degrees = degree(dataset.edge_index[0], num_nodes=dataset.x.shape[0])
# print(sum(degrees))

# row_transform = NormalizeFeatures()
# data = Data(x = dataset.x, edge_index = dataset.edge_index)
# data = row_transform(data)
# dataset.x = data.x
# print(dataset.x.sum(dim=1))
# feat_label_ratio = feature_class_relation(dataset.edge_index, dataset.y, dataset.x)
# print(f"Feature to Label ratio:  {feat_label_ratio.item(): .4f}")

# degree_distribution(dataset.edge_index, dataset.num_nodes, dataname)

# node_degrees = degree(dataset.edge_index[0], num_nodes = dataset.num_nodes)
# avg_degrees = node_degrees.sum() / dataset.num_nodes
# print(f"Avg degree: {avg_degrees}")
# import sys 
# sys.exit()



print("before ", dataset.edge_index.shape)
if rewiring == 'fosr':
    print("Applying FoSR rewiring")
    edge_index, edge_type, _ = fosr.edge_rewire(dataset.edge_index.numpy(), num_iterations=50)
    dataset.edge_index = torch.tensor(edge_index)
elif rewiring == 'sdrf':
    print("Applying SDRF rewiring")
    dataset.edge_index = sdrf.sdrf(dataset, loops=50, remove_edges=False, is_undirected=True)
elif rewiring == 'borf':
    print("Applying BORF rewiring")
    dataset.edge_index, _ = borf.borf3(dataset, 
              loops=10, 
              remove_edges=False, 
              is_undirected=True,
              batch_add=4,
              batch_remove=2,
              dataset_name=None,
              graph_index=g)
else:
    print("Invalid rewiirng method...")
print("after ", dataset.edge_index.shape)

dataset.x = dataset.x.to(device)
dataset.edge_index = dataset.edge_index.to(device)
dataset.y = dataset.y.to(device)
    
train_mask_set = dataset.train_mask
valid_mask_set = dataset.val_mask
test_mask_set = dataset.test_mask
print(train_mask_set.shape, "  ", valid_mask_set.shape, "  ", test_mask_set.shape)

print("Optimization started....")

test_acc_list = []
max_test = []
for run in range(10):
    print("-----------------For Split " + str(run) + "-------------------------")
    
    dataset.train_mask = train_mask_set[:,run]
    # # train_idx = torch.where(train_mask != 0)
    dataset.valid_mask = valid_mask_set[:,run]
    # # valid_idx = torch.where(valid_mask != 0)
    dataset.test_mask = test_mask_set[:,run]
    # # test_idx = torch.where(test_mask != 0)
    
    criterion = torch.nn.CrossEntropyLoss()
    PATH_GCN = os.getcwd() + '/saved_models' + '/best_gcn_model_' + str(run)

    if vanilla == 'True':
        print("Not rewiring the input graph...")
        X = dataset.x.to(device)
        A = dataset.edge_index.to(device)
        with_latent = False
    else:
        PATH_MLP = os.getcwd()+ '/saved_models' + '/best_mlp_model_' + str(run)
        mlp = MLP(in_channels=dataset.num_features, hidden_channels = mlp_hidden,  out_channels = dataset.num_classes, dropout = mlp_dropout).to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr = mlp_learning_rate, weight_decay = mlp_weight_decay)
        best_val_acc, test_acc, pred, Y = mlp_training(mlp, dataset, PATH_MLP, dataname, optimizer = optimizer, i = run, epochs = mlp_epoch)
        print("Rewiring the input graph...")
        X_tilda, X_c = generate_latent_feature(ae_hidden, dataset, pred, ae_epoch, ae_learning_rate, ae_weight_decay, device)
        if with_latent == 'True':
            X = X_tilda.to(device)
        else:
            X = dataset.x.to(device)
        st = time.time()
        A_tilda = manipulate_neighbours(dataset, pred, bottom_k, top_k, Y, X_c)
        en = time.time()
        print("TIme elpased during rewiring: ", (en-st))
        A = A_tilda.to(device)
        edge_homo = homophily(A_tilda, dataset.y, method = 'edge')
        print("Edge homophily after rewiring: ", edge_homo)

    
    # defining GNN models
    if model_name == 'GCN':
        model = GCN(dataset, num_layers = 2, mlp_layers = 1, input_dim = dataset.x.shape[1], hidden_dim = gcn_hidden, dropout=gcn_dropout, device = device, latent_dim = ae_hidden, with_latent=with_latent).to(device)
    elif model_name == 'GAT':
        model = GAT(dataset, dataset.x.shape[1], gcn_hidden, dataset.num_classes, ae_hidden, with_latent).to(device)
    elif model_name == 'APPNP':
        model = APPNP_Net(dataset.x.shape[1], gcn_hidden, dataset.num_classes, ae_hidden, with_latent).to(device)
    elif model_name == 'GCNII':
        model = GCNII(dataset.x.shape[1], gcn_hidden, dataset.num_classes, ae_hidden, with_latent, num_layers=2, alpha=0.1, theta=0.1).to(device)
    elif model_name == 'GPRGNN':
        model = GPRGNN(dataset.x.shape[1], gcn_hidden, dataset.num_classes, ae_hidden, with_latent).to(device)
    elif model_name == 'H2GCN':
        model = H2GCN(dataset.x.shape[1], gcn_hidden, dataset.num_classes, A, X.shape[0], ae_hidden, with_latent).to(device)
    # elif model_name == 'LINKX':
    #   model = LINKX(data.x.shape[1], gcn_hidden, data.num_classes, ae_hidden, with_latent, num_layers=2, num_nodes=X.shape[0]).to(device)
    else:
        print("Invalid model name")
    
    optimizer = torch.optim.Adam(model.parameters(), lr = gcn_learning_rate, weight_decay = gcn_weight_decay)

    print(model_name +  " training starts...")
    best_val_acc = 0
    testAcc = []
    for epoch in range(1, gcn_epoch+1):
        loss, val_acc, train_acc = train_new(dataset, model, criterion, optimizer, X.to(device), A.to(device), device, with_latent = with_latent)
        if val_acc > best_val_acc:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'loss': loss},
                        PATH_GCN)
        best_val_acc = val_acc
        chkp = torch.load(PATH_GCN)
        model.load_state_dict(chkp['model_state_dict'])
        train_acc, valid_acc, test_acc = test_new(dataset, model, X.to(device), A.to(device), with_latent = with_latent)
        testAcc.append(test_acc)
        
        if epoch%100==0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train_acc: {train_acc: .4f}, Val_acc: {val_acc: .4f}, Best_val: {best_val_acc: .4f}, Test Acc : {test_acc: .4f}')

    max_test.append(max(testAcc))

    print(f"Test Accuracy: {max(testAcc)}")
    
    # visualize(out, dataname, run, color = dataset.y.cpu())

    cc_matrix = class_compatibility(dataset.y, A, dataname)
    # print("class compatibility matrix \n", cc_matrix)

    # draw_graph(data.edge_index, train_idx, val_idx, test_idx)

    print("---------------------------------------------\n")

print(np.average(max_test)*100," +- ", np.std(max_test)*100)
