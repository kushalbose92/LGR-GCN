import torch
import torch.nn.functional as F 
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from tqdm import tqdm 
import seaborn as sns

from torch_geometric.utils import to_dense_adj
import networkx as nx
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def visualize(h, dataname, i, color):
    z = TSNE(n_components=2, perplexity = 70).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(5, 5))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=10, c=color, cmap="Set1")
    plt.savefig(os.getcwd() + "/visuals/" + dataname + '_embedding_' + str(i) + '.png')
    plt.close()

def mask_generation(index, num_nodes):
    mask = torch.zeros(num_nodes, dtype = torch.bool)
    mask[index] = 1
    return mask

def label_guided_similarity(data, pred, Y, X_c):
    # pred = pred.cpu()
    # #Class Label Dictionary Creation
    # pred[np.where(data.train_mask)[0]] = data.y[np.where(data.train_mask)[0]].cpu()
    # dict_of_labels = {label : np.where(pred == label)[0] for label in range(data.num_classes)}
    
    # dict_of_labels = class_dict_creation(data, pred)

    # Feature Similarity Matrix
    feature_similarity = cosine_similarity(data.x.cpu())

    # # Label Guided Similarity Matrix
    # label_score_matrix = np.full((data.num_nodes, data.num_nodes), gamma)
    # for i in range(data.num_nodes):
    #     label_score_matrix[i][dict_of_labels[pred[i].item()]] = beta
    #     if i in np.where(data.train_mask)[0]:
    #         class_label = data.y[i].item()
    #         same_class_train = np.intersect1d(np.where(data.y.cpu() == class_label)[0], np.where(data.train_mask)[0])
    #         label_score_matrix[i][same_class_train] = alpha
    #     label_score_matrix[i][i] = 1

    # # Hadamard product
    # label_score_matrix = np.zeros((data.num_nodes, data.num_nodes))
    # modified_similarity = np.multiply(feature_similarity, label_score_matrix)
    
    label_score_matrix = F.softmax(torch.mm(torch.mm(Y, X_c), torch.t(torch.mm(Y, X_c))))
    modified_similarity = np.multiply(feature_similarity, label_score_matrix.detach().cpu().numpy())
    # print(modified_similarity.shape)
    
    # return feature_similarity
    return modified_similarity


def class_dict_creation(data, pred):
    pred = pred.cpu()
    #Class Label Dictionary Creation
    pred[np.where(data.train_mask)[0]] = data.y[np.where(data.train_mask)[0]].cpu()
    dict_of_labels = {label : np.where(pred == label)[0] for label in range(data.num_classes)}
    
    return dict_of_labels


def manipulate_neighbours(data, pred, delete_k1, insert_k2, Y, X_c):
    delete_k = delete_k1
    insert_k = insert_k2
    sim = label_guided_similarity(data, pred, Y, X_c)
    dict_of_labels = class_dict_creation(data, pred)
    edges = []
    pred = pred.cpu()
    data.edge_index = data.edge_index.cpu()


    ### Using for Loop
    for i in tqdm(range(data.num_nodes)):
    
        #Get the neighbourhood of ith node
        neighbour = data.edge_index[1][np.where(data.edge_index[0] == i)[0]].cpu()

        # non_neighbour = data.edge_index[1][np.where(data.edge_index[0] != i)[0]].cpu()

        # nodes_not_in_class_similarity = sim[i][non_neighbour]

        #Get the index of the nodes that are present in the same class
        nodes_in_class = dict_of_labels[pred[i].item()]

        nodes_not_in_class = list(set(np.arange(data.num_nodes)) - set(nodes_in_class))

        nodes_not_in_class_similarity = sim[i][nodes_not_in_class]
        
        top_similarity_nodes_not_in_class = sorted(nodes_not_in_class_similarity, reverse = True)[:insert_k]

        #Get the nodes that are not present in same class
        top_similar_nodes_not_in_class = np.where(np.isin(sim[i], top_similarity_nodes_not_in_class))[0][:insert_k]
        
        
        least_similar_nodes = np.argsort(sim[i][neighbour])[:delete_k]
        nodes_to_be_removed = neighbour[least_similar_nodes]
        
        #Removing nodes
        if i == 0:
            x = np.array(data.edge_index[0])
            y = np.array(data.edge_index[1])

        else:
            x = np.array(new_x_tens)
            y = np.array(new_y_tens)
            
        idx_x = []
        idx_y = []
        
        
        idx_x = np.where(x == i)[0]*len(nodes_to_be_removed)
        idx_y = np.where(y == nodes_to_be_removed)[0]
        idx_rmv = np.intersect1d(idx_x, idx_y)
        x1 = np.delete(x, idx_rmv)
        y1 = np.delete(y, idx_rmv)

        #Rewired graph building

        #Edge index with added nodes
        edges = []
        for j in range(len(top_similar_nodes_not_in_class)):
            edges.append((i, top_similar_nodes_not_in_class[j]))
        if i==0:
            new_x, new_y = list(x[:]),  list(y[:])
        else:
            new_x, new_y = list(x1[:]),  list(y1[:])

        if insert_k > 0:
            new_x, new_y = zip(*edges)
        
        # for j in range(len(edges)):
        #     new_x.append(edges[j][0])
        #     new_y.append(edges[j][1])
        new_edge_index = []
        new_x_tens = torch.tensor(new_x)
        new_y_tens = torch.tensor(new_y)
    
    new_edge_index = torch.vstack((new_x_tens, new_y_tens))

    return new_edge_index


def class_compatibility(y, edge_index, dataname):
    device = edge_index.device
    num_labels = max(y).item() + 1
    num_nodes = y.shape[0] + 1
    print(num_nodes, "   ", num_labels)
    Y = torch.zeros(num_nodes, num_labels).to(device)
    Y = Y.scatter_(1, y.unsqueeze(1), 1)
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    # A += (torch.eye(num_nodes)* 1e-10).to(device)
    E = torch.ones(num_nodes, num_labels).to(device)
    num = torch.matmul(torch.matmul(torch.t(Y), A), Y)
    den = torch.matmul(torch.matmul(torch.t(Y), A), E)
    cc_matrix = num / (den + 1e-3)
    cc_matrix = cc_matrix.detach().cpu().numpy()

    # Create a heatmap with more arguments
    ax = sns.heatmap(
            cc_matrix,
            annot=True,          # Show numerical annotations in each cell
            fmt='.4f',           # Format for annotations (two decimal places)
            cmap='Oranges_r',     # Colormap (coolwarm is just an example, choose any other)
            cbar=False,           # Show colorbar
            linewidths=0.5,      # Width of the lines that divide each cell
            square=True,         # Make the cells square-shaped
            xticklabels=True,  # Custom x-axis labels
            yticklabels=True,  # Custom y-axis labels
            # cbar_kws={'yticklabels': []},  # Custom label for the colorbar
    )
    # cbar = ax.collections[0].colorbar
    # cbar.set_ticks([])

    # Add labels and title
    # plt.xlabel('X-axis (Columns)')
    # plt.ylabel('Y-axis (Rows)')
    # plt.title('Customized Heatmap Example')

    plt.savefig(os.getcwd() + "/class_compatibility_matrix_" + dataname + "_.png")
    plt.close()

    return cc_matrix








