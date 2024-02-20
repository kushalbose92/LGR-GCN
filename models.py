from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GCN2Conv
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

        self.lin4 = nn.Linear(hidden_channels, 16)
        self.lin5 = nn.Linear(16, 8)
        self.lin6 = nn.Linear(8, 3)
        

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

    # def forward_1(self, x):
    #     x = self.lin4(x)
    #     x = F.relu(x)
    #     x = F.dropout(x, p=0.50, training=True)
    #     x = self.lin5(x)
    #     x = F.relu(x)
    #     x = F.dropout(x, p=0.50, training=True)
    #     x = self.lin6(x)
    #     x = torch.exp(x)
    #     x = torch.sort(x, descending=True)[0][0]
    #     return x
    

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    

class GCN(torch.nn.Module):
    def __init__(self, data, hidden_channels, dropout, latent_dim, with_latent):
        super(GCN, self).__init__()
        self.dropout = dropout

        if with_latent == 'True':
            self.conv1 = GCNConv(data.num_features + latent_dim, hidden_channels)
        else:
            self.conv1 = GCNConv(data.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, data.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim = 1)

