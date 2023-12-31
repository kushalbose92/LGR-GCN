import numpy as np
import random
import torch
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Actor
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import Planetoid


class WebKBData():
        def __init__(self, datasetname):

                dataset = WebKB(root='data/WebKB', name = datasetname, transform = NormalizeFeatures())
                data = dataset[0]

                self.data = data
                self.name = datasetname
                self.length = len(dataset)
                self.num_features = dataset.num_features
                self.num_classes = dataset.num_classes

                self.num_nodes = data.num_nodes
                self.num_edges = data.num_edges
                self.avg_node_degree = (data.num_edges / data.num_nodes)
                self.train_mask = self.data.train_mask[:, 0]
                self.val_mask = self.data.val_mask[:, 0]
                self.test_mask = self.data.test_mask[:, 0]
                self.contains_isolated_nodes = data.has_isolated_nodes()
                self.data_contains_self_loops = data.has_self_loops()
                self.is_undirected = data.is_undirected()

                self.x = data.x
                self.y = data.y
                self.edge_index = data.edge_index

    
class ActorData():
        def __init__(self):

                dataset = Actor(root='data/WebKB',  transform = NormalizeFeatures())
                data = dataset[0]
                datasetname = 'Film'
                self.data = data
                self.length = len(dataset)
                self.num_features = dataset.num_features
                self.num_classes = dataset.num_classes

                self.num_nodes = data.num_nodes
                self.num_edges = data.num_edges
                self.avg_node_degree = (data.num_edges / data.num_nodes)
                self.train_mask = self.data.train_mask[:, 0]
                self.val_mask = self.data.val_mask[:, 0]
                self.test_mask = self.data.test_mask[:, 0]
                self.contains_isolated_nodes = data.has_isolated_nodes()
                self.data_contains_self_loops = data.has_self_loops()
                self.is_undirected = data.is_undirected()

                self.x = data.x
                self.y = data.y
                self.edge_index = data.edge_index
                

class PlanetoidData():
  def __init__(self, datasetname):
    dataset = Planetoid(root = 'root/Planetoid', name = datasetname, transform=NormalizeFeatures() )
    self.data = dataset[0]
    self.train_mask = self.data.train_mask
    self.val_mask = self.data.val_mask
    self.test_mask = self.data.test_mask
    self.x = self.data.x
    self.edge_index = self.data.edge_index
    self.y = self.data.y
    self.num_features = dataset.num_features
    self.num_classes = dataset.num_classes

    self.num_nodes = self.data.num_nodes
    self.num_edges = self.data.num_edges
    

class WikipediaData():
       def __init__(self, datasetname):

                dataset = WikipediaNetwork(root='data/WikipediaNetwork', name = datasetname, transform = NormalizeFeatures())
                data = dataset[0]

                self.data = data
                self.name = datasetname
                self.length = len(dataset)
                self.num_features = dataset.num_features
                self.num_classes = dataset.num_classes

                self.num_nodes = data.num_nodes
                self.num_edges = data.num_edges
                self.avg_node_degree = (data.num_edges / data.num_nodes)
                self.train_mask = self.data.train_mask[:, 0]
                self.val_mask = self.data.val_mask[:, 0]
                self.test_mask = self.data.test_mask[:, 0]
                self.contains_isolated_nodes = data.has_isolated_nodes()
                self.data_contains_self_loops = data.has_self_loops()
                self.is_undirected = data.is_undirected()

                self.x = data.x
                self.y = data.y
                self.edge_index = data.edge_index

                
                
