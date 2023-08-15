import torch
import numpy as np
import argparse
import os

from train import run_training
from models import MLP, GCN
from utils import manipulate_neighbours, mask_generation, visualize
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from torch_geometric.utils import homophily
from train import generate_latent_feature, train
from test import test
from datacreater import *

if __name__ == '__main__':

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

  # parser.add_argument('--alpha', type=float)
  # parser.add_argument('--beta', type=float)
  # parser.add_argument('--gamma', type=float)
  parser.add_argument('--with_latent', type=None)
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

  # alpha = args.alpha
  # beta = args.beta
  # gamma = args.gamma
  with_latent = args.with_latent
  device = args.device

  # setting seeds
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if device == 'cuda:0':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


  #Creating Data object
  if dataname == 'Cora' or dataname == 'Citeseer' or dataname == 'Pubmed':
    data = PlanetoidData(dataname)
  elif dataname == 'Chameleon' or dataname == 'Squirrel':
    data = WikipediaData(dataname)
  elif dataname == 'Wisconsin' or dataname == 'Cornell' or dataname == 'Texas':
    data = WebKBData(dataname)
  elif dataname == 'Film':
    data = ActorData()
  else:
    print("Incorrect name of dataset")

  print("Loading " + dataname)
  data.x = data.x.to(device)
  data.edge_index = data.edge_index.to(device)
  data.y = data.y.to(device)
  max_test = []
  
  edge_homo = homophily(data.edge_index, data.y, method = 'edge')
  # print("Edge homophily before rewiring:  ", edge_homo)
  
  
  edge_homo_list = []
  for i in range(10):
        print("-----------------For Split " + str(i) + "-------------------------")
        min_test_acc = 0
        PATH_MLP = os.getcwd()+ '/saved_models' + '/best_mlp_model_' + str(i)
        mlp = MLP(in_channels=data.num_features, hidden_channels = mlp_hidden,  out_channels = data.num_classes, dropout = mlp_dropout).to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr = mlp_learning_rate, weight_decay = mlp_weight_decay)
        best_val_acc, test_acc, pred = run_training(mlp, data, PATH_MLP, dataname, optimizer = optimizer, i = i, epochs = mlp_epoch)
        # print(f"Test Accuracy on MLP: {test_acc}")

        x_noise = torch.randn(1, mlp_hidden).to(device)
        sim_weights = mlp.forward_1(x_noise)
        alpha, beta, gamma = sim_weights[0].item(), sim_weights[1].item(), sim_weights[2].item()
        # print("Weights ", alpha, beta, gamma)

        X_tilda = generate_latent_feature(ae_hidden, data, pred, ae_epoch, ae_learning_rate, ae_weight_decay, device, alpha, beta, gamma)
        
        print("Rewiring the input graph...")
        A_tilda = manipulate_neighbours(data, pred, bottom_k, top_k, alpha, beta, gamma)
        
        # edge_homo = homophily(A_tilda, data.y, method = 'edge')
        # print("Edge homophily for split " + str(i) + " is: ", edge_homo)
        # edge_homo_list.append(edge_homo)
        
        criterion = torch.nn.CrossEntropyLoss()
        PATH_GCN = os.getcwd() + '/saved_models' + '/best_gcn_model_' + str(i)

        testAcc = []
        model = GCN(data, hidden_channels = gcn_hidden, dropout=gcn_dropout, latent_dim = ae_hidden, with_latent=with_latent).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = gcn_learning_rate, weight_decay = gcn_weight_decay)
        
        f = np.load(os.getcwd() + '/splits/' + dataname.title() + '/' + dataname.lower() + '_split_0.6_0.2_'+str(i)+'.npz')
        train_idx, val_idx, test_idx = f['train_mask'], f['val_mask'], f['test_mask']
        train_mask, val_mask, test_mask = mask_generation(train_idx,data.num_nodes), mask_generation(val_idx, data.num_nodes), mask_generation(test_idx, data.num_nodes)
        print("GCN training starts....")
        best_val_acc = 0
        for epoch in range(1, gcn_epoch+1):
          loss, val_acc, train_acc = train(model, data, criterion, optimizer, X_tilda.to(device), A_tilda.to(device), train_mask, val_mask, with_latent = with_latent)
          if val_acc > best_val_acc:
            torch.save({'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'val_acc': val_acc,
                      'loss': loss},
                      PATH_GCN)
            best_val_acc = val_acc
            test_acc, out = test(model, data, PATH_GCN, X_tilda.to(device), A_tilda.to(device), test_mask, with_latent = with_latent)
            testAcc.append(test_acc)
            
          if epoch%100==0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train_acc: {train_acc: .4f}, Val_acc: {val_acc: .4f}, Best_val: {best_val_acc: .4f}, Test Acc : {test_acc: .4f}')

        max_test.append(max(testAcc))

        print(f"Test Accuracy: {max(testAcc)}")
        visualize(out, dataname, i, color = data.y.cpu())

        # draw_graph(data.edge_index, train_idx, val_idx, test_idx)
        print("---------------------------------------------\n")
  print(np.average(max_test)*100," +- ", np.std(max_test)*100)
  # print(edge_homo_list)
  # print("Average Edge Homophily: ", np.average(edge_homo_list))

 
