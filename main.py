import torch
import numpy as np
import argparse
import os

from models import MLP, GCN
from expt_models import GAT, APPNP_Net, GGCN, GCNII, GPRGNN, H2GCN, LINKX
from utils import *
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from torch_geometric.utils import homophily
from train import generate_latent_feature, train, mlp_training
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

  parser.add_argument('--model', type=None)
  parser.add_argument('--vanilla', type=None)
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

  model_name = args.model
  vanilla = args.vanilla
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
  
  # W = (YX_c)(YX_c)^t
  
  edge_homo_list = []
  for i in range(10):
        print("-----------------For Split " + str(i) + "-------------------------")
        min_test_acc = 0
        PATH_MLP = os.getcwd()+ '/saved_models' + '/best_mlp_model_' + str(i)
        mlp = MLP(in_channels=data.num_features, hidden_channels = mlp_hidden,  out_channels = data.num_classes, dropout = mlp_dropout).to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr = mlp_learning_rate, weight_decay = mlp_weight_decay)
        best_val_acc, test_acc, pred, Y = mlp_training(mlp, data, PATH_MLP, dataname, optimizer = optimizer, i = i, epochs = mlp_epoch)
        # print(f"Test Accuracy on MLP: {test_acc}")

        X_tilda, X_c = generate_latent_feature(ae_hidden, data, pred, ae_epoch, ae_learning_rate, ae_weight_decay, device)
        
        print("Rewiring the input graph...")
        A_tilda = manipulate_neighbours(data, pred, bottom_k, top_k, Y, X_c)
        
        # edge_homo = homophily(A_tilda, data.y, method = 'edge')
        # print("Edge homophily for split " + str(i) + " is: ", edge_homo)
        # edge_homo_list.append(edge_homo)
        
        criterion = torch.nn.CrossEntropyLoss()
        PATH_GCN = os.getcwd() + '/saved_models' + '/best_gcn_model_' + str(i)

        # integrating with GNN architecture
        testAcc = []

        if vanilla == 'True':
          X = data.x.to(device)
          A = data.edge_index.to(device)
          with_latent = False
        else:
          X = X_tilda.to(device)
          A = A_tilda.to(device)


        # defining GNN models
        if model_name == 'GCN':
          model = GCN(data, hidden_channels = gcn_hidden, dropout=gcn_dropout, latent_dim = ae_hidden, with_latent=with_latent).to(device)
        elif model_name == 'GAT':
          model = GAT(data.x.shape[1], gcn_hidden, data.num_classes, ae_hidden, with_latent).to(device)
        elif model_name == 'APPNP':
          model = APPNP_Net(data.x.shape[1], gcn_hidden, data.num_classes, ae_hidden, with_latent).to(device)
        elif model_name == 'GCNII':
          model = GCNII(data.x.shape[1], gcn_hidden, data.num_classes, ae_hidden, with_latent, num_layers=2, alpha=0.1, theta=0.1).to(device)
        elif model_name == 'GPRGNN':
          model = GPRGNN(data.x.shape[1], gcn_hidden, data.num_classes, ae_hidden, with_latent).to(device)
        elif model_name == 'H2GCN':
          model = H2GCN(data.x.shape[1], gcn_hidden, data.num_classes, A, X.shape[0], ae_hidden, with_latent).to(device)
        # elif model_name == 'LINKX':
        #   model = LINKX(data.x.shape[1], gcn_hidden, data.num_classes, ae_hidden, with_latent, num_layers=2, num_nodes=X.shape[0]).to(device)
        else:
          print("Invalid model name")
        
        optimizer = torch.optim.Adam(model.parameters(), lr = gcn_learning_rate, weight_decay = gcn_weight_decay)
        
        f = np.load(os.getcwd() + '/splits/' + dataname.title() + '/' + dataname.lower() + '_split_0.6_0.2_'+str(i)+'.npz')
        train_idx, val_idx, test_idx = f['train_mask'], f['val_mask'], f['test_mask']
        train_mask, val_mask, test_mask = mask_generation(train_idx,data.num_nodes), mask_generation(val_idx, data.num_nodes), mask_generation(test_idx, data.num_nodes)

        print(model_name +  " training starts....")
        best_val_acc = 0
        for epoch in range(1, gcn_epoch+1):
          loss, val_acc, train_acc = train(model, data, criterion, optimizer, X.to(device), A.to(device), train_mask, val_mask, with_latent = with_latent)
          if val_acc > best_val_acc:
            torch.save({'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'val_acc': val_acc,
                      'loss': loss},
                      PATH_GCN)
            best_val_acc = val_acc
            test_acc, out = test(model, data, PATH_GCN, X.to(device), A.to(device), test_mask, with_latent = with_latent)
            testAcc.append(test_acc)
            
          if epoch%100==0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train_acc: {train_acc: .4f}, Val_acc: {val_acc: .4f}, Best_val: {best_val_acc: .4f}, Test Acc : {test_acc: .4f}')

        max_test.append(max(testAcc))

        print(f"Test Accuracy: {max(testAcc)}")
        visualize(out, dataname, i, color = data.y.cpu())

        # cc_matrix = class_compatibility(data.y, A, dataname)
        # print("class compatibility matrix ", cc_matrix)

        # draw_graph(data.edge_index, train_idx, val_idx, test_idx)

        print("---------------------------------------------\n")

  print(np.average(max_test)*100," +- ", np.std(max_test)*100)

  # print(edge_homo_list)
  # print("Average Edge Homophily: ", np.average(edge_homo_list))

 
