from utils import mask_generation
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from models import Autoencoder
import numpy as np
import os

from tqdm import tqdm 

#### MLP
def mlp_training(model, data, path, dataname, optimizer, i, epochs):
  
  PATH = path
  dataname = dataname

  def train(train_mask, val_mask):
        model.train()
        optimizer.zero_grad() 

        out = model(data)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])  
        loss.backward() 
        optimizer.step()  
        pred = out.argmax(dim = 1)
        val_acc = (pred[val_mask] == data.y[val_mask]).float().mean()
        return loss,val_acc

  @torch.no_grad()
  def test(test_mask):
        chkp = torch.load(PATH)
        model.load_state_dict(chkp['model_state_dict'])
        model.eval()
        
        out = model(data)
        pred = out.argmax(dim=1)  
        test_correct = pred[test_mask] == data.y[test_mask] 
        test_acc = int(test_correct.sum()) / int(test_mask.sum())  
        return test_acc, pred, out

  best_val_acc = 0
   
  f = np.load(os.getcwd() + '/splits/' + dataname.title() + '/' + dataname.lower() + '_split_0.6_0.2_'+str(i)+'.npz')
  train_idx, val_idx, test_idx = f['train_mask'], f['val_mask'], f['test_mask']
  train_mask, val_mask, test_mask = mask_generation(train_idx,data.num_nodes), mask_generation(val_idx, data.num_nodes), mask_generation(test_idx, data.num_nodes)
  print("Graph created for file  " + str(i)+ '.....')
  print("MLP Training starts.....")
  min_test_acc = 0
  
  for epoch in tqdm(range(1, epochs+1)):
    loss, val_acc = train(train_mask, val_mask)
    if val_acc > best_val_acc:
      torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'loss': loss},
                PATH)
      best_val_acc = val_acc
    if epoch%100==0:
      print(f' Epoch: {epoch:03d}, Loss: {loss:.4f}, Validation Acc: {val_acc: .4f} (Best validation acc: {best_val_acc: .4f})')
    
  test_acc, pred, out = test(test_mask)
  
  print(f"Test Acc on MLP : {test_acc: .4f}")
  return best_val_acc, test_acc, pred, out


#### AutoEncoder
def generate_latent_feature(latent_dim, data, pred, epoch, l_r, w_d, device):
      
      dict_of_labels = class_dict_creation(data, pred)
      # print("dict of label ", dict_of_labels)
      
      dummy_x = np.zeros((data.num_nodes, data.num_features + latent_dim))
      class_embed = torch.zeros(len(dict_of_labels), latent_dim).to(device)
      
      # print("Autoencoder training starts...")
      for i in tqdm(range(len(dict_of_labels))):
            if len(dict_of_labels[i]) == 0:
                  latent_vec = torch.zeros(len(dict_of_labels[i]), latent_dim).to(device)
            else:
                model = Autoencoder(input_dim=data.num_features, hidden_dim=latent_dim).to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=l_r, weight_decay = w_d)

                input_data = data.x[dict_of_labels[i]]
                
                model.train()
                for epoch in range(epoch):
                    optimizer.zero_grad()
                    decoded, encoded = model(input_data)
                    loss = criterion(decoded, input_data)
                    if epoch%50==0:
                          pass
                        # print(f"Loss: {loss:.4f}")
                    loss.backward()
                    optimizer.step()
                latent_vec = encoded[0]
                class_embed[i] = latent_vec
                # print("Class embeddings ", class_embed.shape)
                
            for j in range(len(dict_of_labels[i])):
                a = torch.cat((data.x[dict_of_labels[i][j]], latent_vec)).cpu()
                dummy_x[dict_of_labels[i][j]] = a.detach()

      dummy_x = dummy_x.astype(np.double)
      latent_x = torch.tensor(dummy_x)
      latent_x = latent_x.to(torch.float32)
      return latent_x, class_embed


# general train function
def train(model, data, criterion, optimizer, latent_x, new_edge_index, train_mask, val_mask, with_latent):
    model.train()
    optimizer.zero_grad()  
    if with_latent == 'True':
      out = model(latent_x, new_edge_index) 
    else:
      out = model(data.x, new_edge_index) 
    pred = out.argmax(dim = 1)
    loss = criterion(out[train_mask], data.y[train_mask])  
    loss.backward() 
    optimizer.step() 
    val_acc = (pred[val_mask] == data.y[val_mask]).float().mean()
    train_acc = (pred[train_mask] == data.y[train_mask]).float().mean()
    return loss, val_acc, train_acc
  
