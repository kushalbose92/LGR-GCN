import torch

def test(model, data, PATH, latent_x, new_edge_index, test_mask, with_latent):
      chkp = torch.load(PATH)
      model.load_state_dict(chkp['model_state_dict'])

      model.eval()
      if with_latent:
            out = model(latent_x, new_edge_index) 
      else:
          out = model(data.x, new_edge_index) 
      pred = out.argmax(dim=1)  
      test_correct = pred[test_mask] == data.y[test_mask] 
      test_acc = int(test_correct.sum()) / int(test_mask.sum()) 
      return test_acc, out