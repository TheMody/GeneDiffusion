import torch
from model import MLPModel
from dataloader import  GeneticDataloaders, SynGeneticDataset, GeneticDataSets, GeneticDataset
import numpy as np
import wandb
from cosine_scheduler import CosineWarmupScheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import *

def preprocessing_function(x):
   # x = x*max_value
    # self.x-np.mean(self.x,axis=0)
    # max = np.max(np.abs(self.x),axis=0)
    # max[max == 0.0] +=1
    # self.x = self.x / max
    # self.x = self.x / 2 + 0.5
    return x



def train_classifier(train_frac = 0.8):
    #basic building blocks
    model = MLPModel(num_input=num_channels*gene_size)#75584)#

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_classifier)
    loss_fn = torch.nn.CrossEntropyLoss()
   # wandb.init(project="diffusionGene", config=config)
    #data
    dataset = GeneticDataset()
    train_size = int(train_frac * len(dataset))
    test_size = len(dataset) - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator = generator1)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"],
                                           shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"],
                                          shuffle=True)
    max_step = 10000/16*5
    scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=max_step)

    running_loss = 0.0
    best_acc = 0.0
    
    step = 0
    while True:
        
        dataloader_iter = iter(train_dataloader)
        for i in range(len(train_dataloader) // gradient_accumulation_steps):
                optimizer.zero_grad()
                accloss = 0.0
                accacc = 0.0
                for micro_step in range(gradient_accumulation_steps):
                    inputs, labels = next(dataloader_iter)
                    inputs = inputs.float().to(device)
                    #inputs = preprocessing_function(inputs)
                   # print(inputs[0])
                   # r_inputs = encoder_model(inputs.permute(0,2,1), train = False).permute(0,2,1)
                    labels = labels.to(device)
                 #   print(labels.shape)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    with torch.no_grad():
                      #  accacc += torch.sum(torch.argmax(outputs, axis = 1) == torch.argmax(labels, axis = 1))/labels.shape[0]
                        accacc += torch.sum(torch.argmax(outputs, axis = 1) ==labels)/labels.shape[0]
                        accloss += loss.item()

                    loss.backward()
                step += 1
                acc = accacc/gradient_accumulation_steps

                # Adjust learning weights
                optimizer.step()
                scheduler.step()

                # Gather data and report
                log_freq = 1
                running_loss += accloss
                if i % log_freq == 0:
                 #   acc = np.sum(np.argmax(outputs.detach().cpu().numpy(), axis = 1) == labels.detach().cpu().numpy())/labels.shape[0]
                    avg_loss = running_loss / log_freq # loss per batch
                    log_dict = {"avg_loss_classifier": avg_loss, "accuracy_classifier": acc, "lr_classifier": scheduler.get_lr()[0]}
                   # wandb.log(log_dict)
                    running_loss = 0.
                    if i % 20 == 0:
                        print('  batch {} loss: {} accuracy: {}'.format(i + 1, avg_loss, acc))
                        
        if step > max_step:
          # #evaluate model after epoch
          with torch.no_grad():
              accummulated_acc = 0.0
              accummulated_loss = 0.0
              for i, data in enumerate(test_dataloader):
                  inputs, labels = data
                #  print(inputs.shape)
                  inputs = inputs.float().to(device)
                #  print(inputs[0])
                  #inputs = preprocessing_function(inputs)
                  labels = labels.to(device)
                  outputs = model(inputs)
                  loss = loss_fn(outputs, labels)
                  acc = torch.sum(torch.argmax(outputs, axis = 1) == labels)/labels.shape[0]
                  accummulated_acc += acc.item()
                  accummulated_loss += loss.item()
              avg_acc = accummulated_acc/len(test_dataloader)
              avg_loss = accummulated_loss/len(test_dataloader)
            # wandb.log({"test_loss_classifier": avg_loss, "test_accuracy_classifier": avg_acc})
              print(' step {} loss: {} accuracy: {}'.format(step+1, avg_loss, avg_acc))
          if avg_acc > best_acc:
              best_acc = avg_acc
              #torch.save(model,"classification_models/model.pt")
            # print(f"saved new best model with acc {best_acc}")
          break

    return best_acc


if __name__ == "__main__":
  train_fracs = [0.01,0.02,0.05,0.1,0.2,0.8]
  best_accs = []
  for train_frac in train_fracs:
    best_acc = train_classifier(train_frac)
    print(best_acc)
    best_accs.append(best_acc)
  print(best_accs)
  #[0.7099961180309331, 0.768921568627451, 0.7936286407863438, 0.8186806148692749, 0.8429394812680115, 0.8807471264367817] for [0.01,0.02,0.05,0.1,0.2,0.8]
  from matplotlib import pyplot as plt
  plt.plot(train_fracs, best_accs)
  plt.xlabel("train_frac")
  plt.ylabel("best_acc")
  plt.show()