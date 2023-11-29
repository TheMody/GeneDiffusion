import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import os
from config import *

def load_data(processed = True):
    print("Loading data...")
    dataset_X,Y=pickle.load(open('data/sigSNPs_pca.features.pkl','rb'))
    if processed:
        dataset_X = pickle.load(open("data/processed_ds","rb"))
    print("Data loaded!")

    return np.asarray(dataset_X), np.asarray(Y)

def processes_data():
    ds,Y=pickle.load(open('data/sigSNPs_pca.features.pkl','rb'))
    names = list(ds.columns.values)
   # print(names)
    for i in range(len(names)):
        split  = names[i].split(":")
        names[i] = split[0]+":"+split[1]
   # print(names)

    n_unique, countunique = np.unique(names, return_counts=True)
   # print(len(n_unique))
   # print(countunique)
    max_length = np.max(countunique)
   # print(max_length)
    tokenized_ds = []
    for i in tqdm(range(len(ds))):
        datapoint = np.asanyarray(ds.iloc[i,:].values)
       # print(datapoint)
      #  print(datapoint.shape)
        tokens = []
        last_name = names[0]
        token = np.zeros(max_length)
        pos_int_token = 0
        for a,value in enumerate(datapoint):
            if not last_name == names[a]:
                tokens.append(token)
                last_name = names[a]
                pos_int_token = 0
                token = np.zeros(max_length)
            token[pos_int_token] = value
            pos_int_token += 1
        tokens.append(token)
       # print(len(tokens))
        datapoint = np.asarray(tokens)
      #  print(datapoint.shape)
        #print(datapoint.shape)
        tokenized_ds.append(datapoint)
    # break

    tokenized_ds = np.asarray(tokenized_ds)
    print(tokenized_ds.shape)
    fileObject = open("data/processed_ds", 'wb')
    pickle.dump(tokenized_ds,fileObject )
    fileObject.close()
    # for column in dataset_X:
    #     print(column)


class SynGeneticDataset(Dataset):
    def __init__(self, path = save_path+"/", label = None):
        self.label = label
        self.path = path
        self.all_file_paths = [self.path + file for file in os.listdir(self.path) if file != "model.pt" and file != "vaemodel.pt"]
        print("path of dataset", self.path)
        print("len of syn dataset", len(self.all_file_paths))
        # for file in os.listdir(self.path):
        #     self.x,self.y = torch.load(open(file,"rb"))

    def __len__(self):
        return len(self.all_file_paths)

    def __getitem__(self, idx):
       # print(self.all_file_paths[idx])
        genome, label = torch.load(self.all_file_paths[idx])
        if self.label is not None:
            label = torch.tensor(self.label)
      #  print(label)
        #label = F.one_hot(label.squeeze(),2)
        return genome.permute(1,0), label


class GeneticDataset(Dataset):
    def __init__(self, processed = True, normalize = normalize_data, label = None):
        #super(GeneticDataset, self).__init__()
        self.x,self.y = load_data(processed = processed)
        self.label = label
        if normalize:
           # xstd = np.std(self.x,axis=0)
           # xstd[xstd == 0.0] +=1
            self.x-np.mean(self.x,axis=0)
            max = np.max(np.abs(self.x))
         #   max[max == 0.0] +=1
            self.x = self.x / max# xstd
          #  self.x = self.x / 2 + 0.5
       # self.std = np.std(self.x,axis=0)
       # self.std = F.pad(torch.tensor(self.std), (0,0,0, 18432 - self.std.shape[0]), "constant", 0).float()
        #self.std[self.std == 0.0] +=1
        self.processed = processed
        #now we shuffle x and y
        np.random.seed(42)
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.y = self.y[p]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        genome = self.x[idx]
      #  genome = genome[None,...]
        if self.processed:
            genome = F.pad(torch.tensor(genome), (0,0,0, 18432 - genome.shape[0]), "constant", 0)
     #   print(genome.shape)
        label = torch.tensor(self.y[idx])
        if self.label is not None:
            label = torch.tensor(self.label)
        return genome.float(), label

def GeneticDataSets( processed = True, label = None):
    dataset = GeneticDataset(processed, label = label)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator = generator1)
    return train_dataset,test_dataset

def GeneticDataloaders(batchsize, processed = True):
    train_dataset,test_dataset = GeneticDataSets(processed)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize,
                                           shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize,
                                          shuffle=True)
    return train_dataloader,test_dataloader




#processes_data()