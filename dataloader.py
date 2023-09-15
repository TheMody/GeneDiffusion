import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import os

def load_data(processed = True):
    print("Loading data...")
    dataset_X,Y=pickle.load(open('data/sigSNPs_pca.features.pkl','rb'))
    if processed:
        dataset_X = pickle.load(open("data/processed_ds","rb"))
    print("Data loaded!")

    return np.asarray(dataset_X), np.asarray(Y)

class SynGeneticDataset(Dataset):
    def __init__(self):
        self.path = "syn_data/"
        self.all_file_paths = [self.path + file for file in os.listdir(self.path)]
        # for file in os.listdir(self.path):
        #     self.x,self.y = torch.load(open(file,"rb"))

    def __len__(self):
        return len(self.all_file_paths)

    def __getitem__(self, idx):
        genome, label = torch.load(self.all_file_paths[idx])
        label = F.one_hot(label.squeeze(),2)
        return genome, label.float()

class GeneticDataset(Dataset):
    def __init__(self, processed = True):
        self.x,self.y = load_data(processed = processed)
        self.processed = processed
     #   self.x = self.x[:int(0.1*len(self.x))]
     #   self.y = self.y[:int(0.1*len(self.y))]
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
        label = self.y[idx]
        return genome, label

def GeneticDataloaders(batchsize, processed = True):
    dataset = GeneticDataset(processed)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize,
                                           shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize,
                                          shuffle=False, num_workers=0)
    return train_dataloader,test_dataloader

















def preprocess_data():
    ds = load_data()
    names = list(ds.columns.values)
    for i in range(len(names)):
        split  = names[i].split(":")
        names[i] = split[0]+":"+split[1]

    _, countunique = np.unique(names, return_counts=True)

    max_length = np.max(countunique)
    print(max_length)
    tokenized_ds = []
    for i in tqdm(range(len(ds))):
        datapoint = np.asanyarray(ds.iloc[i,:].values)
        tokens = []
        current = 0
        for count in countunique:
            token = np.zeros(max_length)
            token[:count] = datapoint[current:current+count]
            tokens.append(token)
            current += count
        datapoint = np.asarray(tokens)
        tokenized_ds.append(datapoint)
    # break

    tokenized_ds = np.asarray(tokenized_ds)
    print(tokenized_ds.shape)
    fileObject = open("data/processed_ds", 'wb')
    pickle.dump(tokenized_ds,fileObject )
    fileObject.close()

#preprocess_data()

def load_processed_data():
    fileObject = open("data/processed_ds", 'rb')
    ds = pickle.load(fileObject)
    fileObject.close()
    ds = ds.astype(np.float32)
    _,Y=pickle.load(open('data/sigSNPs_pca.features.pkl','rb'))
    return ds,Y



#ds = load_processed_data()

class GeneticDatasetpreprocessed(Dataset):
    def __init__(self):
        self.x,self.y = load_processed_data()
     #   self.x = self.x[:int(0.1*len(self.x))]
     #   self.y = self.y[:int(0.1*len(self.y))]
        #now we shuffle x and y
        np.random.seed(42)
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.y = self.y[p]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        genome = self.x[idx]
        label = self.y[idx]
        return genome, label