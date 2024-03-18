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
        self.all_file_paths = [self.path + file for file in os.listdir(self.path) if file != "model.pt" and file != "vaemodel.pt" and file != "histogramm.png"]
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
    #    print(genome)
        return genome.permute(1,0), label


class GeneticDataset(Dataset):
    def __init__(self, processed = True, normalize = normalize_data, label = None, percent_unlabeled = percent_unlabeled, train = True):
        #super(GeneticDataset, self).__init__()
        self.x,self.y = load_data(processed = processed)

        print("len of dataset", len(self.x))
        
        self.processed = processed
        self.label = label
        if normalize:
            xstd = np.std(self.x, axis = 0)
            xstd[xstd == 0.0] +=1
            self.x-np.mean(self.x,axis=0)
            self.x = self.x / xstd 
        

        if percent_unlabeled != 0:
            self.y[:int(len(self.y)*percent_unlabeled)] = 2

        #now we shuffle x and y
        np.random.seed(42)
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.y = self.y[p]


        #now we split the data into train and test but balance the test set by taking the same amount of each class
        # we take 500 test samples of each class
        num_test_samples = test_set_size//2

        length_test_p = 0
        length_test_n = 0
        self.indices= []
        for i,y in enumerate(self.y):
            if y == 0:
                if length_test_n >= num_test_samples:
                    continue
                self.indices.append(i)
                length_test_n += 1
            else:
                if length_test_p >= num_test_samples:
                    continue
                self.indices.append(i)
                length_test_p += 1
            if length_test_n >= num_test_samples and length_test_p >= num_test_samples:
                break

        if not train:
            self.x = self.x[self.indices]
            self.y = self.y[self.indices]
            print("len of test set", len(self.x))
        else:
            # train is the complement of test
            self.x = np.delete(self.x, self.indices, axis = 0)
            self.y = np.delete(self.y, self.indices, axis = 0)
            print("len of train set", len(self.x))



        #print("mean label",np.mean(self.y)) #0.30677558865929844
        

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

        # zero_mask = genome == 0
        # torch.save(zero_mask, "zero_mask.pt")
        
        return genome.float(), label

def GeneticDataSets( processed = True, label = None, percent_unlabeled = percent_unlabeled):
    # dataset = GeneticDataset(processed, label = label, percent_unlabeled = percent_unlabeled)
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # generator1 = torch.Generator().manual_seed(42)
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator = generator1)
    return GeneticDataset(processed, label = label, percent_unlabeled = percent_unlabeled, train=True), GeneticDataset(processed, label = label, percent_unlabeled = percent_unlabeled, train = False)

def GeneticDataloaders(batchsize, processed = True, percent_unlabeled = percent_unlabeled):
    train_dataset,test_dataset = GeneticDataSets(processed, percent_unlabeled = percent_unlabeled)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize,
                                           shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize,
                                          shuffle=True)
    return train_dataloader,test_dataloader




#processes_data()