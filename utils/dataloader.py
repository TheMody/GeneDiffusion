import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import os
from ALS.config import *

def load_data(processed = True):
    print("Loading data...")
    dataset_X,Y=pickle.load(open('data/sigSNPs_pca.features.pkl','rb'))
    print(dataset_X)
    print(Y)
    if processed:
        dataset_X = pickle.load(open("data/processed_ds","rb"))
    print("Data loaded!")

    return np.asarray(dataset_X), np.asarray(Y)

def load_data_1k(processed = True):
    print("Loading data...")
    dataset_X = pickle.load(open("data/processed_ds_genepca","rb"))
    print("Data loaded!")
    y = pickle.load(open("data/y_genepca","rb"))
    return np.asarray(dataset_X), np.asarray(y)

def processes_data_1k():
    ds,Y=pickle.load(open('data/gene_pca.features.pkl','rb'))
    names = list(ds.columns.values)
    print(names)
    for i in range(len(names)):
        split  = names[i].split(":")
        names[i] = split[0]+":"+split[1]
   # print(ds)
   # print(Y)
   # for y in Y:
     #   print(y)

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
    fileObject = open("data/processed_ds_genepca", 'wb')
    pickle.dump(tokenized_ds,fileObject )
    fileObject.close()

def processes_data_1k_labels():

    #now load the y data
    with open('genepca/integrated_call_samples_v3.20130502.ALL.panel') as f:
        lines = f.readlines()[1:]
        #seperate lines by tab
        lines = [line.split("\t") for line in lines]
        #get the super population
        y = [line[1] for line in lines]

    y = np.asarray(y)
    y_unique = np.unique(y)
    print("number of unique labels",len(y_unique))
    new_y = np.zeros(len(y))
    for i, yu in enumerate(y_unique):
        new_y[y==yu] = i
   # for i in range(len(y)):
     #   print(y[i], new_y[i])
    #save the y data
    fileObject = open("data/y_genepca", 'wb')
    pickle.dump(new_y.astype(np.int64),fileObject )
    fileObject.close()


def generate_train_test_split_1k(processed = True, normalize = normalize_data):
    x,y = load_data_1k(processed = processed)

    print("len of dataset", len(x))
    
    processed = processed
    if normalize:
        xstd = np.std(x, axis = 0)
        xstd[xstd == 0.0] +=1
        xmean = np.mean(x,axis=0)
        x = x-xmean
        x = x / xstd 
        fileObject = open("data/normlization_1k.pkl", 'wb')
        pickle.dump((xmean,xstd),fileObject )
        fileObject.close()

    

    # if percent_unlabeled != 0:
    #     y[:int(len(self.y)*percent_unlabeled)] = 2

    #now we shuffle x and y
    np.random.seed(42)
    p = np.random.permutation(len(x))
    x = x[p]
    y = y[p]

    x_train = x[:int(len(x)*0.9)]
    y_train = y[:int(len(x)*0.9)]
    x_test = x[int(len(x)*0.9):]
    y_test = y[int(len(x)*0.9):]
    print("len of test set", len(x_test))
    print("len of train set", len(x_train))

    fileObject = open("data/ds_test_1k", 'wb')
    pickle.dump((x_test,y_test),fileObject )
    fileObject.close()

    fileObject = open("data/ds_train_1k", 'wb')
    pickle.dump((x_train,y_train),fileObject )
    fileObject.close()


class GeneticDataset1k(Dataset):
    def __init__(self, processed = True, normalize = normalize_data, label = None, percent_unlabeled = percent_unlabeled, train = True):
        if not os.path.exists('data/ds_train_1k'):
            print("generating new train/test split")
            generate_train_test_split_1k(processed = processed, normalize = normalize)
        if train:
            self.x,self.y=pickle.load(open('data/ds_train_1k','rb'))
        #    self.x = self.x[:int(len(self.x)*0.05)]
         #   self.y = self.y[:int(len(self.y)*0.05)]
        else:
            self.x,self.y=pickle.load(open('data/ds_test_1k','rb'))
        
        print("len of dataset", len(self.x))

        

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        genome = self.x[idx]
      #  genome = genome[None,...]
       # if self.processed:
        genome = F.pad(torch.tensor(genome), (0,0,0, 26624 - genome.shape[0]), "constant", 0)
     #   print(genome.shape)
        label = torch.tensor(self.y[idx]).long()

        # print("calculating zero maks")
        # zero_mask = genome == 0
        # print( zero_mask)
        # torch.save(zero_mask, "data/zero_mask1k.pt")
        return genome.float(), label

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
    fileObject = open("data/processed_ds_new", 'wb')
    pickle.dump(tokenized_ds,fileObject )
    fileObject.close()
    # for column in dataset_X:
    #     print(column)


class SynGeneticDataset(Dataset):
    def __init__(self, path = save_path+"/", label = None):
        self.label = label
        self.path = path
        self.all_file_paths = [self.path + file for file in os.listdir(self.path) if file != "model.pt" and file != "vaemodel.pt" and file[-3:] == ".pt"]
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
    
def save_datameanstd():
    x,y = load_data(processed = True)

    print("len of dataset", len(x))
    
    xstd = np.std(x, axis = 0)
    xstd[xstd == 0.0] +=1
    xmean = np.mean(x,axis=0)
    x = x-xmean
    x = x / xstd 
    fileObject = open("data/normlization.pkl", 'wb')
    pickle.dump((xmean,xstd),fileObject )
    fileObject.close()

def transform_data_back(x, save= "ds/backtransformed_ds"):

    #now undo one kind of padding
    x = x[:,:18279,:]

    #first undo normlization
    fileObject = open("data/normlization.pkl", 'rb')
    xmean,xstd = pickle.load(fileObject)
    fileObject.close()
    x = x * xstd + xmean


    #now transform back to weird gene snps format
    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])


    ds,Y=pickle.load(open('data/sigSNPs_pca.features.pkl','rb'))

    names = list(ds.columns.values)
    
    #undo all the padding
    # zero_mask2 = zero_mask[:18279,:].flatten()
    # x = x[:,zero_mask2==0]
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
    for i in tqdm(range(1)):
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
            token[pos_int_token] = 1
            pos_int_token += 1
        tokens.append(token)
       # print(len(tokens))
        datapoint = np.asarray(tokens)
      #  print(datapoint.shape)
        #print(datapoint.shape)
        tokenized_ds.append(datapoint)

    tokenized_ds = np.asarray(tokenized_ds)
    print(tokenized_ds.shape)

    tokenized_ds = tokenized_ds[0].flatten()
    x = x[:,tokenized_ds==1]

    # now it should be the same lenght as the original dataset
    print(len(names), x.shape[1])
    print(x[-1])
    print(np.asanyarray(ds.iloc[len(ds)-1,:].values))

    #save the dataset
    fileObject = open(save, 'wb')
    pickle.dump(x,fileObject )
    fileObject.close()

    return x

def generate_train_test_split(processed = True, normalize = normalize_data):
    x,y = load_data(processed = processed)

    print("len of dataset", len(x))
    
    processed = processed


    

    # if percent_unlabeled != 0:
    #     y[:int(len(self.y)*percent_unlabeled)] = 2

    #now we shuffle x and y
    np.random.seed(42)
    p = np.random.permutation(len(x))
    x = x[p]
    y = y[p]


    #now we split the data into train and test but balance the test set by taking the same amount of each class
    # we take 500 test samples of each class
    num_test_samples = test_set_size//2

    length_test_p = 0
    length_test_n = 0
    indices= []
    for i,c in enumerate(y):
        if c == 0:
            if length_test_n >= num_test_samples:
                continue
            indices.append(i)
            length_test_n += 1
        else:
            if length_test_p >= num_test_samples:
                continue
            indices.append(i)
            length_test_p += 1
        if length_test_n >= num_test_samples and length_test_p >= num_test_samples:
            break


    x_test = x[indices]
    y_test = y[indices]
    print("len of test set", len(x_test))

    # train is the complement of test
    x_train = np.delete(x, indices, axis = 0)
    y_train = np.delete(y, indices, axis = 0)
    print("len of train set", len(x_train))

    if normalize:
        xstd = np.std(x_train, axis = 0)
        xstd[xstd == 0.0] +=1
        xmean = np.mean(x_train,axis=0)
        x_train = x_train-xmean
        x_train = x_train / xstd 
        x_test = x_test - xmean
        x_test = x_test / xstd
        fileObject = open("data/normlization.pkl", 'wb')
        pickle.dump((xmean,xstd),fileObject )
        fileObject.close()


    fileObject = open("data/ds_test", 'wb')
    pickle.dump((x_test,y_test),fileObject )
    fileObject.close()

    fileObject = open("data/ds_train", 'wb')
    pickle.dump((x_train,y_train),fileObject )
    fileObject.close()


class GeneticDataset(Dataset):
    def __init__(self, processed = True, normalize = normalize_data, label = None, percent_unlabeled = percent_unlabeled, train = True):
        if not os.path.exists('data/ds_train'):
            print("generating new train/test split")
            generate_train_test_split(processed = processed, normalize = normalize)
        if train:
            self.x,self.y=pickle.load(open('data/ds_train','rb'))
        else:
            self.x,self.y=pickle.load(open('data/ds_test','rb'))

        # if train:
        #     self.x = self.x[:int(len(self.x)/20)]
        print("len of dataset", len(self.x))
        #super(GeneticDataset, self).__init__()
        # self.x,self.y = load_data(processed = processed)

        # print("len of dataset", len(self.x))
        
        # self.processed = processed
        # self.label = label
        # if normalize:
        #     xstd = np.std(self.x, axis = 0)
        #     xstd[xstd == 0.0] +=1
        #     self.x-np.mean(self.x,axis=0)
        #     self.x = self.x / xstd 
        

        # if percent_unlabeled != 0:
        #     self.y[:int(len(self.y)*percent_unlabeled)] = 2

        # #now we shuffle x and y
        # np.random.seed(42)
        # p = np.random.permutation(len(self.x))
        # self.x = self.x[p]
        # self.y = self.y[p]


        # #now we split the data into train and test but balance the test set by taking the same amount of each class
        # # we take 500 test samples of each class
        # num_test_samples = test_set_size//2

        # length_test_p = 0
        # length_test_n = 0
        # self.indices= []
        # for i,y in enumerate(self.y):
        #     if y == 0:
        #         if length_test_n >= num_test_samples:
        #             continue
        #         self.indices.append(i)
        #         length_test_n += 1
        #     else:
        #         if length_test_p >= num_test_samples:
        #             continue
        #         self.indices.append(i)
        #         length_test_p += 1
        #     if length_test_n >= num_test_samples and length_test_p >= num_test_samples:
        #         break

        # if not train:
        #     self.x = self.x[self.indices]
        #     self.y = self.y[self.indices]
        #     print("len of test set", len(self.x))
        # else:
        #     # train is the complement of test
        #     self.x = np.delete(self.x, self.indices, axis = 0)
        #     self.y = np.delete(self.y, self.indices, axis = 0)
        #     print("len of train set", len(self.x))



        #print("mean label",np.mean(self.y)) #0.30677558865929844
        

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        genome = self.x[idx]
      #  genome = genome[None,...]
       # if self.processed:
        genome = F.pad(torch.tensor(genome), (0,0,0, 18432 - genome.shape[0]), "constant", 0)
     #   print(genome.shape)
        label = torch.tensor(self.y[idx])
      #  if self.label is not None:
        #label = torch.tensor(self.label)

        # zero_mask = genome == 0
        # torch.save(zero_mask, "zero_mask.pt")
        
        return genome.float(), label

def GeneticDataSets( processed = True, label = None, percent_unlabeled = percent_unlabeled):
    # dataset = GeneticDataset(processed, label = label, percent_unlabeled = percent_unlabeled)
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # generator1 = torch.Generator().manual_seed(42)
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator = generator1)
    # return train_dataset, test_dataset
    return GeneticDataset(processed, label = label, percent_unlabeled = percent_unlabeled, train=True), GeneticDataset(processed, label = label, percent_unlabeled = percent_unlabeled, train = False)

def GeneticDataloaders(batchsize, processed = True, percent_unlabeled = percent_unlabeled):
    train_dataset,test_dataset = GeneticDataSets(processed, percent_unlabeled = percent_unlabeled)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize,
                                           shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize,
                                          shuffle=True)
    return train_dataloader,test_dataloader

def GeneticDataloaders1k(batchsize):
    train_dataset = GeneticDataset1k(train = True)
    test_dataset = GeneticDataset1k(train = False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize,
                                           shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize,
                                          shuffle=True)
    return train_dataloader,test_dataloader


if __name__ == "__main__":
    # for 1kG data
    # first prepare the raw .vcf data with https://github.com/HaploKit/DiseaseCapsule/blob/master/data_preprocessing/gene_pca.py
    # and copy the gene_pca.features.pkl file to the data folder
   # transform_data_back(np.zeros((1,26624,8)))
    processes_data_1k() #only needed once
    processes_data_1k_labels() #only needed once
    GeneticDataloaders1k(32) # <-- data loader for pytorch

    # for the ALS data
    # processes_data() #only needed once
    # save_datameanstd() #needed for backtransformation #only needed once
    #GeneticDataloaders(32) # <-- data loader for pytorch