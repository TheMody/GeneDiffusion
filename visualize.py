

import umap
import matplotlib.pyplot as plt
from dataloader import GeneticDataloaders, SynGeneticDataset
from torch.utils.data import DataLoader
from config import *
import numpy as np
import sklearn
num_samples = 200

def plot_histogramm(x):
    plt.bar(np.arange(len(x)), x, width = 1)
    plt.show()

if __name__ == '__main__':
    train_dataloader,val_dataloader = GeneticDataloaders(config["batch_size"], True)

    geneticData = SynGeneticDataset("syn_data_PosSensitive/")
    syn_dataloader2 = DataLoader(geneticData, batch_size=config["batch_size"])

    geneticData = SynGeneticDataset("UnetMLP_supergood/")
    syn_dataloader = DataLoader(geneticData, batch_size=config["batch_size"])

    def draw_samples(dataloader,num_samples):
        data = []
        data_labels = []
        for i, (x, y) in enumerate(dataloader):
            for d in x:
                data.append(d)
            for l in y:
                data_labels.append(l)
            if len(data) >= num_samples:
                break

        data = torch.stack(data)
        data_labels = torch.stack(data_labels)

        print(data_labels.shape)
        out_label = np.argmax(data_labels.numpy(), axis = 1) if len(data_labels.shape) > 1 else data_labels.numpy()
        return data.numpy(), out_label

    syn_data, syn_data_labels = draw_samples(syn_dataloader, num_samples)
    syn_data2, syn_data_labels2 = draw_samples(syn_dataloader2, num_samples)
    train_data, train_data_labels = draw_samples(train_dataloader, num_samples)
    val_data, val_data_labels = draw_samples(val_dataloader, num_samples)
   # syn_data = syn_data#.transpose(0,2,1)
    print(syn_data_labels.shape)
    print(train_data_labels.shape)
    print(val_data_labels.shape)

    combined_data = np.concatenate([syn_data,syn_data2, train_data, val_data], axis=0)
    combined_data = combined_data.reshape(combined_data.shape[0], -1)
    print(combined_data.shape)

 #   combined_labels = np.concatenate([syn_data_labels,syn_data_labels2 +6, train_data_labels+2, val_data_labels+4])
    combined_labels = np.concatenate([np.zeros(len(syn_data_labels)),np.zeros(len(syn_data_labels2))+3, np.zeros(len(train_data_labels))+1, np.zeros(len(val_data_labels))+2])
  #  print(combined_labels)
    #preprocess with PCA
   # print("computing PCA")
    # pca = sklearn.decomposition.PCA(n_components=50)
    # combined_data = pca.fit_transform(combined_data)

    print("computing UMAP")
    reducer = umap.UMAP(metric = 'cosine', n_neighbors=5)
    embedding = reducer.fit_transform(combined_data)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=combined_labels, cmap=plt.get_cmap('brg'))

   # Create a color bar
    cbar = plt.colorbar()
    cbar.set_label('Data source', rotation=270)

   # plt.legend()

    # Show the plot
    plt.show()
    #plt legend
    # plt.legend(["Synthetic Data 0","Synthetic Data 1", "Train Data 0","Train Data 1", "Validation Data 0", "Validation Data 1"], title = "Data Type", loc = "best")
    # plt.show()