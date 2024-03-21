import torch
from model import MLPModel
from dataloader import  GeneticDataloaders, SynGeneticDataset, GeneticDataset
import numpy as np
import wandb
from cosine_scheduler import CosineWarmupScheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import *


num_checks = 100
def draw_samples(dataloader,num_samples):
        data = []
        data_labels = []
        for i, (x, y) in enumerate(dataloader):
           # for d in x:
            data.append(x)
          #  for l in y:
            data_labels.append(y)
            if len(data) >= num_samples:
                break

        data = torch.stack(data)
        data_labels = torch.stack(data_labels)

        print(data_labels.shape)
       # out_label = np.argmax(data_labels.numpy(), axis = 1) if len(data_labels.shape) > 1 else data_labels.numpy()
        return data

def check_diversity_by_closest():
    geneticDataSyn = SynGeneticDataset("newbestdata/")
    geneticData = GeneticDataset()


    geneticDataSyn = draw_samples(geneticDataSyn, num_checks)
    geneticData = draw_samples(geneticData, num_checks)

    def find_closest(dataset, num = 10):
        all_min_dist = []
        for i,datapoint in enumerate(dataset):
            if i > num:
                break
            sample = datapoint
            #find closest sample in geneticData
            min_dist = 1e10
            for sample2 in geneticData:
                dist = torch.sum(torch.abs(sample-sample2))
                if dist < min_dist and dist > 0:
                    min_dist = dist
            all_min_dist.append(min_dist)
            print(min_dist)
        print(np.mean(all_min_dist))

    find_closest(geneticDataSyn)
    #find_closest(geneticData)
            
if __name__ == "__main__":
    check_diversity_by_closest()