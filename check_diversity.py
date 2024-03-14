import torch
from model import MLPModel
from dataloader import  GeneticDataloaders, SynGeneticDataset, GeneticDataset
import numpy as np
import wandb
from cosine_scheduler import CosineWarmupScheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import *

def check_diversity_by_closest():
    geneticDataSyn = SynGeneticDataset("syn_data_PosSensitive/")
    geneticData = GeneticDataset()


    def find_closest(dataset, num = 10):
        all_min_dist = []
        for i,datapoint in enumerate(dataset):
            if i > num:
                break
            sample, label = datapoint
            #find closest sample in geneticData
            min_dist = 1e10
            for sample2, label2 in geneticData:
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