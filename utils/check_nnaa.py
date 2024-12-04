import torch
from utils.model import MLPModel
from utils.dataloader import  GeneticDataloaders, SynGeneticDataset, GeneticDataset
import numpy as np
import wandb
from utils.cosine_scheduler import CosineWarmupScheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from ALS.config import *

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


num_checks = 100
def check_diversity_by_closest():
    geneticDataSyn = SynGeneticDataset("newbestdata/")
    geneticData = GeneticDataset(train=True)

    geneticDataSyn = draw_samples(geneticDataSyn, num_checks)
    geneticData = draw_samples(geneticData, num_checks)

    def find_closest(dataset1, dataset2, num = num_checks):
        all_min_dist = []
        for i,datapoint in enumerate(dataset1):
            if i > num:
                break
            sample = datapoint.flatten()
            #find closest sample in geneticData
            min_dist = 1e10
            for sample2 in dataset2:
                dist = torch.arccos(torch.nn.functional.cosine_similarity(sample,sample2.flatten(), dim = 0)) / np.pi
                if dist < min_dist and dist > 1e-3:
                    min_dist = dist
            all_min_dist.append(min_dist)
            print(min_dist)
        
        return all_min_dist

    AA_ts = find_closest(geneticData, geneticDataSyn)
    AA_st = find_closest(geneticDataSyn, geneticData)
    AA_tt  = find_closest(geneticData, geneticData)
    AA_ss = find_closest(geneticDataSyn, geneticDataSyn)

    AA_truth = 0
    AA_syn = 0
    for i in range(num_checks):
        AA_truth += 1 if AA_ts[i]>AA_tt[i] else 0
        AA_syn += 1 if AA_st[i]>AA_ss[i] else 0
        print(AA_ts[i], AA_st[i], AA_tt[i], AA_ss[i])
    
    print(f"AA_truth: {AA_truth/num_checks}, AA_syn: {AA_syn/num_checks}")
    #find_closest(geneticData)
            
if __name__ == "__main__":
    check_diversity_by_closest()