
from dataloader import GeneticDataset
import numpy as np
import torch
trainset = GeneticDataset(processed = True, train=True)
testset = GeneticDataset(processed = True, train=False)

for i,gene in enumerate(trainset):
   gene,_ = gene
   print(i)
   for geney in testset:
      geney,_ = geney
      if torch.equal(gene, geney):
         print("Error: Gene in trainset is in testset")
         break