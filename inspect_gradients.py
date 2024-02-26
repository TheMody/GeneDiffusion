import torch
from model import MLPModel
from dataloader import  GeneticDataloaders, SynGeneticDataset, GeneticDataSets, GeneticDataset
import numpy as np
import wandb
from cosine_scheduler import CosineWarmupScheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import *
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_classifier():
    #basic building blocks

    model = torch.load("classification_models/model.pt")
    for param in model.parameters():
      param.requires_grad = False
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    train_dataloader,test_dataloader = GeneticDataloaders(config["batch_size"], True)


    dataloader_iter = iter(train_dataloader)
    avg_inputs_grad = torch.zeros( (18432, 8)).to(device)
    for i in tqdm(range(len(train_dataloader))):
        inputs, labels = next(dataloader_iter)
        inputs = inputs.float().to(device)
        inputs.requires_grad = True
        labels = labels.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        test  = torch.mean(torch.abs(inputs.grad),axis = 0)

        avg_inputs_grad += test
    avg_inputs_grad = avg_inputs_grad / len(train_dataloader)
    
    test = avg_inputs_grad.flatten()
    test, ind = torch.sort(test, descending=True )
    plt.plot(test.cpu().numpy())
    plt.show()

        
            



if __name__ == "__main__":
    train_classifier()