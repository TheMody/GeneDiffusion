import torch
import torch.nn.functional as F
from model import MLPModel, ConvclsModel
from dataloader import  GeneticDataloaders, SynGeneticDataset, GeneticDataSets, GeneticDataset
import numpy as np
import wandb
from cosine_scheduler import CosineWarmupScheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import *


class LinearBottleneckModel(torch.nn.Module):
  def __init__(self, bottleneck_dims = 256) -> None:
      super().__init__()
      self.diffmodel = torch.load(save_path+"/"+"model.pt")#torch.load("model.pt")#
    #   for param in self.diffmodel.parameters():
    #     param.requires_grad = False
      self.linear = torch.nn.Linear(bottleneck_dims, num_classes)

  def forward(self, x):
      x = x.permute(0,2,1)
      y = torch.zeros( (x.shape[0],), dtype=torch.int64).to(device) +num_classes
      t = torch.zeros( (x.shape[0],), dtype=torch.int64).to(device)
      x,bottleneck = self.diffmodel(x, t,y, output_bottleneck=True)
    #  print(bottleneck.shape)
      return self.linear(bottleneck.flatten(1))


def train_classifier():
    #basic building blocks
    model = LinearBottleneckModel()
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_classifier)
    loss_fn = torch.nn.CrossEntropyLoss()
    wandb.init(project="diffusionGene", config=config)
    train_dataloader,test_dataloader = GeneticDataloaders(config["batch_size"], True, percent_unlabeled=0)
    max_step = 10000/16*5
    scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=max_step)#len(train_dataloader)*epochs_classifier//gradient_accumulation_steps)

    running_loss = 0.0
    best_acc = 0.0
    step = 0
    for epoch in range(epochs_classifier):
        dataloader_iter = iter(train_dataloader)
        if step > max_step:
            break
        for i in range(len(train_dataloader) // gradient_accumulation_steps):
                optimizer.zero_grad()
                accloss = 0.0
                accacc = 0.0
                for micro_step in range(gradient_accumulation_steps):
                    inputs, labels = next(dataloader_iter)
                    inputs = inputs.float().to(device)

                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    with torch.no_grad():
                        accacc += torch.sum(torch.argmax(outputs, axis = 1) ==labels)/labels.shape[0]
                        accloss += loss.item()

                    loss.backward()
                step += 1
                acc = accacc/gradient_accumulation_steps

                # Adjust learning weights
                optimizer.step()
                scheduler.step()

                # Gather data and report
                log_freq = 1
                running_loss += accloss
                if i % log_freq == 0:
                    avg_loss = running_loss / log_freq # loss per batch
                    log_dict = {"avg_loss_classifier": avg_loss, "accuracy_classifier": acc, "lr_classifier": scheduler.get_lr()[0]}
                    wandb.log(log_dict)
                    running_loss = 0.
                    if i % 20 == 0:
                        print('  batch {} loss: {} accuracy: {}'.format(i + 1, avg_loss, acc))
                        

        # #evaluate model after epoch
        with torch.no_grad():
            accummulated_acc = 0.0
            accummulated_loss = 0.0
            for i, data in enumerate(test_dataloader):
                inputs, labels = data
              #  print(inputs.shape)
                inputs = inputs.float().to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                acc = torch.sum(torch.argmax(outputs, axis = 1) == labels)/labels.shape[0]
                accummulated_acc += acc.item()
                accummulated_loss += loss.item()
            avg_acc = accummulated_acc/len(test_dataloader)
            avg_loss = accummulated_loss/len(test_dataloader)
            wandb.log({"test_loss_classifier": avg_loss, "test_accuracy_classifier": avg_acc})
            print(' test epoch {} loss: {} accuracy: {}'.format(epoch+1, avg_loss, avg_acc))
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model,"classification_models/model.pt")
            print(f"saved new best model with acc {best_acc}")


if __name__ == "__main__":
    train_classifier()