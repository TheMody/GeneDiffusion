import torch
from model import MLPModel
from dataloader import  GeneticDataloaders, SynGeneticDataset, GeneticDataSets
import numpy as np
import wandb
from cosine_scheduler import CosineWarmupScheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import *

def train_adversarial():
    #basic building blocks
    model = MLPModel(num_input=num_channels*gene_size)#75584)#

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_classifier)
    loss_fn = torch.nn.CrossEntropyLoss()
    wandb.init(project="disease_prediction_syn", config=config)
    #data
    geneticDataSyn = SynGeneticDataset(label = 0)
    train_size = int(0.8 * len(geneticDataSyn))
    test_size = len(geneticDataSyn) - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_dataset_syn, test_dataset_syn = torch.utils.data.random_split(geneticDataSyn, [train_size, test_size], generator = generator1)

    geneticDatatrain,geneticDatatest = GeneticDataSets(label = 1)

    train_dataloader = DataLoader(torch.utils.data.ConcatDataset([geneticDatatrain, train_dataset_syn]), batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(torch.utils.data.ConcatDataset([geneticDatatest, test_dataset_syn]), batch_size=config["batch_size"], shuffle=False)
   # train_dataloader = DataLoader(geneticDataSyn, batch_size=config["batch_size"], shuffle=True)
    
    #_,test_dataloader = GeneticDataloaders(config["batch_size"], True)

    scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=len(train_dataloader)*epochs_classifier//gradient_accumulation_steps)

  #  encoder_model = torch.load( save_path+"/"+"vaemodel.pt")
    running_loss = 0.0
    best_acc = 0.0
    for epoch in range(epochs_classifier):
        dataloader_iter = iter(train_dataloader)
        for i in range(len(train_dataloader) // gradient_accumulation_steps):
                optimizer.zero_grad()
                accloss = 0.0
                accacc = 0.0
                for micro_step in range(gradient_accumulation_steps):
                    inputs, labels = next(dataloader_iter)
                    inputs = inputs.float().to(device)
                   # r_inputs = encoder_model(inputs.permute(0,2,1), train = False).permute(0,2,1)
                    labels = labels.to(device)
                   # print(labels)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    with torch.no_grad():
                      #  accacc += torch.sum(torch.argmax(outputs, axis = 1) == torch.argmax(labels, axis = 1))/labels.shape[0]
                        accacc += torch.sum(torch.argmax(outputs, axis = 1) ==labels)/labels.shape[0]
                        accloss += loss.item()

                    loss.backward()
                    
                acc = accacc/gradient_accumulation_steps

                # Adjust learning weights
                optimizer.step()
                scheduler.step()

                # Gather data and report
                log_freq = 1
                running_loss += accloss
                if i % log_freq == 0:
                 #   acc = np.sum(np.argmax(outputs.detach().cpu().numpy(), axis = 1) == labels.detach().cpu().numpy())/labels.shape[0]
                    avg_loss = running_loss / log_freq # loss per batch
                    log_dict = {"avg_loss_classifier": avg_loss, "accuracy_classifier": acc, "lr_classifier": scheduler.get_lr()[0]}
                    wandb.log(log_dict)
                    running_loss = 0.
                    if i % 20 == 0:
                        print('  batch {} loss: {} accuracy: {}'.format(i + 1, avg_loss, acc))


        #evaluate model after epoch
        with torch.no_grad():
            accummulated_acc = 0.0
            accummulated_loss = 0.0
            for i, data in enumerate(test_dataloader):
                inputs, labels = data
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
           # torch.save(model.state_dict(), "classification_models/model"+ str(best_acc) +".pt")
          #  print(f"saved new best model with acc {best_acc}")


if __name__ == "__main__":
    train_adversarial()