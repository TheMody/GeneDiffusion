import torch
from utils.model import MLPModel, ConvclsModel, EncoderModel, EncoderModelPreTrain
from utils.dataloader import  GeneticDataloaders, SynGeneticDataset, GeneticDataSets, GeneticDataset
import numpy as np
import wandb
from utils.cosine_scheduler import CosineWarmupScheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from ALS.config import *
import time



def train_classifier(model = "mlp", data = "syn", path = save_path+"/"):
    print("Training Classifier using ", model, "on " , data)
    #basic building blocks
    if model == "mlp":
      model = MLPModel(num_input=num_channels*gene_size)#75584)#
    elif model == "cnn":
        model = ConvclsModel(input_dim=num_channels)
    elif model == "transformer":
        model = EncoderModel()
        
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_classifier)
    loss_fn = torch.nn.CrossEntropyLoss()
    wandb.init(project="diffusionGene", config=config)
    #data
    
    if data == "syn":
      geneticDataSyn = SynGeneticDataset(path = path)#path = "syndaUnetconv/")path = "UnetMLP_supergood/"path = "syn_data_Transformer/"path = "syn_data_Transformer/"
      train_dataloader = DataLoader(geneticDataSyn, batch_size=config["batch_size"], shuffle=True)#//2 for half syn
      _,test_dataloader = GeneticDataloaders(config["batch_size"], True, percent_unlabeled=0)#//2
    else:
       train_dataloader,test_dataloader = GeneticDataloaders(config["batch_size"], True, percent_unlabeled=0) 
    
    max_step = (num_of_samples-test_set_size)/(batch_size*gradient_accumulation_steps)*epochs_classifier
    scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=max_step)#len(train_dataloader)*epochs_classifier//gradient_accumulation_steps)

    running_loss = 0.0
    best_acc = 0.0
    step = 0
    for epoch in range(epochs_classifier):
        dataloader_iter = iter(train_dataloader)
        #dataloader_iter2 = iter(train_2)
        if step >= max_step-2:
            break
        for i in range(len(train_dataloader) // gradient_accumulation_steps):
            start = time.time()
            optimizer.zero_grad()
            accloss = 0.0
            accacc = 0.0
            for micro_step in range(gradient_accumulation_steps):
                inputs, labels = next(dataloader_iter)
                # try:
                #     inputs2, labels2 = next(dataloader_iter2)
                # except StopIteration:
                #     dataloader_iter2 = iter(train_2)
                #     inputs2, labels2 = next(dataloader_iter2)
                # inputs = torch.cat((inputs,inputs2),0)
                # labels = torch.cat((labels,labels2),0)

                inputs = inputs.float().to(device)

                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                with torch.no_grad():
                    accacc += (torch.sum(torch.argmax(outputs, axis = 1) ==labels)/labels.shape[0]).item()
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
            if step % log_freq == 0:
                #   acc = np.sum(np.argmax(outputs.detach().cpu().numpy(), axis = 1) == labels.detach().cpu().numpy())/labels.shape[0]
                avg_loss = running_loss / log_freq # loss per batch
                log_dict = {"avg_loss_classifier": avg_loss, "accuracy_classifier": acc, "lr_classifier": scheduler.get_lr()[0], "time_per_step": time.time()-start}
                wandb.log(log_dict)
                running_loss = 0.
                if step % 100 == 0:
                    print('  batch {} loss: {} accuracy: {}'.format(i + 1, avg_loss, acc))
                    

            # #evaluate model after epoch
            if step % 500 == 0:
                with torch.no_grad():
                    accummulated_acc = 0.0
                    accummulated_loss = 0.0
                    for i, data in enumerate(test_dataloader):
                        inputs, labels = data
                        inputs = inputs.float().to(device)
                        #inputs = preprocessing_function(inputs)
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
                # torch.save(model,"classification_models/model.pt")
                    print(f"saved new best model with acc {best_acc}")


if __name__ == "__main__":
    # train_classifier("mlp")
    # train_classifier("cnn")
    # train_classifier("transformer")
    train_classifier("mlp", path = "syn_data_PosSensitive/", data = "syn")
