import torch
from utils.model import MLPModel, ConvclsModel, EncoderModel, EncoderModelPreTrain
from utils.dataloader import  SynGeneticDataset, GeneticDataset1k
import numpy as np
import wandb
from utils.cosine_scheduler import CosineWarmupScheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from KG1.config_1k import *
from utils.caps_net import CapsNet
from salsa.SaLSA import SaLSA
import time



def train_classifier(model = "mlp", data = "syn", path = save_path+"/"):
    print("Training Classifier using ", model, "on " , data)
    #basic building blocks
    if model == "mlp":
      model = MLPModel(num_classes= num_classes,num_input=num_channels*gene_size)#75584)#
    elif model == "cnn":
        model = ConvclsModel(num_classes= num_classes,input_dim=num_channels, input_size = gene_size)
    elif model == "transformer":
        model = EncoderModel(num_classes= num_classes, input_size = gene_size)
    elif model == "capsule":
        model = CapsNet(input_dim = num_channels*gene_size, num_classes = num_classes)
        
    model = model.to(device)

  #  
    loss_fn = torch.nn.CrossEntropyLoss()
    wandb.init(project="diffusionGene1k", config=config)
    #data
    
    if data == "syn":
        geneticDataSyn = SynGeneticDataset(path = path)
        train_dataloader = DataLoader(geneticDataSyn, batch_size=config["batch_size"], shuffle=True)
      
    elif data == "real":
        train_dataset = GeneticDataset1k(train = True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    elif data == "combined":
        train_dataset = GeneticDataset1k(train = True)
        print("amount of real data", len(train_dataset))
        geneticDataSyn = SynGeneticDataset(path = path)
        print("amount of synthetic data", len(geneticDataSyn))
        concat_ds = torch.utils.data.ConcatDataset([train_dataset, geneticDataSyn])
        train_dataloader = DataLoader(concat_ds,batch_size=config["batch_size"], shuffle=True)
    test_dataset = GeneticDataset1k(train = False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
    
    max_step = (num_of_samples-test_set_size)/(batch_size*gradient_accumulation_steps)*epochs_classifier
    if optim == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_classifier)
        scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=max_step)#len(train_dataloader)*epochs_classifier//gradient_accumulation_steps)
    if optim == "salsa":
        optimizer = SaLSA(model.parameters(), use_mv=True)
    running_loss = 0.0
    best_acc = 0.0
    step = 0
    steps_since_last_eval = 0
    for epoch in range(epochs_classifier):
        dataloader_iter = iter(train_dataloader)
        if step >= max_step-2:
            break
        for i in range(len(train_dataloader) // gradient_accumulation_steps):
                steps_since_last_eval += 1
                start = time.time()
                inputs_list = []
                labels_list = []
                for micro_step in range(gradient_accumulation_steps):
                    inputs, labels = next(dataloader_iter)
                    inputs_list.append(inputs)
                    labels_list.append(labels)
                def closure(backwards = False):
                    accloss = 0.0
                    for micro_step in range(gradient_accumulation_steps):
                        inputs, labels = inputs_list[micro_step],labels_list[micro_step]
                        inputs = inputs.float().to(device)

                        labels = labels.to(device)
                        outputs = model(inputs)
                    #  print(outputs.shape, labels.shape)
                        loss = loss_fn(outputs, labels)
                        loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                        with torch.no_grad():
                        #    accacc += (torch.sum(torch.argmax(outputs, axis = 1) ==labels)/labels.shape[0]).item()
                            accloss += loss
                        if backwards:
                            loss.backward()

                    return accloss
                optimizer.zero_grad()
                step += 1
               # acc = accacc/gradient_accumulation_steps

                # Adjust learning weights
                if optim == "salsa":
                    loss = optimizer.step(closure = closure)
                if optim == "adam":
                    loss = closure(backwards=True)
                    optimizer.step()
                    scheduler.step()
             #   scheduler.step()
#
                # Gather data and report
                log_freq = 1
                running_loss += loss
                if i % log_freq == 0:
                 #   acc = np.sum(np.argmax(outputs.detach().cpu().numpy(), axis = 1) == labels.detach().cpu().numpy())/labels.shape[0]
                    avg_loss = running_loss / log_freq # loss per batch
                    lr = optimizer.state["lr"] if optim == "salsa" else scheduler.get_lr()[0]
                    log_dict = {"avg_loss_classifier": avg_loss,  "time_per_step": time.time()-start, "lr_classifier": lr}#"accuracy_classifier": acc, "lr_classifier": scheduler.get_lr()[0],
                    wandb.log(log_dict)
                    running_loss = 0.
                    if (i+1) % 25 == 0:
                        print('  batch {} loss: {} lr {}'.format(i + 1, avg_loss, lr))
                        

        # #evaluate model after epoch
        if steps_since_last_eval > 100: 
            steps_since_last_eval = 0
            with torch.no_grad():
                accummulated_acc = 0.0
                accummulated_loss = 0.0
                for i, data in enumerate(test_dataloader):
                    inputs, labels = data
                    inputs = inputs.float().to(device)
                    #inputs = preprocessing_function(inputs)
                    labels = labels.to(device)
                    outputs = model(inputs)
                  #  print(outputs.shape, labels.shape)
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
            #    torch.save(model,"classification_models/model.pt")
                print(f"saved new best model with acc {best_acc}")


if __name__ == "__main__":
   # train_classifier("mlp", path = "finalruns1k/UnetCombined/", data = "syn")
    train_classifier("mlp", path = "finalruns1k/UnetCombined/", data = "real")
   # 
   # train_classifier("cnn", path = "finalruns/UnetCombined", data = "real")
  #  train_classifier("transformer", path = "finalruns/UnetCombined", data = "real")
