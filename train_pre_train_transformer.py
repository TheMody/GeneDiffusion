import torch
from model import MLPModel, ConvclsModel, EncoderModel, EncoderModelPreTrain
from dataloader import  GeneticDataloaders, SynGeneticDataset, GeneticDataSets, GeneticDataset
import numpy as np
import wandb
from cosine_scheduler import CosineWarmupScheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import *
import time

#input is shape (batch, gene_size, num_channels)
def mask_input(mask_ratio, input, mask_token):
    mask = torch.rand(input.shape[:2]) < mask_ratio
    mask = mask.to(device)
    input = torch.where(mask.unsqueeze(-1), mask_token*torch.ones_like(input), input)
    return input, mask

def mse_loss_masked(input, target, mask):
    loss = (input - target)**2
    loss = loss*mask.unsqueeze(-1)
    return torch.sum(loss)/torch.sum(mask)

def mae_loss_masked(input, target, mask):
    loss = (input - target).abs()
    loss = loss*mask.unsqueeze(-1)
    return torch.sum(loss)/torch.sum(mask)

def train_pre_train():
    #basic building blocks
    model = EncoderModelPreTrain()
    save_model = model.to(device)
    model = torch.compile(save_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_pretrain)
    #loss_fn = torch.nn.CrossEntropyLoss()
    wandb.init(project="diffusionGene", config=config)
    #data
    train_dataloader,test_dataloader = GeneticDataloaders(config["batch_size"], True, percent_unlabeled=0)
    max_step = epochs * len(train_dataloader) // gradient_accumulation_steps
    scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=max_step)#len(train_dataloader)*epochs_classifier//gradient_accumulation_steps)

    running_loss = 0.0
    best_loss = 1e10
    step = 0
    ema_time = 0.0
    log_freq = 1
    for epoch in range(epochs_classifier):
        dataloader_iter = iter(train_dataloader)
        if step >= max_step-1:
            break
        for i in range(len(train_dataloader) // gradient_accumulation_steps):
                start = time.time()
                optimizer.zero_grad()
                accloss = 0.0
                for micro_step in range(gradient_accumulation_steps):
                    inputs, _ = next(dataloader_iter)
                    inputs = inputs.float().to(device)

                    inputs, mask = mask_input(mask_ratio, inputs, torch.zeros(inputs.shape[2]).to(device))
                    outputs = model(inputs, last_hidden = True)

                   # loss = loss_fn(outputs, labels)
                    loss = mae_loss_masked(outputs, inputs, mask)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    with torch.no_grad():
                     #   accacc += (torch.sum(torch.argmax(outputs, axis = 1) ==labels)/labels.shape[0]).item()
                        accloss += loss.item()

                    loss.backward()
                step += 1
              #  acc = accacc/gradient_accumulation_steps
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                # Adjust learning weights
                optimizer.step()
                scheduler.step()

                # Gather data and report
                running_loss += accloss
                time_taken = time.time() - start
                ema_time = ema_time * 0.99 + time_taken * 0.01 #exponential moving average
                ema_time_corrected = ema_time / (1 - 0.99 ** (step + 1 + epoch * len(train_dataloader)//gradient_accumulation_steps))#bias corrected ema
                remaining_time = int((max_step - step) * ema_time_corrected)
                print(f'estimated remaining time {remaining_time:3d} sec at step {step}/{(len(train_dataloader)//gradient_accumulation_steps)*epochs_classifier} steps', end='\r')

                if i % log_freq == 0:
                    avg_loss = running_loss / log_freq # loss per batch
                    log_dict = {"avg_loss_classifier": avg_loss,  "lr_classifier": scheduler.get_lr()[0], "time_per_step": time_taken}
                    wandb.log(log_dict)
                    running_loss = 0.
                    if i % 20 == 0:
                        print('  batch {} loss: {} '.format(i + 1, avg_loss))

                        

        # #evaluate model after epoch
        with torch.no_grad():
            accummulated_loss = 0.0
            for i, data in enumerate(test_dataloader):
                inputs, _ = data
              #  print(inputs.shape)
                inputs = inputs.float().to(device)
                inputs, mask = mask_input(mask_ratio, inputs, torch.zeros(inputs.shape[2]).to(device))
              #  print(inputs[0])
                #inputs = preprocessing_function(inputs)
                outputs = model(inputs, last_hidden = True)

                   # loss = loss_fn(outputs, labels)
                loss = mae_loss_masked(outputs, inputs, mask)
                #acc = torch.sum(torch.argmax(outputs, axis = 1) == labels)/labels.shape[0]
               # accummulated_acc += acc.item()
                accummulated_loss += loss.item()
          #  avg_acc = accummulated_acc/len(test_dataloader)
            avg_loss = accummulated_loss/len(test_dataloader)
            wandb.log({"test_loss_classifier": avg_loss})
            print(' test epoch {} loss: {} '.format(epoch+1, avg_loss))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(save_model,"pretrained_models/model.pt")
            print(f"saved new best model with best_loss {best_loss}")

    return torch.load("pretrained_models/model.pt")


if __name__ == "__main__":
    model = train_pre_train()
    from train_classifier import train_classifier
    train_classifier(model)
