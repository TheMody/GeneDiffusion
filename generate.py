import torch
from diffusion_Process import GuassianDiffusion
from unets import UNet
from data import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_steps = 500
gene_size = 18432
num_classes = 2
batch_size  = 8
num_channels = 8

@torch.no_grad()
def generate_sample(model,num_samples = 10000, save = True, savefolder = "syn_data"):
   diffusion = GuassianDiffusion(timesteps=max_steps)
   for i in range(0,num_samples):
      xt = torch.randn_like(torch.zeros(batch_size,num_channels,gene_size)).to(device)
      label = torch.randint(num_classes, (batch_size,), dtype=torch.int64).to(device)
      sample = diffusion.sample_from_reverse_process(model,xt, timesteps=max_steps-1,model_kwargs={"y":label})
      if save:
         #save label and sample as single file with pickle
         for a in range(batch_size):
            torch.save((sample[a].cpu().detach(),  label[a].cpu().detach()), savefolder +"/sample"+str(i*batch_size + a)+".pt")
            # with open("syn_data/sample"+str(i*batch_size + a)+".pkl", "wb") as f:
            #    pickle.dump((sample[a].cpu().detach(), label[a]), f)
   return sample

if __name__ == '__main__':
   # diffusion = diffusion_process(steps, img_size, img_size)
   
   model = torch.load("modelbestmodel.pt")
   model = model.to(device)
   model.eval()
   generate_sample(model, num_samples = 10000, save = True)
   # xt = torch.randn_like(torch.zeros(1,3,32,32)).to(device)
   # label = torch.LongTensor([8]).to(device)
   
   # sample = diffusion.sample_from_reverse_process(model,xt, timesteps=max_steps-1,model_kwargs={"y":label})

 


 