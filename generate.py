import torch
from diffusion_Process import GuassianDiffusion
from config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def generate_sample(model,num_samples = 10000, save = True, savefolder = "syn_data"):
   diffusion = GuassianDiffusion(timesteps=max_steps)
   for i in range(0,num_samples//batch_size):
      xt = torch.randn_like(torch.zeros(batch_size,num_channels,gene_size)).to(device)
      label = torch.randint(num_classes, (batch_size,), dtype=torch.int64).to(device)
      sample = diffusion.sample_from_reverse_process(model,xt, timesteps=max_steps-1,y= label, guidance = "c_free", w = 0.1)
      if save:
         for a in range(batch_size):
            torch.save((sample[a].cpu().detach(),  label[a].cpu().detach()), savefolder +"/sample"+str(i*batch_size + a)+".pt")
   return sample

if __name__ == '__main__':
   model = torch.load("modelbestmodel.pt")
   model = model.to(device)
   model.eval()
   generate_sample(model, num_samples = 10000, save = True)

 


 