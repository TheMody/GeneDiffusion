import torch
from utils.diffusion_Process import GuassianDiffusion
from KG1.config_1k import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def generate_sample(model,num_samples = num_of_samples, save = True, savefolder = save_path):
   diffusion = GuassianDiffusion(timesteps=max_steps)
   diffusion.zero_mask = torch.tensor(zero_mask).permute(1,0)
   for i in range(num_samples//config["batch_size"]):
      xt = torch.randn_like(torch.zeros(config["batch_size"],num_channels,gene_size)).to(device)
    #  label = (torch.rand(config["batch_size"]) <  label_proportion_for_generation).long().to(device)
      label = torch.randint(num_classes, (config["batch_size"],), dtype=torch.int64).to(device)
      print("at step:",i, "generating samples with label" ,label)
      sample = diffusion.sample_from_reverse_process(model,xt, timesteps=max_steps,y= label, guidance = "normal",ddim=False, w = 0.1)
      if save:
         for a in range(config["batch_size"]):
            torch.save((sample[a].cpu().detach(),  label[a].cpu().detach()), savefolder +"/sample"+str(i*config["batch_size"] + a)+".pt")
   return sample



if __name__ == '__main__':
   model = torch.load(save_path+"/"+"model.pt").to(device)
   model.eval()
   generate_sample(model, num_of_samples, savefolder=save_path)


 


 
