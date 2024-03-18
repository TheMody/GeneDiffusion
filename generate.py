import torch
from diffusion_Process import GuassianDiffusion
from config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def generate_sample(model,num_samples = num_of_samples, save = True, savefolder = save_path):
   diffusion = GuassianDiffusion(timesteps=max_steps)
   for i in range(num_samples//config["batch_size"]):
      xt = torch.randn_like(torch.zeros(config["batch_size"],num_channels,gene_size)).to(device)
      label = (torch.rand(config["batch_size"]) <  label_proportion_for_generation).long().to(device)
      # label = torch.randint(num_classes, (config["batch_size"],), dtype=torch.int64).to(device)
      print("at timestep:",i, "generating samples with label" ,label)
      sample = diffusion.sample_from_reverse_process(model,xt, timesteps=max_steps,y= label, guidance = "normal",ddim=False, w = 0.1)
      if save:
         for a in range(config["batch_size"]):
            print(sample[a])
            torch.save((sample[a].cpu().detach(),  label[a].cpu().detach()), savefolder +"/sample"+str(i*config["batch_size"] + a)+".pt")
            sample1,label1 = torch.load(savefolder +"/sample"+str(i*config["batch_size"] + a)+".pt")
            print(sample1)
   return sample

@torch.no_grad()
def generate_sample_combined(model1, model2, num_steps_combine = 75,num_samples = num_of_samples, save = True, savefolder = save_path):
   diffusion = GuassianDiffusion(timesteps=max_steps)
   for i in range(num_samples//config["batch_size"]):
      xt = torch.randn_like(torch.zeros(config["batch_size"],num_channels,gene_size)).to(device)
      label = (torch.rand(config["batch_size"]) > 0.7).long().to(device)
      print(label)
      label = torch.randint(num_classes, (config["batch_size"],), dtype=torch.int64).to(device)
      print(label)
      print("at timestep:",i, "generating samples with label" ,label)
      sample = diffusion.sample_from_reverse_process(model1,xt, start_timesteps=max_steps-1, end_timestep=max_steps-1-num_steps_combine,y= label, guidance = "normal", w = 0.1)
      sample = diffusion.sample_from_reverse_process(model2,sample, start_timesteps=max_steps-1-num_steps_combine, y= label, guidance = "normal", w = 0.1)
      if save:
         for a in range(config["batch_size"]):
            torch.save((sample[a].cpu().detach(),  label[a].cpu().detach()), savefolder +"/sample"+str(i*config["batch_size"] + a)+".pt")
            
   return sample

if __name__ == '__main__':
   #model1 = torch.load(save_path +"/model.pt").to(device).eval()
   model = torch.load("syn_data_PosSensitive/model.pt").to(device).eval()
   generate_sample(model, num_samples = num_of_samples, save = True, savefolder= "syn_data_PosSensitive_new")
  # generate_sample_combined(model1, model2, num_samples = num_of_samples, save = True, savefolder= "combined")

 


 
