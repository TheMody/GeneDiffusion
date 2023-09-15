
import torch
from diffusion_Process import GuassianDiffusion
from unets import UNet, UNet1d
from model import  Unet2D, UnetMLP
from dataloader import *
import wandb
import matplotlib.pyplot as plt
from utils import *
from generate import generate_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsize = 4
gradient_accumulation_steps = 8
epochs = 100
max_steps = 500
num_classes = 2
num_channels = 75584

if __name__ == '__main__':
    wandb.init(project="diffusionGene")
    dataloader,valdataloader = GeneticDataloaders(batchsize, True)
   # diffusion = diffusion_process(max_steps, 32, 32)
    diffusion = GuassianDiffusion(500)
  #  model = Unet2D(3,3, hidden_dim=[64,128,256,512], c_emb_dim=num_classes).to(device)
    model = UNet1d(32,in_channels=8, out_channels=8 ,num_classes=num_classes).to(device)
  #  model = UnetMLP(num_channels,num_channels, c_emb_dim=num_classes).to(device)
    critertion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
 #   optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
  #  lrs = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    lrs = CosineWarmupScheduler(optimizer, warmup=100, max_iters=epochs*len(dataloader))
    minloss = 1
    for e in range(epochs):
        model.train()
        avgloss = 0
        avglosssteps = 0
        for step in range(len(dataloader)//gradient_accumulation_steps):
            #assert (genes.max().item() <= 1) and (0 <= genes.min().item()) todo normalize genes
            optimizer.zero_grad()
            # must use [-1, 1] pixel range for images
            accloss = 0.0
            for micro_step in range(gradient_accumulation_steps):
                genes, labels = next(iter(dataloader))
                genes  = genes.to(device).float().permute(0,2,1)
                labels = labels.to(device)
                t = torch.randint(max_steps, (len(genes),), dtype=torch.int64).to(device)
                xt, eps = diffusion.sample_from_forward_process(genes,t)
                #xt, eps = diffusion.perturb_input(images, t)
                pred_eps = model(xt, t, y = labels)
                loss = critertion(pred_eps,eps)
                
                avgloss = avgloss  + loss.item()
                avglosssteps = avglosssteps + 1
                loss = loss/ gradient_accumulation_steps
                accloss += loss.item()
                loss.backward()

            optimizer.step()

            if step % 100 == 0 and step != 0:
                print(f"epoch: {e}, step: {step}, loss: {avgloss/avglosssteps}")
                avgloss = 0
                avglosssteps = 0
            log_dict = {"loss": accloss}
          #  print(accloss)
            if lrs is not None:
                lrs.step()
                log_dict["lr"] = lrs.get_last_lr()[0]

            wandb.log(log_dict)

        model.eval()
        with torch.no_grad():
            avgloss = 0
            avglosssteps = 0
            for step, (genes, labels) in enumerate(valdataloader):
               # assert (genes.max().item() <= 1) and (0 <= genes.min().item())
                # must use [-1, 1] pixel range for images
                genes  = genes.to(device).float().permute(0,2,1)
                labels = labels.to(device)
                t = torch.randint(max_steps, (len(genes),), dtype=torch.int64).to(device)
                xt, eps = diffusion.sample_from_forward_process(genes,t)
                #xt, eps = diffusion.perturb_input(images, t)
                pred_eps = model(xt, t, y = labels)
                loss = critertion(pred_eps,eps)
                avgloss = avgloss  + loss.item()
                avglosssteps = avglosssteps + 1
            log_dict = {"valloss": avgloss/avglosssteps}
            wandb.log(log_dict)
          
            print(f"val at epoch: {e},  loss: {avgloss/avglosssteps}")
            if avgloss/avglosssteps < minloss:
                minloss = avgloss/avglosssteps
                print("saving model at epoch: "  + str(e) +" ,with loss: "+ str(avgloss/avglosssteps) )
                torch.save(model, "model.pt")

    model = torch.load("model.pt").to(device)
    model.eval()
    generate_sample(model, 10000, savefolder="syn_data_model")

