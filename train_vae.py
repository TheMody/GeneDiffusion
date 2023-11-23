
import torch
from diffusion_Process import GuassianDiffusion
from unets import UNet, UNet1d, PosSensitiveUnet, PosSensitiveUnetDeep
from model import  Unet2D, UnetMLP
from vae import VariationalAutoencoder
from dataloader import *
import wandb
from utils import *
from generate import generate_sample
from config import *
import time
from train_classifier import train_classifier
from sls.adam_sls import AdamSLS

def compute_mse_baseline(dataloader):
    avg_gene = torch.zeros((gene_size,num_channels)).to(device)
    for step, sample in enumerate(dataloader):
        gene, label = sample
        avg_gene += gene.to(device).mean(dim=0)
    avg_gene = avg_gene/len(dataloader)
    avg_mse = 0
    for step, sample in enumerate(dataloader):
        gene, label = sample
        gene = gene.to(device)
        mse = F.mse_loss(gene, avg_gene)
        avg_mse = avg_mse + mse.item()
        if step % 100 == 0:
            print("step", step, "mse", mse)
    avg_mse = avg_mse/len(dataloader)
    return avg_mse
        

if __name__ == '__main__':
    wandb.init(project="diffusionGeneVAE", config = config)
    dataloader,valdataloader = GeneticDataloaders(config["batch_size"]*gradient_accumulation_steps, True)
   # mean_mse = compute_mse_baseline(dataloader) #results in mean_mse 0.5125327877716526
   # print("mean_mse", mean_mse)
    model = VariationalAutoencoder(num_channels).to(device)
    critertion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_vae)
  #  optimizer = AdamSLS([[param for param in model.parameters()]], c = 0.3)
  #  optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    lrs = CosineWarmupScheduler(optimizer, warmup=100, max_iters=epochs_vae*len(dataloader)//gradient_accumulation_steps)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    minloss = 1
    ema_time = 0.0
    for e in range(epochs_vae):
        model.train()
        avgloss = 0
        avglosssteps = 0
        train_iter = iter(dataloader)
        for step in range(len(dataloader)//gradient_accumulation_steps):
            start = time.time()
            #assert (genes.max().item() <= 1) and (0 <= genes.min().item()) todo normalize genes
            optimizer.zero_grad()
            genes, _ = next(train_iter)
            def closure(backward=False):
                accloss = 0.0
                for micro_step in range(gradient_accumulation_steps):
                    genes_small = genes[micro_step*batch_size:(micro_step+1)*batch_size].to(device)
                    genes_small  = genes_small.to(device).float().permute(0,2,1) #shape (batch_size, num_channels, gene_size)
                    genes_r = model(genes_small)
                #  genes_r = genes_r.reshape(genes.shape)
                    mseloss = ((genes_small - genes_r)**2).mean() 
                    kldloss = model.kl * kl_factor
                    loss = mseloss + kldloss
                    loss = loss/ gradient_accumulation_steps
                    accloss += loss
                    if backward:
                        loss.backward()
                return accloss
            def closure_with_backward():
                return closure(backward=True)
            
          #  loss = optimizer.step(closure = closure, closure_with_backward = closure_with_backward)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            loss = closure_with_backward()
            optimizer.step()
            avgloss = avgloss  + loss.item()
            avglosssteps = avglosssteps + 1

            if step % 100 == 0 and step != 0:
                print(f"epoch: {e}, step: {step}, loss: {avgloss/avglosssteps}")
                avgloss = 0
                avglosssteps = 0
            log_dict = {"loss": loss.item()}#, "kldloss": kldloss, "mseloss": mseloss}

            time_taken = time.time() - start
            ema_time = ema_time * 0.99 + time_taken * 0.01 #exponential moving average
            ema_time_corrected = ema_time / (1 - 0.99 ** (step + 1 + e * len(dataloader)//gradient_accumulation_steps))#bias corrected ema
            log_dict["time_per_step"] = time_taken
            remaining_time = int((len(dataloader)//gradient_accumulation_steps - step) * ema_time_corrected + (epochs_vae-1 -e)*(len(dataloader)//gradient_accumulation_steps) * ema_time_corrected)
            print(f'estimated remaining time {remaining_time:3d} sec at step {step}/{len(dataloader)//gradient_accumulation_steps} of epoch {e}/{epochs_vae}', end='\r')
          #  log_dict["lr"] = optimizer.state["step_sizes"][0]
           # print(optimizer.state["step_sizes"][0])
            if lrs is not None:
                lrs.step()
                log_dict["lr"] = lrs.get_last_lr()[0]

            wandb.log(log_dict)

       # if e % 1000 == 0 and e != 0:
        model.eval()
        with torch.no_grad():
            avgloss = 0
            avglosssteps = 0
            for step, (genes, labels) in enumerate(valdataloader):
                # if step > 100:
                #     break
            # assert (genes.max().item() <= 1) and (0 <= genes.min().item())
                genes  = genes.to(device).float().permute(0,2,1)
                genes_r = model(genes, train = False)
                loss = ((genes - genes_r)**2).mean() 
                avgloss = avgloss  + loss.item()
                avglosssteps = avglosssteps + 1
            log_dict = {"valloss": avgloss/avglosssteps}
            wandb.log(log_dict)
        
            print(f"val at epoch: {e},  loss: {avgloss/avglosssteps}")
            if avgloss/avglosssteps < minloss:
                minloss = avgloss/avglosssteps
                print("saving model at epoch: "  + str(e) +" ,with loss: "+ str(avgloss/avglosssteps) )
                torch.save(model, save_path+"/"+"vaemodel.pt")


