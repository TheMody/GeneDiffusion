
import torch
from diffusion_Process import GuassianDiffusion
from unets import UNet, UNet1d, PosSensitiveUnet, PosSensitiveUnetDeep
from model import  Unet2D, UnetMLP
from dataloader import *
import wandb
from utils import *
from generate import generate_sample
from config import *
import time
from train_classifier import train_classifier

if __name__ == '__main__':
    wandb.init(project="diffusionGene", config = config)
    dataloader,valdataloader = GeneticDataloaders(config["batch_size"], True)
   # diffusion = diffusion_process(max_steps, 32, 32)
    diffusion = GuassianDiffusion(max_steps)
  #  model = Unet2D(3,3, hidden_dim=[64,128,256,512], c_emb_dim=num_classes).to(device)
  #  model = torch.load("modellarge.pt")
    if model_name == "Unet":
        model = UNet1d(in_channels=8, out_channels=8 ,num_classes=num_classes+1).to(device)
    if model_name == "UnetLarge":
        model = UNet1d(in_channels=8, out_channels=8 ,num_classes=num_classes+1, base_width=192).to(device)
    if model_name == "PosSensitive":
        model = PosSensitiveUnet(sequence_length = gene_size, in_channels=8, out_channels=8 ,num_classes=num_classes+1).to(device)
    if model_name == "PosSensitiveLarge":
        model = PosSensitiveUnet(sequence_length = gene_size, in_channels=8, out_channels=8 ,num_classes=num_classes+1, base_width=192).to(device)
    if model_name == "PosSensitiveDeep":
        model = PosSensitiveUnetDeep(sequence_length = gene_size, in_channels=8, out_channels=8 ,num_classes=num_classes+1, base_width=64).to(device)
    if model_name == "PosSensitiveDeepLarge":
        model = PosSensitiveUnetDeep(sequence_length = gene_size, in_channels=8, out_channels=8 ,num_classes=num_classes+1, base_width=192).to(device)
    if model_name == "PosSensitiveDeepVeryLarge":
        model = PosSensitiveUnetDeep(sequence_length = gene_size, in_channels=8, out_channels=8 ,num_classes=num_classes+1, base_width=384).to(device)
  #  model = UnetMLP(num_channels,num_channels, c_emb_dim=num_classes).to(device)
    critertion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_diffusion)
 #   optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    lrs = CosineWarmupScheduler(optimizer, warmup=100, max_iters=epochs*len(dataloader)//gradient_accumulation_steps)


    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    minloss = 1
    ema_time = 0.0
    for e in range(epochs):
        model.train()
        avgloss = 0
        avglosssteps = 0
        for step in range(len(dataloader)//gradient_accumulation_steps):
            start = time.time()
            #assert (genes.max().item() <= 1) and (0 <= genes.min().item()) todo normalize genes
            optimizer.zero_grad()
            accloss = 0.0
            for micro_step in range(gradient_accumulation_steps):
                genes, labels = next(iter(dataloader))
                genes  = genes.to(device).float().permute(0,2,1)
                labels = labels.to(device)

                #mask out label with 10% probability
                random__label_masks = torch.rand(labels.size()).to(device)
                random__label_masks = random__label_masks > 0.1
                labels = torch.where(random__label_masks, labels, num_classes)      

                t = torch.randint(max_steps, (len(genes),), dtype=torch.int64).to(device)
                xt, eps = diffusion.sample_from_forward_process(genes,t)
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

            time_taken = time.time() - start
            ema_time = ema_time * 0.99 + time_taken * 0.01 #exponential moving average
            ema_time_corrected = ema_time / (1 - 0.99 ** (step + 1 + e * len(dataloader)//gradient_accumulation_steps))#bias corrected ema
            log_dict["time_per_step"] = time_taken
            remaining_time = int((len(dataloader)//gradient_accumulation_steps - step) * ema_time_corrected + (epochs-1 -e)*(len(dataloader)//gradient_accumulation_steps) * ema_time_corrected)
            print(f'estimated remaining time {remaining_time:3d} sec at step {step}/{len(dataloader)//gradient_accumulation_steps} of epoch {e}/{epochs}', end='\r')

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
                genes  = genes.to(device).float().permute(0,2,1)
                labels = labels.to(device)
                t = torch.randint(max_steps, (len(genes),), dtype=torch.int64).to(device)
                xt, eps = diffusion.sample_from_forward_process(genes,t)
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
                torch.save(model, save_path+"/"+"model.pt")
    wandb.finish()
    model = torch.load(save_path+"/"+"model.pt").to(device)
    model.eval()
    generate_sample(model, num_of_samples, savefolder=save_path)
    train_classifier()

