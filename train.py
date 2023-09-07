
import torch
from diffusion_Process import GuassianDiffusion
from model import DiffusionMLPModel
from unets import UNet1d
from dataloader import *
import wandb
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsize = 2
epochs = 2

def train_one_epoch(
    model,
    dataloader,
    diffusion,
    optimizer,
    lrs,
    class_cond
):
    model.train()
    avgloss = 1
    avglosssteps = 1
    minloss = 1
    for step, (genes, labels) in enumerate(dataloader):
        
        genes = (genes - genes.min().item() )/ (genes.max().item() - genes.min().item())
        assert (genes.max().item() <= 1) and (0 <= genes.min().item())
      #  print(genes.shape)
        # must use [-1, 1] pixel range for images
        genes, labels = (
            2 * genes.to(device) - 1,
            labels.to(device) if class_cond else None,
        )
        genes = genes.permute(0,2,1)
        t = torch.randint(diffusion.timesteps, (len(genes),), dtype=torch.int64).to(
            device
        )
        xt, eps = diffusion.sample_from_forward_process(genes, t)
      #  print(eps.min().item()), print(eps.max().item())
        print(xt.shape)
        pred_eps = model(xt, t, y=labels)

        if step % 50 == 0:
         #   model.train()
            print(f"step: {step}, loss: {avgloss/avglosssteps}")
            if avgloss/avglosssteps < minloss:
                minloss = avgloss/avglosssteps
                print("saving model at step: "  + str(step) +" ,with loss: "+ str(avgloss/avglosssteps) )
                torch.save(model, "model.pt")
            avgloss = 0
            avglosssteps = 0

     #   print(pred_eps[0])
        loss = ((pred_eps - eps) ** 2).mean()
        avgloss = avgloss  + loss.item()
        avglosssteps = avglosssteps + 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_dict = {"loss": loss.item()}
        if lrs is not None:
            lrs.step()
            log_dict["lr"] = lrs.get_last_lr()[0]

        wandb.log(log_dict)
        print(f"step: {step}, loss: {loss.item()}")



if __name__ == '__main__':
    wandb.init(project="diffusionGene")
    dataloader,_ = GeneticDataloaders(batchsize)

    diffusion = GuassianDiffusion(device =  device)
  #  model = DiffusionMLPModel().to(device)
    model = UNet1d(32,8,8, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lrs = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    for e in range(epochs):
        train_one_epoch(model, dataloader, diffusion, optimizer, lrs, True)
