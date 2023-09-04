
import torch
from diffusion_Process import GuassianDiffusion
from model import DiffusionMLPModel
from dataloader import *
import wandb
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsize = 128
epochs = 100

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

        # must use [-1, 1] pixel range for images
        genes, labels = (
            2 * genes.to(device) - 1,
            labels.to(device) if class_cond else None,
        )
        t = torch.randint(diffusion.timesteps, (len(genes),), dtype=torch.int64).to(
            device
        )
        xt, eps = diffusion.sample_from_forward_process(genes, t)
        pred_eps = model(xt, t, y=labels)


        #logging
        if step % 50 == 0:
            model.train()
            print(f"step: {step}, loss: {avgloss/avglosssteps}")
            if avgloss/avglosssteps < minloss:
                minloss = avgloss/avglosssteps
                print("saving model at step: "  + str(step) +" ,with loss: "+ str(avgloss/avglosssteps) )
                torch.save(model, "model.pt")
            avgloss = 0
            avglosssteps = 0

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
     #   print(f"step: {step}, loss: {loss.item()}")



if __name__ == '__main__':
    wandb.init(project="diffusionGene")
    dataloader,_ = GeneticDataloaders(batchsize)

    diffusion = GuassianDiffusion(device =  device)
    model = DiffusionMLPModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    lrs = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    for e in range(epochs):
        train_one_epoch(model, dataloader, diffusion, optimizer, lrs, False)
