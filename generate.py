import torch
from diffusion_Process import GuassianDiffusion
from dataloader import *
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

steps = 500
img_size = 128

if __name__ == '__main__':
    diffusion = GuassianDiffusion(device =  device)
    model = torch.load("model.pt")
    model = model.to(device)

    xt = torch.randn(1,3,img_size,img_size).to(device)
    plt.imshow(diffusion.sample_from_reverse_process(model,xt, steps)[0].cpu().detach().permute(1,2,0)/ 2 + 0.5)
    plt.show()