import torch
from diffusion_Process import GuassianDiffusion
import matplotlib.pyplot as plt
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

steps = 50
gene_count = 18279
batchsize = 1

if __name__ == '__main__':
    diffusion = GuassianDiffusion(device =  device)
    model = torch.load("modelcondfull.pt")
    model = model.to(device)

    for i in range(35,10000):
        print("sample: " + str(i))
        xt = torch.randn(batchsize,8,gene_count)
        xt = F.pad(xt, (0, 18432 - xt.shape[2]), "constant", 0).to(device)
       # print(xt.shape)
        y = torch.randint(0,2,(batchsize,)).to(device)
        samples = diffusion.sample_from_reverse_process(model,xt, steps, model_kwargs= {"y":y})[0].cpu().detach()
        torch.save((samples,y), "syn_data/sample"+str(i)+".pt")
        