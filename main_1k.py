
import torch
from utils.diffusion_Process import GuassianDiffusion
from utils.unets import UNet, UNet1d, PosSensitiveUnet, PosSensitiveUnetDeep
from utils.model import  Unet2D, UnetMLP, UnetMLPandCNN, EncoderModelDiffusion, BaselineNet
from utils.dataloader import *
import wandb
from utils.utils import *
from KG1.generate1k import generate_sample
from KG1.config_1k import *
import time
from KG1.train_classifier_1k import train_classifier
from PIL import Image
import matplotlib.pyplot as plt

def data_abs_mean(x):
    return torch.mean(torch.abs(x))

def mse_loss_masked(pred, target, mask):   
    return torch.mean(((pred - target) ** 2) * torch.stack([mask for _ in range(pred.shape[0])]))

def train_diffusion():
    wandb.init(project="diffusionGene1k", config = config)
   # dataloader,valdataloader = GeneticDataloaders(config["batch_size"], True)

    train_dataset = GeneticDataset1k(train = True)
    test_dataset = GeneticDataset1k(train = False)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)
    valdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=True)
   # diffusion = diffusion_process(max_steps, 32, 32)
    diffusion = GuassianDiffusion(max_steps)
    diffusion.zero_mask = torch.tensor(zero_mask).permute(1,0)
  #  model = Unet2D(3,3, hidden_dim=[64,128,256,512], c_emb_dim=num_classes).to(device)
  #  model = torch.load("modellarge.pt")
    if model_name == "Baseline":
        model = BaselineNet(num_channels*gene_size,num_channels*gene_size, c_emb_dim=num_classes+1).to(device)
    if model_name == "UnetMLP":
        model = UnetMLP(num_channels*gene_size,num_channels*gene_size, c_emb_dim=num_classes+1).to(device)
    if model_name == "UnetMLPlarge":
        model = UnetMLP(num_channels*gene_size,num_channels*gene_size,hidden_dim=[2048,1024,512,256], c_emb_dim=num_classes+1).to(device)
    if model_name == "Transformer":
        model = EncoderModelDiffusion(num_classes=num_classes+1, num_genes=gene_size).to(device)
    if model_name == "UnetCombined":
        model = UnetMLPandCNN(channels_CNN = num_channels,channels_MLP = num_channels*gene_size,  base_width=base_width,num_classes=num_classes+1).to(device)
    if model_name == "Unet":
        model = UNet1d(in_channels=8, out_channels=8 ,num_classes=num_classes+1, base_width=base_width).to(device)
    if model_name == "PosSensitive":
        model = PosSensitiveUnet(sequence_length = gene_size, in_channels=8, out_channels=8 ,num_classes=num_classes+1).to(device)
    if model_name == "PosSensitiveLarge":
        model = PosSensitiveUnet(sequence_length = gene_size, in_channels=8, out_channels=8 ,num_classes=num_classes+1, base_width=192).to(device)
    if model_name == "PosSensitiveDeep":
        model = PosSensitiveUnetDeep(sequence_length = gene_size, in_channels=8, out_channels=8 ,num_classes=num_classes+1, base_width=64).to(device)
    if model_name == "PosSensitiveDeepLarge":
        model = PosSensitiveUnetDeep(sequence_length = gene_size, in_channels=8, out_channels=8 ,num_classes=num_classes+1, base_width=128).to(device)
    if model_name == "PosSensitiveDeepVeryLarge":
        model = PosSensitiveUnetDeep(sequence_length = gene_size, in_channels=8, out_channels=8 ,num_classes=num_classes+1, base_width=384).to(device)
  #  model = UnetMLP(num_channels,num_channels, c_emb_dim=num_classes).to(device)
    critertion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_diffusion)
  #  optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    lrs = CosineWarmupScheduler(optimizer, warmup=100, max_iters=epochs*len(dataloader)//gradient_accumulation_steps)


    if enforce_zeros:
        not_zero_mask = (zero_mask==0).permute(1,0).to(device)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    minloss = 1e10
    ema_time = 0.0
    for e in range(epochs):
        model.train()
        avgloss = 0
        avglosssteps = 0
        train_iter = iter(dataloader)
        for step in range(len(dataloader)//gradient_accumulation_steps):
            start = time.time()
            #assert (genes.max().item() <= 1) and (0 <= genes.min().item()) todo normalize genes
            optimizer.zero_grad()
            accloss = 0.0
            acc_extra = 0.0
            for micro_step in range(gradient_accumulation_steps):
                genes, labels = next(train_iter)
                genes  = genes.to(device).float().permute(0,2,1)
                labels = labels.to(device)
                t = torch.randint(max_steps, (len(genes),), dtype=torch.int64).to(device)

                xt, eps = diffusion.sample_from_forward_process(genes,t)
                pred_eps = model(xt, t, y = labels)
                if enforce_zeros:
                    loss = mse_loss_masked(pred_eps,eps, not_zero_mask)
                else:
                    loss = critertion(pred_eps,eps)
                
                x_r = diffusion.reverse_forward_process_simple(xt,  t, pred_eps)
                acc_extra += data_abs_mean(x_r-genes).item()
                
                avgloss = avgloss  + loss.item()
                avglosssteps = avglosssteps + 1
                loss = loss/ gradient_accumulation_steps
                accloss += loss.item()
                loss.backward()

            optimizer.step()

            if step % 20 == 0 and step != 0:
                print(f"epoch: {e}, step: {step}, loss: {avgloss/avglosssteps}")
                avgloss = 0
                avglosssteps = 0
            
            log_dict = {"loss": accloss, "reconstruction_error": acc_extra/gradient_accumulation_steps}
            if model_name == "UnetCombined":
                log_dict["weighing_factor"] = torch.mean(model.weighing_factor).item()
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
            acc_rec_error = 0.0
            ts = 0
            histogramm = torch.zeros((max_steps,2)).to(device)
            for step, (genes, labels) in enumerate(valdataloader):
                genes  = genes.to(device).float().permute(0,2,1)
                labels = labels.to(device)
                if ts + len(genes) > max_steps:
                    t = torch.Tensor([*[i for i in range(ts, min(ts+len(genes),max_steps))], *[i for i in range(1, ts+len(genes)-(max_steps-1))]]).type(torch.int64).squeeze(0).to(device)
                    ts =  ts+len(genes)-(max_steps-1)
                else:
                    t = torch.Tensor([range(ts, ts+len(genes))]).type(torch.int64).squeeze(0).to(device)
                    ts = ts + len(genes)

                t = t*2 % max_steps
                
                xt, eps = diffusion.sample_from_forward_process(genes,t)
                pred_eps = model(xt, t, y = labels)

                if enforce_zeros:
                    loss = mse_loss_masked(pred_eps,eps, not_zero_mask)
                else:
                    loss = critertion(pred_eps,eps)

                x_r = diffusion.reverse_forward_process_simple(xt,  t, pred_eps)
                
                histogramm[t, 0] = (histogramm[t, 0]  * histogramm[t, 1] + torch.mean(torch.abs(x_r-genes), dim = (1,2))) / (histogramm[t, 1]  +1)
                histogramm[t, 1] += 1
                
                acc_rec_error += data_abs_mean(x_r-genes).item()

                avgloss = avgloss  + loss.item()
                avglosssteps = avglosssteps + 1

            histogramm = histogramm.cpu().numpy()
            histogramm = histogramm[histogramm[:, 1] != 0]
            plt.bar(np.arange(len(histogramm[:, 0]))/len(histogramm[:, 0]), histogramm[:, 0], width = 1/len(histogramm[:, 0]))
            plt.yscale("log")
            plt.ylim (1e-2, 5e1)
            plt.xlabel("noise ratio")
            plt.ylabel("reconstruction error")
            plt.savefig(save_path+"/"+"histogramm.png")
            img = Image.open(save_path+"/"+"histogramm.png")
            log_dict = {"valloss": avgloss/avglosssteps, "val_rec_error": acc_rec_error/avglosssteps, "histogramm": wandb.Image(img)}
            wandb.log(log_dict)
        
            print(f"val at epoch: {e},  loss: {avgloss/avglosssteps} rec error:  {acc_rec_error/avglosssteps}")
            if acc_rec_error/avglosssteps < minloss:
                minloss = acc_rec_error/avglosssteps
                print("saving model at epoch: "  + str(e) +" ,with val rec error: "+ str(acc_rec_error/avglosssteps) )
                torch.save(model, save_path+"/"+"model.pt")

                
if __name__ == '__main__':
    train_diffusion()
    model = torch.load(save_path+"/"+"model.pt").to(device)
    model.eval()
    generate_sample(model, num_of_samples, savefolder=save_path)
    train_classifier("mlp")
    train_classifier("cnn")
    train_classifier("transformer")

