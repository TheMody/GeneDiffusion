from config import *
from dataloader import GeneticDataSets, SynGeneticDataset
import torch
import torch.nn.functional as F

t = torch.randint(max_steps, (1,), dtype=torch.int64).to(device)
x = torch.rand((1, 8,18432)).to(device)
y = torch.randint(num_classes+1, (1,), dtype=torch.int64).to(device)

model = torch.load(save_path+"/"+"model.pt").to(device)
x,bottleneck = model(x, t,y, output_bottleneck=True)
print(x.shape)
print(bottleneck.shape)



# geneticDataSyn_working = SynGeneticDataset(path= "syn_data_UnetMLP_working/")
# geneticDataSyn_not = SynGeneticDataset(path= "syn_data_UnetMLP/")
# geneticDataSyn_not2 = SynGeneticDataset(path= "syn_data_PosSensitive/")
# geneticData_real, test = GeneticDataSets()

# def func(x):
#     return torch.mean(torch.abs(x))

# print(func(geneticData_real[2][0]))
# print(func(geneticDataSyn_working[2][0]))
# print(func(geneticDataSyn_not[2][0]))
# print(func(geneticDataSyn_not2[2][0]))
