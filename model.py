import torch.nn as nn
import torch.nn.functional as F
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffusionMLPModel(nn.Module):
    #very simple MLP based diffusion model for gene pca data
    def __init__(self, num_classes=2, num_input = 75584):
        super().__init__()
        self.layersizes = [num_input,int(num_input/64),int(num_input/128), int(num_input/256)]
        self.guidance = nn.Linear(num_classes, self.layersizes[3])
        self.time_step_emb = nn.Linear(1, self.layersizes[3])
        self.dense1 = nn.Linear(self.layersizes[0], self.layersizes[1])
        self.dense2 = nn.Linear(self.layersizes[1], self.layersizes[2])
        self.dense3 = nn.Linear(self.layersizes[2], self.layersizes[3])
        self.dense4 = nn.Linear(self.layersizes[3], self.layersizes[2])
        self.dense5 = nn.Linear(self.layersizes[2], self.layersizes[1])
        self.dense6 = nn.Linear(self.layersizes[1], self.layersizes[0])
       # self.dense7 = nn.Linear(self.layersizes[0], self.layersizes[0])

    #x should be in range [0,1], y should be one-hot encoded, t should be float, all batched
    def forward(self,x1, t,y=None):
        t_emb = self.time_step_emb(t.unsqueeze(-1).type(torch.float32).to(device))
        if not y == None:
           guidance_emb = self.guidance(y)
           t_emb = t_emb + guidance_emb
        x2 = F.silu(self.dense1(x1))
        x3 = F.silu(self.dense2(x2))
        x = F.silu(self.dense3(x3)) + t_emb 
        x = F.silu(self.dense4(x)) + x3
        x = F.silu(self.dense5(x)) + x2
        return self.dense6(x)

class MLPModel(nn.Module):
    def __init__(self, num_classes=2, num_input = 75584):
        super().__init__()
        hidden_dim = 1024
        self.dense1 = nn.Linear(num_input, hidden_dim)
     #   self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(8)])
        self.dense3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.gelu(self.dense1(x))
        for i in range(4):
            x2 = x
            x = F.gelu(self.linears[i*2](x))
            x = F.gelu(self.linears[i*2+1](x2)) +x2
    #    x = F.gelu(self.dense2(x))
        return F.softmax(self.dense3(x), dim=1)

