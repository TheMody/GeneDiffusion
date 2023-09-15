import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from unets import UNet1d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class DownsamplinBlock2d(nn.Module):
    def __init__(self,input_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding = 1) 
        self.conv2 = nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding = 1) 
        self.conv3 = nn.Conv2d(output_size, output_size, kernel_size=3, stride=2, padding = 1) 
        self.norm = nn.GroupNorm(32,output_size)
    
    def forward(self,x):
        x = F.silu(self.conv1(x))
        x_skip = F.silu(self.conv2(x))
        x = self.norm(x)
        x = F.silu(self.conv3(x))
        return x, x_skip

class UpsamplingBlock2d(nn.Module):
    def __init__(self,input_size, output_size, c_emb_dim = 1):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(input_size, output_size, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(2*output_size, output_size, kernel_size=3, stride=1, padding = 1) 
        self.conv3 = nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding = 1) 
        self.norm = nn.GroupNorm(32,output_size)
        self.t_emb_linear = nn.Linear(1, output_size)
        self.c_emb_linear = nn.Linear(c_emb_dim, output_size)
    
    def forward(self,x, x_skip, t_emb, c_emb = None):
        x = F.silu(self.conv1(x))
        x = self.norm(x)
        x = F.silu(self.conv2(torch.cat((x,x_skip), dim = 1)))
        emb = self.t_emb_linear(t_emb.unsqueeze(-1).float())
        if c_emb == None:
            x = x + emb.unsqueeze(-1).unsqueeze(-1)
        else:
            x = x * self.c_emb_linear(c_emb.float()).unsqueeze(-1).unsqueeze(-1) + emb.unsqueeze(-1).unsqueeze(-1)
        x = F.silu(self.conv3(x))
        return x


class Unet2D(nn.Module):

    def __init__(self,input_channel = 3, output_channel = 3, hidden_dim = [64,128,256,512,1024], c_emb_dim = 10):
        super().__init__()
        self.c_emb_dim = c_emb_dim
        self.downBlock = nn.ModuleList()
        for dim in hidden_dim[:-1]:
            self.downBlock.append(DownsamplinBlock2d(input_channel, dim))
            input_channel = dim

        self.bottleNeck = nn.Sequential(
            nn.Conv2d(input_channel, hidden_dim[-1], kernel_size=3, stride=1, padding = 1),
            nn.GroupNorm(32,hidden_dim[-1]),
            nn.SiLU(),
            nn.Conv2d(hidden_dim[-1], hidden_dim[-1], kernel_size=3, stride=1, padding = 1),
            nn.SiLU(),)

        self.upBlock = nn.ModuleList()
        hidden_dim.reverse()
        for dim in hidden_dim[1:]:
            self.upBlock.append(UpsamplingBlock2d(dim*2, dim, c_emb_dim))   
        hidden_dim.reverse()
        self.out = nn.Sequential(
            nn.GroupNorm(32,hidden_dim[0]),
            nn.SiLU(),
            nn.Conv2d(hidden_dim[0], output_channel, kernel_size=1, stride=1, padding = 0),
            )

    def forward(self,x,t,y=None):
        x_skips = []
        y = F.one_hot(y, self.c_emb_dim)
        for block in self.downBlock:
            x, x_skip = block(x)
            x_skips.append(x_skip)
        x = self.bottleNeck(x)
        for block in self.upBlock:
            x = block(x, x_skips.pop(),t,y)
        x = self.out(x)
        return x

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class DiffUnet1d(nn.Module):
    #very simple MLP based diffusion model for gene pca data
    def __init__(self, num_classes=2, num_channels  = 3):
        super().__init__()
        self.unet = UNet1d(32)

    def forward(self,x, t, y=None):
        x = x.reshape((x.shape[0],3, x.shape[2]*x.shape[3]))
        x = self.unet(x, t, y)
        return x.reshape((x.shape[0],3,32,32))


class MLPBlock(nn.Module):
    def __init__(self,input_size, output_size, emb_size):
        super().__init__()

        self.emb_projection = nn.Linear(emb_size, input_size)  
        self.dense1 = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm( output_size)
        self.dense2 = nn.Linear(output_size, output_size)

    def forward(self,x,emb):
        x = self.emb_projection(emb) + x
        x = F.silu(self.dense1(x))
        x = self.norm(x)
        x = self.dense2(x)
        return x

class D1ConvBlock(nn.Module):
    def __init__(self,input_size, output_size, emb_size, mode = "down"):
        super().__init__()
        self.mode = mode
        self.emb_projection =  nn.Linear(emb_size, input_size)
        if mode == "down":
            stride = 2
        else:
            stride = 1
        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size=5, stride=stride, padding = 2) 
        self.norm = nn.InstanceNorm1d(output_size)
        self.conv2 = nn.Conv1d(output_size, output_size, kernel_size=5, padding = 2) 

    def forward(self,x,emb):
        if self.mode == "up":
            x = F.interpolate(x, (x.shape[2]*2), mode="nearest")
        x = x + self.emb_projection(emb).unsqueeze(-1)
        x = F.silu(self.conv1(x))
        x = self.norm(x)
        x = self.conv2(x)
        return x


class D2ConvBlock(nn.Module):
    def __init__(self,input_size, output_size, emb_size, mode = "down"):
        super().__init__()
        self.mode = mode
        self.emb_projection =  nn.Linear(emb_size, input_size)
       # input_size_conv = input_size
        if mode == "down":
            stride = 2
        else:
            stride = 1
      #  if self.mode == "up":
           # self.upconv3 = nn.ConvTranspose2d(input_size, output_size, kernel_size=2, stride=2)
         #   input_size_conv = output_size
        self.conv1 = nn.Conv2d(input_size, output_size, kernel_size=3, stride=stride, padding = 1) 
        self.norm = nn.BatchNorm2d(output_size)
        self.conv2 = nn.Conv2d(output_size, output_size, kernel_size=3, padding = 1) 

    def forward(self,x,emb):
        x = x + self.emb_projection(emb).unsqueeze(-1).unsqueeze(-1)
        if self.mode == "up":
          #  x = self.upconv3(x)
            x = F.interpolate(x, (x.shape[2]*2, x.shape[3]*2), mode="nearest")
        
        x = F.silu(self.conv1(x))
        x = self.norm(x)
        x = F.silu(self.conv2(x))
        return x

class DiffusionConv2dModel(nn.Module):
    #very simple MLP based diffusion model for gene pca data
    def __init__(self, num_classes=2, num_channels  = 3):
        super().__init__()

        emb_dim = 256
        self.num_channels = num_channels
        self.layersizes = [num_channels,32,64,128]
        self.guidance = nn.Linear(num_classes, emb_dim)
        self.time_step_emb = nn.Linear(1, emb_dim)

    
        self.down = nn.ModuleList()
        for i in range(1, len(self.layersizes)):
            print(self.layersizes[i-1], self.layersizes[i])
            self.down.append(D2ConvBlock(self.layersizes[i-1], self.layersizes[i], emb_dim, "down"))
         #   self.down.append(D2ConvBlock(self.layersizes[i], self.layersizes[i], emb_dim, "bottleneck"))
       # print("bottleneck")
        self.bottleneck = D2ConvBlock(self.layersizes[-1], self.layersizes[-1], emb_dim, "bottleneck")
        self.up = nn.ModuleList()
        for i in range(len(self.layersizes)-1, 0,-1):
            print(self.layersizes[i], self.layersizes[i-1])
            self.up.append(D2ConvBlock(self.layersizes[i], self.layersizes[i-1], emb_dim, "up"))
          #  self.up.append(D2ConvBlock(self.layersizes[i-1], self.layersizes[i-1], emb_dim, "bottleneck"))
        self.out = zero_module(nn.Conv2d(num_channels, num_channels, kernel_size=1) )

    #x should be in range [0,1], y should be one-hot encoded, t should be float, all batched
    def forward(self,x, t,y=None):
        #x = x.reshape((x.shape[0],self.num_channels, x.shape[2]*x.shape[3]))
        emb = self.time_step_emb(t.unsqueeze(-1).type(torch.float32).to(device))
        if not y == None:
           guidance_emb = self.guidance(y)
           emb = emb + guidance_emb

        xs = []
        for i,module in enumerate(self.down):
            x = module(x, emb)
            #if (i+1)%2 == 0:
            xs.append(x)
        xs.pop()
        x = self.bottleneck(x, emb)
        for i,module in enumerate(self.up):
          #  if (i)%2 == 0:
            if not i == 0:
                x = torch.cat([x, xs.pop()], dim = 1)
            x = module(x, emb)

        return self.out(x)#x.reshape((x.shape[0],3,32,32))

class DiffusionConv1dModel(nn.Module):
    #very simple MLP based diffusion model for gene pca data
    def __init__(self, num_classes=2, num_channels  = 3):
        super().__init__()

        emb_dim = 256
        self.num_channels = num_channels
        self.layersizes = [num_channels,32,64,128]
        self.guidance = nn.Linear(num_classes, emb_dim)
        self.time_step_emb = nn.Linear(1, emb_dim)

    
        self.down = nn.ModuleList()
        for i in range(1, len(self.layersizes)):
            self.down.append(D1ConvBlock(self.layersizes[i-1], self.layersizes[i], emb_dim, "down"))

        self.bottleneck = D1ConvBlock(self.layersizes[-1], self.layersizes[-1], emb_dim, "bottleneck")
        self.up = nn.ModuleList()
        for i in range(len(self.layersizes)-1, 0,-1):
            self.up.append(D1ConvBlock(self.layersizes[i], self.layersizes[i-1], emb_dim, "up"))


    #x should be in range [0,1], y should be one-hot encoded, t should be float, all batched
    def forward(self,x, t,y=None):
        x = x.reshape((x.shape[0],self.num_channels, x.shape[2]*x.shape[3]))
        emb = self.time_step_emb(t.unsqueeze(-1).type(torch.float32).to(device))
        if not y == None:
           guidance_emb = self.guidance(y)
           emb = emb + guidance_emb

        xs = []
        for i,module in enumerate(self.down):
            x = module(x, emb)
            xs.append(x)
        x = self.bottleneck(x, emb)
        for i,module in enumerate(self.up):
            x = module(x + xs.pop(), emb)

        return x.reshape((x.shape[0],3,32,32))

class DiffusionMLPModel(nn.Module):
    #very simple MLP based diffusion model for gene pca data
    def __init__(self, num_classes=2, num_input = 75584):
        super().__init__()

        emb_dim = 256
        self.num_input = num_input
        self.layersizes = [num_input,int(num_input/64),int(num_input/128), int(num_input/256)]
        self.guidance = nn.Linear(num_classes, emb_dim)
        self.time_step_emb = nn.Linear(1, emb_dim)

    
        self.down = nn.ModuleList()
        for i in range(1, len(self.layersizes)):
            self.down.append(MLPBlock(self.layersizes[i-1], self.layersizes[i], emb_dim))
        self.up = nn.ModuleList()
        for i in range(len(self.layersizes)-1, 0,-1):
            self.up.append(MLPBlock(self.layersizes[i], self.layersizes[i-1], emb_dim))


    #x should be in range [0,1], y should be one-hot encoded, t should be float, all batched
    def forward(self,x, t,y=None):
        x = x.reshape((x.shape[0],self.num_input))
        emb = self.time_step_emb(t.unsqueeze(-1).type(torch.float32).to(device))
        if not y == None:
           guidance_emb = self.guidance(y)
           emb = emb + guidance_emb

        xs = []
        for i,module in enumerate(self.down):
            x = module(x, emb)
            xs.append(x)

        for i,module in enumerate(self.up):
            x = module(x, emb)

        return x.reshape((x.shape[0],3,32,32))

class MLPModel(nn.Module):
    def __init__(self, num_classes=2, num_input = 8 * 18432):
        super().__init__()
        hidden_dim = 512
        self.dense1 = nn.Linear(num_input, hidden_dim)
     #   self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(2)])
        self.dense3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.flatten(start_dim = 1)
        x = F.gelu(self.dense1(x))
        for i in range(1):
            x2 = x
            x = F.gelu(self.linears[i*2](x))
            x = F.gelu(self.linears[i*2+1](x2)) +x2
    #    x = F.gelu(self.dense2(x))
        return F.softmax(self.dense3(x), dim=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
     #   self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x

class EncoderModel(nn.Module):
    #input should be (batchsize, num_pcas, dim_pcas)
    def __init__(self, num_classes=2, input_dim = 8):
        super().__init__()
        hidden_dim = 16
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1, batch_first=True, activation='gelu')
        self.PositionalEncoding = PositionalEncoding(hidden_dim, max_len=18279)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
       # 
       # self.dense1 = nn.Linear(num_input, hidden_dim)
       # self.dense2 = nn.Linear(hidden_dim, hidden_dim)
      #  self.dense3 = nn.Linear(64, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
      #  print(x.shape)
      #  print(x.type())
        x = self.dense1(x)
        x = self.PositionalEncoding(x)
       # x = self.encoder_layer(x)[:,0,:]
       # x,_ = torch.max(x, dim=1)#/x.shape[1]
        x = torch.sum(x, dim=1)/x.shape[1]
        x = F.gelu(self.dense2(x))
      #  x = F.gelu(self.dense2(x))
     #   x = F.gelu(self.dense3(x))
        return F.softmax(self.dense3(x), dim=1)

class MultichannelLinear(nn.Module): #maybe this is missing the bias term
    def __init__(self, channels, in_features, out_features,  down_project = 1):
        super(MultichannelLinear, self).__init__()
        self.down_project = down_project
        self.weight_pw = nn.Parameter(torch.empty(int(math.ceil(channels/down_project)), out_features, in_features*down_project))
        self.weight_bias = nn.Parameter(torch.empty(int(math.ceil(channels/down_project)), out_features))
        nn.init.uniform_(self.weight_pw, a=-1/math.sqrt(in_features*down_project), b=1/math.sqrt(in_features*down_project))
        nn.init.uniform_(self.weight_pw, a=-1/math.sqrt(in_features*down_project), b=1/math.sqrt(in_features*down_project))

    def __call__(self, x):
        if not self.down_project ==1:   
            #reshape x to (batchsize, num_pcas/down_project, dim_pcas*down_project)
            if x.shape[1] % self.down_project != 0:
                x = F.pad(x, (0,0,0,self.down_project - x.shape[1] % self.down_project))
            x = x.reshape(x.shape[0], int(x.shape[1]/self.down_project), x.shape[2]*self.down_project)
            
        x = torch.matmul(self.weight_pw.unsqueeze(0),x.unsqueeze(-1)).squeeze(-1) + self.weight_bias.unsqueeze(0)
     #   print(x.shape)
        return x

class IndMLPModel(nn.Module):
    #input should be (batchsize, num_pcas, dim_pcas)
    def __init__(self, num_classes=2, input_dim = 8, num_pcas = 18279):
        super().__init__()
        self.num_pcas = num_pcas
       # self.linears = nn.ModuleList([nn.Linear(input_dim, 1) for i in range(self.num_pcas)])
        self.linears1 = MultichannelLinear(self.num_pcas, input_dim, 32,32)
     #   self.linears2 = MultichannelLinear(int(math.ceil(self.num_pcas/8)), 8, 16,8)
        hidden_dim = 512
        self.dense1 = nn.Linear(18304, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense8 = nn.Linear(hidden_dim, hidden_dim)
        self.dense4 = nn.Linear(hidden_dim, hidden_dim)
        self.dense5 = nn.Linear(hidden_dim, hidden_dim)
        self.dense6 = nn.Linear(hidden_dim, hidden_dim)
        self.dense7 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
    #    print(x.shape)
    #    print(F.gelu(self.linears[0](x[:,0,:])).shape)
       # x = torch.cat([F.gelu(self.linears[i](x[:,i,:])) for i in range(self.num_pcas)], dim=1)
     #   x = F.gelu(self.linears1(x))
        x = F.gelu(self.linears1(x).flatten(1))
     #   print(x.shape)
        x2 = F.gelu(self.dense1(x))
        x = F.gelu(self.dense2(x2))  
        x2 = F.gelu(self.dense8(x)) +x2
        x = F.gelu(self.dense4(x2)) 
        x2 = F.gelu(self.dense5(x)) +x2
        x = F.gelu(self.dense6(x2))
        x = F.gelu(self.dense7(x)) +x2
     #   x = F.gelu(self.dense2(x))
        return F.softmax(self.dense3(x), dim=1)
