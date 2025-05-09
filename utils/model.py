import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from ALS.config import *
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


class DownsamplinBlockMLP(nn.Module):
    def __init__(self,input_size, output_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, output_size) 
        self.lin2 = nn.Linear(output_size, output_size) 
        self.lin3 = nn.Linear(output_size, output_size) 
        self.norm = nn.GroupNorm(32,output_size)
    
    def forward(self,x):

        x = F.silu(self.lin1(x))
        skip = x
        x = F.silu(self.lin2(x))
        x_skip = x
        x = self.norm(x)
        x = skip + F.silu(self.lin3(x))
        return x, x_skip

class UpsamplingBlockMLP(nn.Module):
    def __init__(self,input_size, output_size, c_emb_dim = 1):
        super().__init__()
        self.lin1 = nn.Linear(input_size, output_size)
        self.lin2 = nn.Linear(2*output_size, output_size) 
        self.lin3 = nn.Linear(output_size, output_size) 
        self.norm = nn.GroupNorm(32,output_size)
        self.t_emb_linear = nn.Linear(1, output_size)
        self.c_emb_linear = nn.Linear(c_emb_dim, output_size)
      #  self.emb_linear = nn.Linear(output_size*2, 2*output_size)
    
    def forward(self,x, x_skip, t_emb, c_emb = None):
        
        x = F.silu(self.lin1(x))
        skip = x
        x = self.norm(x)
        x = F.silu(self.lin2(torch.cat((x,x_skip), dim = 1)))
        # temb = self.t_emb_linear(t_emb.unsqueeze(-1).float())
        # if c_emb == None:
        #     cemb = torch.zeros_like(temb)
        # else:
        #     cemb = self.c_emb_linear(c_emb.float())
        # emb_out = self.emb_linear(F.silu(torch.cat((temb,cemb), dim = 1)))
        # scale, shift = torch.chunk(emb_out, 2, dim=1)
        # x = x * (1 + scale) + shift

        emb = self.t_emb_linear(t_emb.unsqueeze(-1).float())
        if c_emb == None:
            x = x + emb
        else:
            x = x * self.c_emb_linear(c_emb.float()) + emb
        x =  F.silu(self.lin3(x)) + skip 
        return x


class UnetMLP(nn.Module):

    def __init__(self,input_channel = 3, output_channel = 3, hidden_dim = [1024,512, 256], c_emb_dim = 10):
        super().__init__()
        self.c_emb_dim = c_emb_dim
        self.downBlock = nn.ModuleList()
        for dim in hidden_dim[:-1]:
            self.downBlock.append(DownsamplinBlockMLP(input_channel, dim))
            input_channel = dim

        self.bottleNeck = nn.Sequential(
            nn.Linear(input_channel, hidden_dim[-1]),
            nn.GroupNorm(32,hidden_dim[-1]),
            nn.SiLU(),
            nn.Linear(hidden_dim[-1], hidden_dim[-1]),
            nn.SiLU(),)

        self.upBlock = nn.ModuleList()
        hidden_dim.reverse()
        for dim in hidden_dim[:-1]:
            self.upBlock.append(UpsamplingBlockMLP(dim, dim*2, c_emb_dim))   
        hidden_dim.reverse()
        self.out = nn.Sequential(
            nn.GroupNorm(32,hidden_dim[0]),
            nn.SiLU(),
            nn.Linear(hidden_dim[0], output_channel),
            )

    def forward(self,x,t,y=None, output_bottleneck = False):
        shape = x.shape
        x = x.flatten(1)
        x_skips = []
        x_skips.append(x)
        y = F.one_hot(y, self.c_emb_dim)
        for block in self.downBlock:
         #   print(x.shape)
            x, x_skip = block(x)
            x_skips.append(x_skip)
      #  print(x.shape)
        x = self.bottleNeck(x)
        bottleneck = x
        for block in self.upBlock:
         #   print(x.shape)
         #   print(x_skips[-1].shape)
            x = block(x, x_skips.pop(),t/max_steps,y)
       # print(x.shape)
        x = self.out(x)+x_skips.pop()
        x = x.reshape(shape)
        if output_bottleneck:
            return x, bottleneck
        return x
    
class BaselineNet(nn.Module):
    def __init__(self,input_channel = 3, output_channel = 3, hidden_dim = 1024, c_emb_dim = 10):
        super().__init__()
        self.linear1 = nn.Linear(input_channel, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,hidden_dim)
        self.linear3 = nn.Linear(hidden_dim,output_channel)
        self.time_embedding = nn.Linear(1, hidden_dim)
        self.class_embedding = nn.Embedding(c_emb_dim, hidden_dim)

    def forward(self,x,t,y=None):
        shape = x.shape
        x = x.flatten(1)
        x_skip = x
        x = F.silu(self.linear1(x))
        c_emb = self.class_embedding(y)
        emb = self.time_embedding(t.unsqueeze(-1).float()/max_steps)
        x = x * c_emb + emb
        x = F.silu(self.linear2(x))
        x = self.linear3(x)
        x = x + x_skip
        x = x.reshape(shape)
        return x

class UnetMLPandPosSensitive(nn.Module):
    def __init__(self,channels_CNN ,channels_MLP , base_width = 64,  num_classes = 10 ):
        super().__init__()
        from utils.unets import PosSensitiveUnet
        self.MLP = UnetMLP(channels_MLP, channels_MLP,c_emb_dim = num_classes)
       # self.CNN = UNet1d(channels_CNN, channels_CNN, base_width = base_width, num_classes= num_classes)
        self.CNN = PosSensitiveUnet(sequence_length = gene_size, in_channels=channels_CNN, out_channels=channels_CNN ,num_classes=num_classes+1).to(device)
        self.learnable_weight_time = nn.Sequential(nn.Linear(1,128), nn.SiLU(), nn.Linear(128,1))
        self.weighing_factor = 0

    def forward(self,x,t,y=None, output_bottleneck = False):
        weighing_factor = F.sigmoid(self.learnable_weight_time(t.unsqueeze(-1).float()/max_steps)).unsqueeze(-1)
       # if output_bottleneck:
        # x1,bottleneck_mlp = self.MLP(x,t,y, output_bottleneck = output_bottleneck)
        # x2, bottleneck_cnn = self.CNN(x,t,y, output_bottleneck = output_bottleneck)
        # bottleneck = bottleneck_mlp * (1-weighing_factor) + torch.mean(bottleneck_cnn, dim = 2)* weighing_factor
        # else:  
        x1 = self.MLP(x,t,y)#, output_bottleneck = output_bottleneck)
        x2 = self.CNN(x,t,y)#, output_bottleneck = output_bottleneck)
       
        self.weighing_factor = weighing_factor
        x = x1 * (1-weighing_factor) + x2* weighing_factor
        # if output_bottleneck:
        #     return x, bottleneck
        return x


class UnetMLPandCNN(nn.Module):
    def __init__(self,channels_CNN ,channels_MLP , base_width = 64,  num_classes = 10 ):
        super().__init__()
        from utils.unets import UNet1d
        self.MLP = UnetMLP(channels_MLP, channels_MLP,c_emb_dim = num_classes)
        self.CNN = UNet1d(channels_CNN, channels_CNN, base_width = base_width, num_classes= num_classes)
        self.learnable_weight_time = nn.Sequential(nn.Linear(1,128), nn.SiLU(), nn.Linear(128,1))
        self.weighing_factor = 0

    def forward(self,x,t,y=None, output_bottleneck = False):
        weighing_factor = F.sigmoid(self.learnable_weight_time(t.unsqueeze(-1).float()/max_steps)).unsqueeze(-1)
        if output_bottleneck:
            x1,bottleneck_mlp = self.MLP(x,t,y, output_bottleneck = output_bottleneck)
            x2, bottleneck_cnn = self.CNN(x,t,y, output_bottleneck = output_bottleneck)
            bottleneck = bottleneck_mlp * (1-weighing_factor) + torch.mean(bottleneck_cnn, dim = 2)* weighing_factor
        else:  
            x1 = self.MLP(x,t,y, output_bottleneck = output_bottleneck)
            x2 = self.CNN(x,t,y, output_bottleneck = output_bottleneck)
       
        self.weighing_factor = weighing_factor
        x = x1 * (1-weighing_factor) + x2* weighing_factor
        if output_bottleneck:
            return x, bottleneck
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
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
    

class EncoderModelDiffusion(nn.Module):
    #input should be (batchsize, num_pcas, dim_pcas)
    def __init__(self, num_classes=2, input_dim = 8,  hidden_dim = 384,n_layers = 12, num_genes = 18432):
        super().__init__()
        self.num_classes = num_classes
        self.EmbeddingLayer = MultichannelLinear(num_genes, input_dim, hidden_dim,16)
        self.module_list = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=6,dim_feedforward=4*hidden_dim, batch_first=True, activation='gelu') for i in range(n_layers)])
        self.classification_token = nn.Parameter(torch.Tensor(hidden_dim), requires_grad=True)
        nn.init.uniform_(self.classification_token, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))
        self.time_embedding = nn.Linear(1, hidden_dim)
        self.class_embedding = nn.Embedding(num_classes, hidden_dim)
        self.c_emb_linear = nn.Linear(hidden_dim, hidden_dim)
        self.DeEmbeddingLayer = MultichannelLinear(num_genes//16, hidden_dim, input_dim,16, up = True)
        self.dense = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, t,y,last_hidden = True, embedding = False):
        x = x.permute(0,2,1)
        x_skip = x
        x = self.EmbeddingLayer(x)
        classification_token = torch.stack([self.classification_token.unsqueeze(0) for _ in range(x.shape[0])])
        
        time_token = self.time_embedding(t.unsqueeze(-1).float()/max_steps).unsqueeze(1)
        c_token = self.class_embedding(y).unsqueeze(1)
        x = torch.cat((classification_token,time_token,c_token,x),dim = 1)

        for layer in self.module_list:
            x = layer(x)

        classification_token = x[:,0,:]

        if embedding:
            return x
        if last_hidden:
            x = self.DeEmbeddingLayer(x[:,3:,:])
           # x = x.permute(0,2,1)
            return (x_skip + x).permute(0,2,1)
        return self.dense(classification_token)

class EncoderModelPreTrain(nn.Module):
    #input should be (batchsize, num_pcas, dim_pcas)
    def __init__(self, num_classes=2, input_dim = 8,  hidden_dim = 384,n_layers = 12):
        super().__init__()
       # self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.EmbeddingLayer = MultichannelLinear(18432, input_dim, hidden_dim,16)
        self.module_list = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=6,dim_feedforward=4*hidden_dim, batch_first=True, activation='gelu') for i in range(n_layers)])
        # self.PositionalEncoding = nn.Embedding(gene_size, hidden_dim)
        self.classification_token = nn.Parameter(torch.Tensor(hidden_dim), requires_grad=True)
        # self.pos_input = torch.zeros(batch_size, gene_size).long().to(device)
        # for i in range(gene_size):
        #     self.pos_input[:,i] = i
        nn.init.uniform_(self.classification_token, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))
        #self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.DeEmbeddingLayer = MultichannelLinear(18432//16, hidden_dim, input_dim,16, up = True)
        self.dense = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, last_hidden = False, embedding = False):

        x = self.EmbeddingLayer(x)
      #  x = self.dense1(x)
       # x += self.PositionalEncoding(self.pos_input)

        classification_token = torch.stack([self.classification_token.unsqueeze(0) for _ in range(x.shape[0])])
        x = torch.cat((classification_token,x),dim = 1)
        for layer in self.module_list:
            x = layer(x)
    
        classification_token = x[:,0,:]

        if embedding:
            return x
        if last_hidden:
            x = self.DeEmbeddingLayer(x[:,1:,:])
            return x
        return self.dense(classification_token)

class EncoderModel(nn.Module):
    #input should be (batchsize, num_pcas, dim_pcas)
    def __init__(self, num_classes=2, input_dim = 8,  hidden_dim = 384, input_size = 18432):
        super().__init__()
        self.EmbeddingLayer = MultichannelLinear(input_size, input_dim, hidden_dim,16)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=6,dim_feedforward=4*hidden_dim, batch_first=True, activation='gelu')
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=6,dim_feedforward=4*hidden_dim, batch_first=True, activation='gelu')
        self.encoder_layer3 = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=6,dim_feedforward=4*hidden_dim, batch_first=True, activation='gelu')
        self.encoder_layer4 = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=6,dim_feedforward=4*hidden_dim, batch_first=True, activation='gelu')
        self.encoder_layer5 = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=6,dim_feedforward=4*hidden_dim, batch_first=True, activation='gelu')
        self.encoder_layer6 = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=6,dim_feedforward=4*hidden_dim, batch_first=True, activation='gelu')
       # self.PositionalEncoding = nn.Embedding(18432, hidden_dim)
        self.encoding_token = nn.Parameter(torch.Tensor(hidden_dim), requires_grad=True)
        # self.pos_input = torch.zeros(batch_size, gene_size).long().to(device)
        # for i in range(gene_size):
        #     self.pos_input[:,i] = i
        nn.init.uniform_(self.encoding_token, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))
        #self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, last_hidden = False):

      #  x = self.dense1(x)
        x = self.EmbeddingLayer(x)
     #   print(x.shape)
        # if self.pos_input.shape[0] != x.shape[0]:
        #     self.pos_input = torch.zeros(x.shape[0], gene_size).long().to(device)
        #     for i in range(gene_size):
        #         self.pos_input[:,i] = i
        # x += self.PositionalEncoding(self.pos_input)

        encoding_token = torch.stack([self.encoding_token.unsqueeze(0) for _ in range(x.shape[0])])
        x = torch.cat((encoding_token,x),dim = 1)
        x = self.encoder_layer(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        x = self.encoder_layer5(x)
        x = self.encoder_layer6(x)[:,0,:]
        #x = F.gelu(self.dense2(x))
        if last_hidden:
            return x
        return self.dense3(x)

class MultichannelLinear(nn.Module): #maybe this is missing the bias term
    def __init__(self, channels, in_features, out_features,  project = 1, up = False):
        super(MultichannelLinear, self).__init__()
        self.up = up
        self.project = project
        if not up:
            self.weight_pw = nn.Parameter(torch.empty(int(math.ceil(channels/project)), out_features, in_features*project))
            self.weight_bias = nn.Parameter(torch.empty(int(math.ceil(channels/project)), out_features))
        else:
            self.weight_pw = nn.Parameter(torch.empty(channels, out_features*project, in_features))
            self.weight_bias = nn.Parameter(torch.empty(channels, out_features*project))
        nn.init.uniform_(self.weight_pw, a=-1/math.sqrt(in_features*project), b=1/math.sqrt(in_features*project))
        nn.init.uniform_(self.weight_bias, a=-1/math.sqrt(in_features*project), b=1/math.sqrt(in_features*project))

    def __call__(self, x):
        if not self.up:
            if not self.project ==1:   
                #reshape x to (batchsize, num_pcas/down_project, dim_pcas*down_project)
                if x.shape[1] % self.project != 0:
                    x = F.pad(x, (0,0,0,self.project - x.shape[1] % self.project))
                x = x.reshape(x.shape[0], int(x.shape[1]/self.project), x.shape[2]*self.project)
                
            x = torch.matmul(self.weight_pw.unsqueeze(0),x.unsqueeze(-1)).squeeze(-1) + self.weight_bias.unsqueeze(0)
        else:
            x = torch.matmul(self.weight_pw.unsqueeze(0),x.unsqueeze(-1)).squeeze(-1) + self.weight_bias.unsqueeze(0)
            #reshape x from (batchisze, channels,in_features)to (batchsize, channels * project, out_features)
            x = x.reshape(x.shape[0], int(x.shape[1]*self.project), int(x.shape[2]/self.project))
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
            x = F.gelu(self.linears[i*2+1](x)) +x2
    #    x = F.gelu(self.dense2(x))
        return self.dense3(x)

class ConvclsModel(nn.Module):
    def __init__(self, num_classes=2, input_dim = 8, num_layers = 3, input_size = 18432):
        super().__init__()
        hidden_dim = 64
     #   self.multilin = MultichannelLinear(18432, input_dim, 32)
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, 3, stride = 2)
        self.convs = nn.ModuleList([nn.Conv1d(hidden_dim, hidden_dim, 3, stride = 2) for i in range(num_layers)])
        def size_after_conv(kernel, stride, size):
            return math.floor((size-kernel)/stride) +1
        size = size_after_conv(3,2,size_after_conv(3,2,size_after_conv(3,2,size_after_conv(3,2,input_size))))
        print(size)
        self.dense3 = nn.Linear(hidden_dim*size, num_classes)
       # self.dense3 = nn.Linear(73664, num_classes)

    def forward(self, x):
       # x = F.gelu(self.multilin(x))
        x = x.permute(0,2,1)
        x = F.gelu(self.conv1(x))
        for conv in self.convs:
            x = F.gelu(conv(x))
        x = x.flatten(1)
       # print(x.shape)
        #x = torch.mean(x, dim = 2)
        return self.dense3(x)
