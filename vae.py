
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
from model import MultichannelLinear
from config import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EncoderBlock(nn.Module):
    def __init__(self,input_dims, output_dims, down = True):
        super(EncoderBlock, self).__init__()
        self.down = down
        if down:
            self.convdown = nn.Conv1d(input_dims, output_dims, 5, stride=2, padding=2)
        self.norm = nn.GroupNorm(8,output_dims)
        self.conv1 = nn.Conv1d(output_dims, output_dims*4, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(output_dims*4, output_dims, 3, stride=1, padding=1)
    def forward(self,x):
        if self.down:
            x = F.silu(self.convdown(x))
        skipx = x
        x = self.norm(x)
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = x + skipx
        return x

class DecoderBlock(nn.Module):
    def __init__(self,input_dims, output_dims, up = True):
        super(DecoderBlock, self).__init__()
        self.up = up
        if up:
            self.convup = nn.ConvTranspose1d(input_dims, output_dims, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(8,output_dims)
        self.conv1 = nn.Conv1d(output_dims, output_dims*4, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(output_dims*4, output_dims, 3, stride=1, padding=1)

    def forward(self,x):
        if self.up:
            x = F.silu(self.convup(x))
        skipx = x
        x = self.norm(x)
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = x + skipx
        return x


class Encoder(nn.Module):
    def __init__(self, input_dims, hidden_sizes = [32,64,128,256, 512]):
        super(Encoder, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv1d(input_dims, hidden_sizes[0], 3, stride=1, padding=1))
        for i in range(len(hidden_sizes)-1):
            self.convs.append(EncoderBlock(hidden_sizes[i], hidden_sizes[i+1]))
        self.convmu = nn.Conv1d(hidden_sizes[-1], hidden_sizes[-1], 1)
        self.convsigma = nn.Conv1d(hidden_sizes[-1], hidden_sizes[-1], 1)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        mu =  self.convmu(x)
        logsigma = self.convsigma(x)
        return mu, logsigma


class Decoder(nn.Module):
    def __init__(self,  output_dims, hidden_sizes = [32,64,128,256, 512, 1024,2048]):
        super(Decoder, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(len(hidden_sizes)-2, -1 , -1):
            self.convs.append(DecoderBlock(hidden_sizes[i+1], hidden_sizes[i]))
        self.convs.append(nn.Conv1d( hidden_sizes[0],output_dims, 3, stride=1, padding=1))

    def forward(self, x):
        for conv in self.convs[:-1]:
            x = conv(x)
        x = self.convs[-1](x)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, data_dim, hidden_sizes = [128,192,256,384,512,768]):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder( hidden_sizes[0], hidden_sizes)
        self.decoder = Decoder(hidden_sizes[0], hidden_sizes)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
        self.input_layer = MultichannelLinear(gene_size, data_dim, hidden_sizes[0],16)
        self.output_layer = MultichannelLinear(gene_size//16, hidden_sizes[0], data_dim,16, up = True)

    def forward(self, x, train = True):
        x = self.input_layer(x.permute(0,2,1)).permute(0,2,1)
        mu, logsigma = self.encoder(x)
        if train:    
            z = mu + torch.exp(logsigma)*self.N.sample(mu.shape)
            self.kl = (torch.exp(logsigma)**2 + mu**2 - logsigma - 1/2).mean()
        else:
            z = mu
        x = self.decoder(z)
        x = self.output_layer(x.permute(0,2,1)).permute(0,2,1)
        return x
    
class Block(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(Block, self).__init__()
        self.projection = nn.Linear(input_dims, output_dims)
        self.norm = nn.GroupNorm(8,output_dims)
        self.conv1 = nn.Linear(output_dims, output_dims)
        self.conv2 = nn.Linear(output_dims, output_dims)


    def forward(self,x):
        x = F.silu(self.projection(x))
        skipx = x
        x = self.norm(x)
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = x + skipx
        return x 
        
    
class Encoder(nn.Module):
    def __init__(self,  hidden_sizes ):
        super().__init__()
        self.blocks = nn.ModuleList([Block(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self,  hidden_sizes ):
        super().__init__()
        self.layers = nn.ModuleList([Block(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_size = gene_size*num_channels, hidden_sizes = [2048,1024]):
        super().__init__()
        self.encoder = Encoder([input_size]+hidden_sizes)
        self.decoder = Decoder(list(reversed(hidden_sizes)))
        self.out = nn.Linear(hidden_sizes[0], input_size)

    def forward(self, x):
        xshape = x.shape
        x = x.flatten(1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.out(x)
        x = x.reshape(xshape)
        return x
        