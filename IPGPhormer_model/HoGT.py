import numpy as np
import random
import math
import dgl
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from dgl.nn.pytorch import GATConv

class HoGT(nn.Module):
    def __init__(self, dim_in=768, dim_hidden=512, S=3, num_gnns=4, num_type=6):
        super(HoGT, self).__init__()          
        self.dim_hidden = dim_hidden
        self.num_gnns = num_gnns        #层数？
        self.S = S
        
        #转变初始维度
        self.fc = nn.Linear(dim_in, dim_hidden)
        
        self.GATLayer = torch.nn.ModuleList()        
        #初始化变换层
        for layer in range(self.num_gnns):
            self.GATLayer.append(GATConv(self.dim_hidden, self.dim_hidden, 1))

        self.beta = nn.Parameter(torch.ones(S + 1))     
    
    def forward(self, G:dgl.DGLGraph):
        device = G.device
        if G.ndata['feat'].shape[-1] != 512:
            G.ndata['feat'] = self.fc(G.ndata['feat'])
       
        for layer in range(self.num_gnns):
            G = self.GATLayer[layer](G)

        h = G.ndata['feat']
        self.beta = self.beta.to(device)
        A_hat = G.adj(scipy_fmt=None)
        A_hat = A_hat + torch.eye(A_hat.shape[0]).to(device)
        Z_prev = h
        Z = self.beta[0] * Z_prev

        for s in range(1, self.S + 1):
            # 稀疏矩阵乘法
            Z_s = torch.sparse.mm(A_hat, Z_prev)
            # 累加 beta_s * Z_s
            Z = Z + self.beta[s] * Z_s
            # 为下次迭代保存
            Z_prev = Z_s
        return Z_prev

