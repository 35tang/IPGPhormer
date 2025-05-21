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
    
class AGTLayer(nn.Module):
    def __init__(self, dim_in, nheads=6, emb_dropout=0.15):
        super(AGTLayer, self).__init__()

        self.nheads = nheads
        self.dim_in = dim_in    #输入维度：dim_in
        self.head_dim = self.dim_in // self.nheads

        #激活函数
        self.linear_k = nn.Linear(
            self.dim_in, self.head_dim * self.nheads, bias=False)
        self.linear_q = nn.Linear(
            self.dim_in, self.head_dim * self.nheads, bias=False)
        self.linear_v = nn.Linear(
            self.dim_in, self.head_dim * self.nheads, bias=False)
        self.relu = nn.ReLU()

        self.linear_final = nn.Linear(self.head_dim * self.nheads, self.dim_in, bias=False)
        
        self.dropout = nn.Dropout(emb_dropout) #嵌入dropout
        self.LN = nn.LayerNorm(dim_in)

    def forward(self, h):
        ''' transpose：交换两个维度，适用于二维或更高维的张量。
            permute：用于重排多个维度顺序，更灵活适应复杂的维度变换。
        '''
        batch_size = h.size()[0]
        #张量后两位是头数和头维度
        k = self.linear_k(h).reshape(batch_size, self.nheads, self.head_dim)
        q = self.linear_q(h).reshape(batch_size, self.nheads, self.head_dim)
        v = self.linear_v(h).reshape(batch_size, self.nheads, self.head_dim)

        q = self.relu(q)
        k = self.relu(k)
        
        #分子
        kv = torch.einsum('bnd, bne -> bde', k, v)
        num = torch.einsum('bnd, bde -> bne', q, kv)
        
        #分母
        qk = 1 / torch.einsum('bnd, bd -> bn', q, k.sum(dim=1))
        attn = torch.einsum('bne, bn -> bne', num, qk)
        
        fh = self.linear_final(attn)
        fh = self.dropout(fh)
        
        h = self.LN(h + fh)

        return h
    
#重点改
class HeGT(nn.Module):
    def __init__(self, dim_in=768, dim_hidden=512, S=3, num_gnns=4, num_type=6):
        super(HeGT, self).__init__()          
        self.dim_hidden = dim_hidden
        self.num_gnns = num_gnns        #层数？
        self.S = S
        
        #转变初始维度
        self.fc = nn.Linear(dim_in, dim_hidden)
        
        self.GATLayer = torch.nn.ModuleList()
        #初始化变换层
        for layer in range(self.num_gnns):
            self.GATLayer.append(GATConv(self.dim_hidden, self.dim_hidden, 1))
        
        #独热编码初始化
        self.type_emb = nn.Parameter(torch.eye(num_type))
        self.beta = nn.Parameter(torch.ones(S + 1))
        
    def forward(self, G:dgl.DGLGraph):
        #添加特征，转为同构图
        device = G.device
        for ntype in G.ntypes:
            if G.nodes[ntype].data['feat'].shape[-1] != 512:
                G.nodes[ntype].data['feat'] = self.fc(G.nodes[ntype].data['feat'])
            G.nodes[ntype].data['het_feat'] = self.type_emb[int(ntype)].repeat(G.num_nodes(ntype),1).to(device)
            G.nodes[ntype].data['node_type'] = torch.full((G.num_nodes(ntype), 1), int(ntype), device=device)

        G = dgl.to_homogeneous(G, ndata=['feat','het_feat','node_type'])
        G.ndata['feat'] = torch.cat([G.ndata['feat'], G.ndata['het_feat']], dim = -1)
        
        for layer in range(self.num_gnns):
            G = self.GATLayer[layer](G, G.ndata['feat'])
        
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