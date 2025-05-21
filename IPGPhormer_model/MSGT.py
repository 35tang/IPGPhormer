import torch
# torch.backends.cudnn.enabled = False
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention

import dgl
import dgl.function as fn
from dgl.nn import edge_softmax
from dgl.nn.pytorch.glob import GlobalAttentionPooling

from .HeGT import HeGT
from .HoGT import HoGT

class LearnableAttentionMapFusion(nn.Module):
    def __init__(self, init_weight_small=0.5, init_weight_large=0.5):
        """
        初始化可学习权重的注意力图融合模块
        
        参数:
            init_weight_small (float): 小尺寸注意力图的初始权重
            init_weight_large (float): 大尺寸注意力图的初始权重
        """
        super(LearnableAttentionMapFusion, self).__init__()
        
        # 创建可学习的权重参数，并用初始值进行初始化
        self.weight_small = nn.Parameter(torch.tensor(init_weight_small, dtype=torch.float))
        self.weight_large = nn.Parameter(torch.tensor(init_weight_large, dtype=torch.float))
        
        # 均值池化层，用于将大尺寸注意力图缩小
        # 实际的pooling size会在forward中动态确定
        self.pool = nn.AdaptiveAvgPool2d(output_size=None)
        
    def forward(self, attention_small, attention_large):
        """
        将两个不同尺寸的注意力图进行融合
        
        参数:
            attention_small (torch.Tensor): 形状为 (b, 1, n, n) 的注意力图
            attention_large (torch.Tensor): 形状为 (b, 1, 4n, 4n) 的注意力图
            
        返回:
            torch.Tensor: 形状为 (b, 1, n, n) 的融合后的注意力图
        """
        batch_size, channels, n, _ = attention_small.shape
        
        # 检查输入形状是否符合要求
        _, _, large_n, _ = attention_large.shape
        assert large_n == 4 * n, f"Large attention map size should be 4n, got {large_n} instead of {4*n}"
        
        # 使用平均池化将大的注意力图从 (b, 1, 4n, 4n) 缩小到 (b, 1, n, n)
        pooled_attention = self.pool(attention_large, output_size=(n, n))
        
        # 使用学习的权重合并两个注意力图
        # 应用softmax来确保权重求和为1
        weights = F.softmax(torch.stack([self.weight_small, self.weight_large]), dim=0)
        
        fused_attention = weights[0] * attention_small + weights[1] * pooled_attention
        
        return fused_attention


class AGTLayer(nn.Module):
    def __init__(self, dim_in, nheads=8, emb_dropout=0.05):
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
        return h, q, k, v


class MSGT(nn.Module):
    def __init__(self, dim_in=768, dim_hidden = 512, cell_dim = 80, dropout = 0.15, n_classes = 4, node_types = 6, n_layers = 1, m_layers = 1, pooling_type='mean'):
        super(MSGT, self).__init__()
        
        self.n_layers = n_layers    #10X层数
        self.m_layers = m_layers    #5X层数
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.node_type = node_types
        self.cell_dim = cell_dim

        self.he_branch = HeGT(dim_in = self.dim_in, dim_hidden = self.dim_hidden - self.node_type)
        self.ho_branch = HoGT(dim_in = self.dim_in, dim_hidden = self.dim_hidden)
        self.hegt = nn.ModuleList()
        self.hogt = nn.ModuleList()

        for _ in range(n_layers):
            self.hegt.append(AGTLayer(dim_in = dim_hidden))
        
        for _ in range(m_layers):
            self.hogt.append(AGTLayer(dim_in = dim_hidden))

        self.cox_linear = nn.Linear(self.cell_dim, 1, bias = False)
        
        if pooling_type == "mean":
            self.pooling = global_mean_pool 
        elif pooling_type == "max":
            self.pooling = global_max_pool 
        elif pooling_type == "attn":
            att_net=nn.Sequential(nn.Linear(dim_hidden, dim_hidden // 2), nn.LeakyReLU(), nn.Linear(dim_hidden//2, 1))     
            self.pooling = GlobalAttention(att_net)
        else:
            raise NotImplementedError
        
        self.conv = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
        self.mean = nn.MaxPool2d(kernel_size = 4, stride = 4)
        self.linear_attn = nn.Linear(dim_hidden, dim_hidden, bias=False)

        # 添加注意力图融合层
        self.attention_fusion = LearnableAttentionMapFusion(init_weight_small=0.5, init_weight_large=0.5)
        
        self.message_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_hidden)
        self.classfication = nn.Linear(dim_hidden, n_classes)

    def forward(self, G_het:dgl.DGLGraph, G_homo:dgl.DGLGraph, cell_feature):
        h1 = self.he_branch(G_het)
        h2 = self.ho_branch(G_homo)
        
        for i in range(self.n_layers):
            h1, q1, k1, v1 = self.hegt[i](h = h1)
        for j in range(self.m_layers):
            h2, q2, k2, v2 = self.hogt[j](h = h2)
        
        qk1 = torch.einsum('bnd, bne -> bnn', q1, k1)
        qk1_sum = 1 / torch.einsum('bnd, bd -> bn', q1, k1.sum(dim=1))
        attn1 = torch.einsum('bne, bn -> bnn', qk1, qk1_sum)

        qk2 = torch.einsum('bnd, bne -> bnn', q2, k2)
        qk2_sum = 1 / torch.einsum('bnd, bd -> bn', q2, k2.sum(dim=1) + 1e-6)
        attn2 = torch.einsum('bne, bn -> bnn', qk2, qk2_sum)

        attn1 = attn1.unsqueeze(1)  # [b, 1, n, n]
        attn2 = attn2.unsqueeze(1)  # [b, 1, 4n, 4n]
        
        # 使用mean池化将attn2缩小
        pooled_attn = self.mean(attn2)  # [b, 1, n, n]
        
        # 使用可学习权重融合两个注意力图
        fused_attn = self.attention_fusion(pooled_attn, attn2)
        fused_attn = fused_attn.squeeze(1)
        h = torch.einsum('bnn, bnd -> bnd', fused_attn, v2)
        h = self.linear_attn(h)

        cell_h = self.cox_linear(cell_feature)
        cell_h = torch.exp(cell_h)
        
        h = self.norm(h1)  # 这里使用h1，您可能需要根据实际情况调整
        h = self.classfication(h)
        h = h * cell_h  
        h = self.pooling(h.squeeze(0), batch=None)
        
        logits = h
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
       
        return hazards, S, Y_hat 