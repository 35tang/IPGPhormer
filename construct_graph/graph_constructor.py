import os
import glob
import pickle
import sys
from pathlib import Path
from importlib import import_module
import h5py
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import numpy as np
from termcolor import colored
from itertools import chain
import nmslib

# Graph Network Packages
import dgl
from scipy.stats import pearsonr

from .extractor import Extractor
# from efficientnet_pytorch import EfficientNet

from data import PatchData

'''
Graph construction v2 for new format of patches

区分为6类
Node types:
From PanNuke dataset
0) No-label '0'
1) Neoplastic '1'
2) Inflammatory '2'
3) Connective '3'
4) Dead '4'
5) Non-Neoplastic Epithelial '5'

Edge types:
0 or 1 based on Personr correlation between nodes
'''

#KNN算法，寻找K个最近邻居。每次只能对一个向量找，所以多个要用循环
class Hnsw:
    """
    KNN model cloned from https://github.com/mahmoodlab/Patch-GCN/blob/master/WSI-Graph%20Construction.ipynb
    """

    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        #余弦相似度
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices

#将多 GPU 模式（DataParallel 模式）保存的模型权重转换为单 GPU 模式。
def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                    "%s: Detect checkpoint saved in data-parallel mode."
                    " Converting saved model to single GPU mode." % colored_word
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict

#节点分类
class Hovernet_infer:
    """ Run HoverNet inference """

    def __init__(self, config, dataloader):
        self.dataloader = dataloader

        method_args = {
            'method': {
                'model_args': {'nr_types': config['nr_types'], 'mode': config['mode'], },
                'model_path': config['hovernet_model_path'],
            },
        }
        run_args = {
            'batch_size': config['batch_size'],
        }

        model_desc = import_module("hovernet.net_desc")
        model_creator = getattr(model_desc, "create_model") #调用create_model函数
        
        #加载参数
        net = model_creator(**method_args['method']["model_args"])
        #["desc"] 从加载的文件中提取出模型描述部分（状态字典），通常是一个包含模型权重和其它信息的字典。
        saved_state_dict = torch.load(method_args['method']["model_path"])["desc"]
        #将多 GPU 模式（DataParallel 模式）保存的模型权重转换为单 GPU 模式。
        saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)
        #strict=False 表示如果模型中有某些层的权重与检查点中的层不匹配，也不会抛出错误。
        net.load_state_dict(saved_state_dict, strict=False)
        #并行计算。模型将自动分配到多个 GPU 上进行训练，提高计算效率。
        net = torch.nn.DataParallel(net, device_ids=[0, 1])
        net = net.to("cuda")

        #把 infer_step 函数包装成了一个可调用的对象。
        module_lib = import_module("hovernet.run_desc")
        run_step = getattr(module_lib, "infer_step")
        self.run_infer = lambda input_batch: run_step(input_batch, net) #lambda函数将将 input_batch 和 net 作为参数传递给 run_step

    def predict(self):
        output_list = []
        # features_list = []
        for idx, data in enumerate(self.dataloader):
            #重置维度（适应卷积要求）
            data = data.permute(0, 3, 2, 1)
            output, __ = self.run_infer(data)
            # curr_batch_size = output.shape[0]
            # output = np.split(output, curr_batch_size, axis=0)[0].flatten()
            
            # features_list.append(features)
            for out in output:      #output_list的元素为单个数值而非列表
                #判断张量元素是否均非0
                if out.any() == 0:
                    output_list.append(0)
                else:
                    #计算非零元素中出现频率最高的元素，并将该元素添加到 output_list 中。（多数表决法）
                    out = out[out != 0]
                    #np.bincount返回一个数组，其中数组的索引表示数字，值表示该数字出现的次数。
                    #argmax() 返回数组中最大值的索引，即返回出现次数最多的数字的值的索引。
                    #综合来看，max_occur_node_type记录最多次数类别
                    max_occur_node_type = np.bincount(out).argmax()
                    output_list.append(max_occur_node_type)

        return output_list

class GraphConstructor:
    def __init__(self, config: OrderedDict, hovernet_config: OrderedDict, wsi_data, name):
        self.config = config
        self.hovernet_config = hovernet_config
        # self.kimianet_config = kimianet_config
        self.wsi_data = wsi_data
        self.name = name
        
        #半径
        self.radius = self.config['radius']
        #在欧氏空间上进行K近邻
        self.knn_model = Hnsw(space='l2')

        #获取patch的路径
        patch_path = Path(wsi_data)
        patch_dataset = PatchData(patch_path)
        #import torch.utils.data as data中的data
        dataloader = data.DataLoader(
            patch_dataset,
            num_workers=0,
            batch_size=8,
            shuffle=False
        )

        #对patch进行编码的方式，已经确定用Ctranspath
        self.encoder_name = config['encoder_name']
        node_type_dir = config["node_type_dir"]
        hovernet_model = Hovernet_infer(self.hovernet_config, dataloader)
        #默认用hover_net进行分类
        if node_type_dir is None or self.encoder_name == "hover":
            self.node_type = hovernet_model.predict()
        elif node_type_dir:
            #head为目录路径；tail为文件名。
            head, tail = os.path.split(wsi_data)
            node_type_file = os.path.join(node_type_dir + tail + '.pkl')
            with open(node_type_file, "rb") as f:
                self.node_type = pickle.load(f)
        
        #提取patch1特征，进行编码

        if self.encoder_name == "CtransPath":
            print("Use CtransPath feature.Use Euler data construct")
            root_dir = '/data115_2/jsh/TCGA/KIRC/10x/Ctranspath/h5_files'
            final_path = os.path.join(root_dir, f"{self.name}.h5")
            
            #检查是否存在
            files = glob.glob(final_path)
            if files:
                print(f"找到文件: {files[0]}")  # 这里files[0]是第一个匹配的文件
            else:
                print("没有找到匹配的文件")
            
            with h5py.File(final_path, 'r') as f:
                # 列出所有顶级的键（组和数据集的名字）
                print("Keys:", list(f.keys()))
                self.features_cons = f['coords'][:]         #用于连接边
                self.features_realfeat = f['features'][:]   #用作节点特征
            
    def construct_graph(self):

        ####################
        # Step 1 Fit the coordinate with KNN and construct edges
        ####################
        # Number of patchesz
        
        n_patches = self.features_cons.shape[0]

        # Construct graph using spatial coordinates（欧氏空间）
        self.knn_model.fit(self.features_cons)
        print("work 20")
            
        a = np.repeat(range(n_patches), self.radius - 1)    #源节点索引
        b = np.fromiter(
            chain(
                *[self.knn_model.query(self.features_cons[v_idx], topn=self.radius)[1:] for v_idx in range(n_patches)]
            ), dtype=int
        )   #目标节点索引
        edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)
        print("work 21")
        
        
        # Create edge types
        edge_type = []
        edge_sim = []
        for (idx_a, idx_b) in zip(a, b):
            metric = pearsonr
            corr = metric(self.features_realfeat[idx_a], self.features_realfeat[idx_b])[0]
            #type只设置两个值（完全相关或不相关）
            edge_type.append(1 if corr > 0 else 0)
            edge_sim.append(corr)
        print("work 22")
        
        # Construct dgl heterogeneous graph
        graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
        graph.ndata.update({'_TYPE': torch.tensor(self.node_type)})
        self.features = torch.tensor(self.features_realfeat).float()
        self.patches_coords = torch.tensor(self.features_cons).float()
        
        graph.ndata['feat'] = self.features                 #存特征
        graph.ndata['patches_coords'] = self.patches_coords #存坐标
        
        graph.edata.update({'_TYPE': torch.tensor(edge_type)})
        graph.edata.update({'sim': torch.tensor(edge_sim)})
        
        het_graph = dgl.to_heterogeneous(
            graph,
            [str(t) for t in range(self.config["n_node_type"])],
            ['neg', 'pos']
        )
        print("work 23")

        # homo_graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
        # homo_graph.ndata['feat'] = self.features
        # self.edge_sim = torch.tensor(edge_sim)
        # homo_graph.edata['sim'] = self.edge_sim
        # homo_graph.ndata['patches_coords'] = self.patches_coords

        self.node_type = torch.tensor(self.node_type)
        return het_graph, self.node_type
