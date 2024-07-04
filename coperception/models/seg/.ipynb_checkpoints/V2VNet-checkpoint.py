import torch

import coperception.utils.convolutional_rnn as convrnn
from coperception.models.seg.SegModelBase import SegModelBase
import torch.nn.functional as F
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LSTM
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np

class V2VNet(SegModelBase):
    def __init__(self, n_channels, n_classes, num_agent=5, compress_level=0, only_v2i=False):
        super().__init__(
            n_channels, n_classes, num_agent=num_agent, compress_level=compress_level, only_v2i=only_v2i
        )
        self.layer_channel = 512
        self.gnn_iter_num = 1
#         self.num_relations = 5
#         self.edge_encoder = Linear(1048576,self.num_relations)
#         self.edge_ext_thresh = 0.5
#         self.graph = {}
#         self.graph_list = []
#         self.activation = F.relu
#         self.graph['edge_attr'] = []
#         self.graph['edge_index'] = []
        
        self.convgru = convrnn.Conv2dGRU(
            in_channels=self.layer_channel * 2,
            out_channels=self.layer_channel,
            kernel_size=3,
            num_layers=1,
            bidirectional=False,
            dilation=1,
            stride=1,
        )

        
        
    def weight1(self,node_feature_list,weight):
        weight_feat = []
        for i in range(len(node_feature_list)):
            a = 0
            for j in range(len(node_feature_list)):
                a = a+weight[i][j]*node_feature_list[j]
            weight_feat.append(a)
        return weight_feat[0]
    
    
    
    def weight2(self,node_feature_list,weight):
        a = 0
#         weight0 = weight[0] / np.sum(np.abs(weight[0]))
#         print(weight0)
        print(weight[0])
        weight0 = weight[0] / torch.sum(torch.abs(weight[0]))  # 对张量进行归一化操作
        print(weight0)
        for i in range(len(node_feature_list)):
            a = a+weight0[i]*node_feature_list[i]
        return a
    
    def weight3(self,node_feature_list,weight):
        a = 1/len(node_feature_list)*node_feature_list[0]
#         print(len(node_feature_list))
#         print(len(weight))
#         weight0 = weight[0] / np.sum(np.abs(weight[0]))
#         print(weight0)
        weight1 = [x * (1-1/len(node_feature_list)) for x in weight]
#         print(sum(weight1))
        for i in range(len(weight1)):
            a = a+weight1[i]*node_feature_list[i+1]
        return a
    
    def weight4(self,node_feature_list,weight):
        a = 0.3*node_feature_list[0]
#         print(len(node_feature_list))
#         print(len(weight))
#         weight0 = weight[0] / np.sum(np.abs(weight[0]))
#         print(weight0)
        weight1 = [x * 0.7 for x in weight]
#         print(sum(weight1))
        for i in range(len(weight1)):
            a = a+weight1[i]*node_feature_list[i+1]
        return a
    
    
    def weight_rsu(self,node_feature_list):
        return torch.mean(torch.stack(node_feature_list), dim=0)
    
    
    def weight_ve(self,node_feature_list,weight):
#         if weight[0] > 1/len(node_feature_list):
#             a = 1/len(node_feature_list)*node_feature_list[0]
#             weight1 = [x * (1-1/len(node_feature_list)) for x in weight]
#         for i in range(len(weight)):
#             weight[i] = weight[i].detach()
#         print(weight)
        if weight[0]*(1-1/len(node_feature_list)) > 1/len(node_feature_list):
            a = 1/len(node_feature_list)*node_feature_list[0]
            weight1 = [x * (1-1/len(node_feature_list)) for x in weight]
    #         print(sum(weight1))
            for i in range(len(weight1)):
                a = a+weight1[i]*node_feature_list[i+1]
            return a  
        else:
            return torch.mean(torch.stack(node_feature_list), dim=0)
            
    
    def forward(self, x, trans_matrices, num_agent_tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # b 512 32 32
        size = (1, 512, 32, 32)

        if self.compress_level > 0:
            x4 = F.relu(self.bn_compress(self.com_compresser(x4)))
            x4 = F.relu(self.bn_decompress(self.com_decompresser(x4)))

        batch_size = x.size(0) // self.num_agent
        feat_list = super().build_feat_list(x4, batch_size)
#         print("feat_list：",len(feat_list))
        local_com_mat = torch.cat(tuple(feat_list), 1)
#         print("local_com_mat：",local_com_mat.size())
        local_com_mat_update = torch.cat(tuple(feat_list), 1)
#         print("local_com_mat_update：",local_com_mat_update.size())
        for b in range(batch_size):
            com_num_agent = num_agent_tensor[b, 0] 
#             print("com_num_agent:",com_num_agent)
            agent_feat_list = list() #表示
            for nb in range(self.num_agent):
                agent_feat_list.append(local_com_mat[b, nb])# 将一个batch的所有智能体向量都提到一个列表中

            for _ in range(self.gnn_iter_num):
                updated_feats_list = list()  #有几层gnn

                for i in range(com_num_agent):
                    tg_agent = local_com_mat[b, i]

                    neighbor_feat_list = list()
                    neighbor_feat_list.append(tg_agent)

                    for j in range(com_num_agent):
                        if j != i:
                            if self.only_v2i and i != 0 and j != 0:
                                continue
                            
                            neighbor_feat_list.append(
                                super().feature_transformation(
                                    b,
                                    j,
                                    i,
                                    local_com_mat,
                                    size,
                                    trans_matrices,
                                )
                            )
#                     print(len(neighbor_feat_list))
                    node_feature_list1 = neighbor_feat_list
                    node_feature_list = torch.stack(node_feature_list1, dim=0)
#                     print(node_feature_list.shape)
                    node_feature_list = node_feature_list.view(node_feature_list.size(0), -1)#(6,512*32*32)   
                    weight = []
                    weights = []
                    d = node_feature_list[0]
                    for i in range(1,len(node_feature_list)):
                        c = node_feature_list[i]
                        weight.append(1-torch.cosine_similarity(d, c,dim=0))
#                         weight.append(torch.cosine_similarity(d, c,dim=0))
                    for i in range(len(weight)):
                        if weight[i] < 0:
                            weight[i] = 0
                    for i in range(len(weight)):
                        weights.append(weight[i]/sum(weight))
#                     print(weight)
#                     print("weights:",weights)
#                     weights = F.softmax(torch.tensor(weight), dim=0)
#                     new_arr = torch.ones([len(node_feature_list), len(node_feature_list)]).triu(diagonal=1)
#                     new_arr[0][0] = torch.tensor(1)
#                     new_arr_idx = torch.where(new_arr==1.0)

#                     combo_list = torch.stack(new_arr_idx).t()  #将含矩阵中含有1的元素坐标，再次变得好看
#                     new_arr_2 = new_arr.flatten().int()   #将上半矩阵平铺为一维

#                     new_arr_idx2 = torch.where(new_arr_2==1.0) #平铺之后的1的索引
        
#                     node_combo_a = node_feature_list.unsqueeze(0).repeat((node_feature_list.size(0), 1,1))
#                     node_combo_b = node_feature_list.unsqueeze(1).repeat((1, node_feature_list.size(0),1))
#                     node_combo = torch.cat([node_combo_b, node_combo_a], dim=-1).flatten(start_dim=0, end_dim=1)
#                     node_combinations = node_combo[new_arr_idx2]
#                     edge_vectors = self.edge_encoder(node_combinations.float())
# #                     print(edge_vectors)
#                     edge_vectors = torch.sigmoid(edge_vectors)
# #                     print(edge_vectors)
#                     top_edges = torch.max(edge_vectors,dim=1)
#                     n = len(node_feature_list)
#                     weight = torch.zeros((n, n))
#                     top_edges_list =  top_edges[0]
#                     weight[0][0] = top_edges_list[0]
#                     k = 1
#                     for i in range(len(node_feature_list)):
#                         for j in range(i+1, len(node_feature_list)):
#                             weight[i][j] = top_edges_list[k]
#                             k = k+1
#                     A = weight
#                     A = A + A.T - torch.diag(torch.diag(A))

# #                     print(A)
                    if i ==1 :
#                     mean_feat = self.weight_rsu(node_feature_list1,weights)
                        mean_feat = self.weight_rsu(node_feature_list1)
                    else:
                        mean_feat = self.weight_ve(node_feature_list1,weights)
                    
#                     mean_feat = neighbor_feat_list
#                     mean_feat = torch.mean(torch.stack(neighbor_feat_list), dim=0)
#                     print(agent_feat_list[i].shape)
#                     print(mean_feat.shape)
#                     print(agent_feat_list[i].shape)
#                     print(mean_feat.shape)
                    cat_feat = torch.cat([agent_feat_list[i], mean_feat], dim=0)
                    cat_feat = cat_feat.unsqueeze(0).unsqueeze(0)
                    updated_feat, _ = self.convgru(cat_feat, None)
                    updated_feat = torch.squeeze(torch.squeeze(updated_feat, 0), 0)
                    updated_feats_list.append(updated_feat)
                agent_feat_list = updated_feats_list
#             print("x4:",x4.size())
            for k in range(com_num_agent):
                local_com_mat_update[b, k] = agent_feat_list[k]

        feat_mat = super().agents_to_batch(local_com_mat_update)
#         print("feat_mat:",feat_mat.size())
        x5 = self.down4(feat_mat)
#         print("x5:",x5.size())
        x = self.up1(x5, feat_mat)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
#         print("x:",x.size())
        logits = self.outc(x)
        return logits
# import torch

# import coperception.utils.convolutional_rnn as convrnn
# from coperception.models.seg.SegModelBase import SegModelBase
# import torch.nn.functional as F


# class V2VNet(SegModelBase):
#     def __init__(self, n_channels, n_classes, num_agent=5, compress_level=0, only_v2i=False):
#         super().__init__(
#             n_channels, n_classes, num_agent=num_agent, compress_level=compress_level, only_v2i=only_v2i
#         )
#         self.layer_channel = 512
#         self.gnn_iter_num = 1
#         self.convgru = convrnn.Conv2dGRU(
#             in_channels=self.layer_channel * 2,
#             out_channels=self.layer_channel,
#             kernel_size=3,
#             num_layers=1,
#             bidirectional=False,
#             dilation=1,
#             stride=1,
#         )

#     def forward(self, x, trans_matrices, num_agent_tensor):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)  # b 512 32 32
#         size = (1, 512, 32, 32)

#         if self.compress_level > 0:
#             x4 = F.relu(self.bn_compress(self.com_compresser(x4)))
#             x4 = F.relu(self.bn_decompress(self.com_decompresser(x4)))

#         batch_size = x.size(0) // self.num_agent
#         feat_list = super().build_feat_list(x4, batch_size)
# #         print("feat_list：",len(feat_list))
#         local_com_mat = torch.cat(tuple(feat_list), 1)
# #         print("local_com_mat：",local_com_mat.size())
#         local_com_mat_update = torch.cat(tuple(feat_list), 1)

#         for b in range(batch_size):
#             com_num_agent = num_agent_tensor[b, 0]

#             agent_feat_list = list()
#             for nb in range(self.num_agent):
#                 agent_feat_list.append(local_com_mat[b, nb])

#             for _ in range(self.gnn_iter_num):
#                 updated_feats_list = list()

#                 for i in range(com_num_agent):
#                     tg_agent = local_com_mat[b, i]

#                     neighbor_feat_list = list()
#                     neighbor_feat_list.append(tg_agent)

#                     for j in range(com_num_agent):
#                         if j != i:
#                             if self.only_v2i and i != 0 and j != 0:
#                                 continue
                            
#                             neighbor_feat_list.append(
#                                 super().feature_transformation(
#                                     b,
#                                     j,
#                                     i,
#                                     local_com_mat,
#                                     size,
#                                     trans_matrices,
#                                 )
#                             )

#                     mean_feat = torch.mean(torch.stack(neighbor_feat_list), dim=0)
#                     cat_feat = torch.cat([agent_feat_list[i], mean_feat], dim=0)
#                     cat_feat = cat_feat.unsqueeze(0).unsqueeze(0)
#                     updated_feat, _ = self.convgru(cat_feat, None)
#                     updated_feat = torch.squeeze(torch.squeeze(updated_feat, 0), 0)
#                     updated_feats_list.append(updated_feat)
#                 agent_feat_list = updated_feats_list
# #             print("x4:",x4.size())
#             for k in range(com_num_agent):
#                 local_com_mat_update[b, k] = agent_feat_list[k]

#         feat_mat = super().agents_to_batch(local_com_mat_update)
# #         print("feat_mat:",feat_mat.size())
#         x5 = self.down4(feat_mat)
# #         print("x5:",x5.size())
#         x = self.up1(x5, feat_mat)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
# #         print("x:",x.size())
#         logits = self.outc(x)
#         return logits