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

class AR2VP(SegModelBase):
    def __init__(self, n_channels, n_classes, num_agent=5, compress_level=0, only_v2i=False):
        super().__init__(
            n_channels, n_classes, num_agent=num_agent, compress_level=compress_level, only_v2i=only_v2i
        )
        self.layer_channel = 512
        self.gnn_iter_num = 1
        self.convgru = convrnn.Conv2dGRU(
            in_channels=self.layer_channel * 2,
            out_channels=self.layer_channel,
            kernel_size=3,
            num_layers=1,
            bidirectional=False,
            dilation=1,
            stride=1,
        )
        
        
    def weight_rsu(self,node_feature_list):
        return torch.mean(torch.stack(node_feature_list), dim=0)
    
    
    def weight_ve(self,node_feature_list,weight):
        if weight[0]*(1-1/len(node_feature_list)) > 1/len(node_feature_list):
            a = 1/len(node_feature_list)*node_feature_list[0]
            weight1 = [x * (1-1/len(node_feature_list)) for x in weight]
            for i in range(len(weight1)):
                a = a+weight1[i]*node_feature_list[i+1]
            return a  
        else:
            return torch.mean(torch.stack(node_feature_list), dim=0)
        
#     def bridge(self,node_list,num_agent):
# #         sore = []
#         rsu1 = node_list[-1:].detach()
#         rsu = node_list[-1:].flatten().detach()
#         print(rsu.requires_grad)
#         print(rsu1.requires_grad)
#         other = node_list[:24]
# #         print("rsu:",rsu.size())
#         for i in range(len(other)):
#             a = torch.cosine_similarity(rsu, other[i].flatten(),dim=0)
# #             sore.append(torch.cosine_similarity(rsu, other[i].flatten(),dim=0))
#             if a < 1/num_agent :
#                 other[i] = other[i]+ (1/num_agent - a)*rsu1
#         return other
# #         print("sore:",sore)


    def bridge(self, node_list, num_agent,batch_size):
        rsu1 = node_list[-1:]
        rsu = rsu1.flatten()
        e = num_agent*batch_size
#         print(e)
#         print(rsu.requires_grad)
#         print(rsu1.requires_grad)
#         if node_list.size()[0] < 24 :
        other = node_list[:e].clone().requires_grad_(True)
#         else:
#             other = node_list[:24].clone().requires_grad_(True)
        for i in range(len(other)):
            with torch.no_grad():
                a = torch.cosine_similarity(rsu, other[i].flatten(), dim=0)
            if a < 1/num_agent:
                other[i] = other[i] + (1/num_agent - a) * rsu1
#             print(other[i].requires_grad)
        return other

    
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
        local_com_mat = torch.cat(tuple(feat_list), 1)
        local_com_mat_update = torch.cat(tuple(feat_list), 1)
#         print("x4:",x4.size())
#         rsu = torch.tensor([0], requires_grad=False).to(torch.device('cuda:0'))
        rsu = 0
#         print(rsu.requires_grad)
        for b in range(batch_size):
            com_num_agent = num_agent_tensor[b, 0] 
            agent_feat_list = list() #表示
            for nb in range(self.num_agent):
                agent_feat_list.append(local_com_mat[b, nb])# 将一个batch的所有智能体向量都提到一个列表中
#             print("agent_feat_list[0]:",agent_feat_list[0].size())
            rsu = rsu + agent_feat_list[0]/batch_size
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
                    node_feature_list1 = neighbor_feat_list
                    node_feature_list = torch.stack(node_feature_list1, dim=0)
                    node_feature_list = node_feature_list.view(node_feature_list.size(0), -1)#(6,512*32*32)   
                    weight = []
                    weights = []
                    d = node_feature_list[0]
#                     print("d:",d.size())
                    for i in range(1,len(node_feature_list)):
                        c = node_feature_list[i]
                        weight.append(1-torch.cosine_similarity(d, c,dim=0))
#                     print("weight:",weight)
                    for i in range(len(weight)):
                        if weight[i] < 0:
                            weight[i] = 0
                    for i in range(len(weight)):
                        weights.append(weight[i]/sum(weight))
                    if i ==0 :
                        mean_feat = self.weight_rsu(node_feature_list1)
                    else:
                        mean_feat = self.weight_ve(node_feature_list1,weights)
                    cat_feat = torch.cat([agent_feat_list[i], mean_feat], dim=0)
                    cat_feat = cat_feat.unsqueeze(0).unsqueeze(0)
                    updated_feat, _ = self.convgru(cat_feat, None)
                    updated_feat = torch.squeeze(torch.squeeze(updated_feat, 0), 0)
                    updated_feats_list.append(updated_feat)
                agent_feat_list = updated_feats_list
            for k in range(com_num_agent):
                local_com_mat_update[b, k] = agent_feat_list[k]
#         rsu = rsu.detach()
        rsu = rsu.unsqueeze(0)
#         print(rsu.requires_grad)
        
        feat_mat = super().agents_to_batch(local_com_mat_update)
#         print("feat_mat:",feat_mat.size())
        feat_mat = torch.cat((feat_mat, rsu), dim=0)
        x5 = self.down4(feat_mat)
        x = self.up1(x5, feat_mat)
        other = self.bridge(x,self.num_agent,batch_size)
#         print("x",x.size())
#         x = x[:24]
        x = self.up2(other, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits,agent_feat_list


# import torch #有rsu

# import coperception.utils.convolutional_rnn as convrnn
# from coperception.models.seg.SegModelBase import SegModelBase
# import torch.nn.functional as F
# import os
# import sys
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Linear, LSTM
# from torch.utils.data import DataLoader
# sys.path.append(os.path.dirname(sys.path[0]))
# import numpy as np

# # class TransitiveAttention(nn.Module):
# #     def __init__(self, input_dim, num_heads, hidden_dim):
# #         super(TransitiveAttention, self).__init__()
# #         self.num_heads = num_heads
# #         self.head_dim = hidden_dim // num_heads

# #         self.query = nn.Linear(input_dim, hidden_dim)
# #         self.key = nn.Linear(input_dim, hidden_dim)
# #         self.value = nn.Linear(input_dim, hidden_dim)

# #         self.softmax = nn.Softmax(dim=-1)

# #     def forward(self, x1, x2):#x2是rsu
# #         # 计算查询、键和值
# #         q = self.query(x1)
# #         k = self.key(x2)
# #         v = self.value(x2)

# #         # 将查询、键和值分成多个头并计算注意力得分
# #         q = q.view(q.size(0), self.num_heads, -1, self.head_dim).transpose(1, 2)
# #         k = k.view(k.size(0), self.num_heads, -1, self.head_dim).transpose(1, 2)
# #         v = v.view(v.size(0), self.num_heads, -1, self.head_dim).transpose(1, 2)

# #         attention_scores = torch.matmul(q, k.transpose(-2, -1))

# #         # 归一化得分并计算加权和
# #         attention_weights = self.softmax(attention_scores / self.head_dim**0.5)
# #         attention_output = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(x1.size(0), -1, self.num_heads * self.head_dim)
# # #         print(attention_output.size())

# # #         print(x1.size())
# # #         # 融合传递的特征和原始特征
# # #         fusion_output = (x1 + attention_output) / 2.0  # 取简单平均

# # #         return fusion_output
# #         return attention_output

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
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.a = nn.Linear(1024*16*16,256)
#         self.b = nn.Linear(256,1024*16*16)
#     def weight_rsu(self,node_feature_list):
#         return torch.mean(torch.stack(node_feature_list), dim=0)
    
    
#     def weight_ve(self,node_feature_list,weight):
#         if weight[0]*(1-1/len(node_feature_list)) > 1/len(node_feature_list):
#             a = 1/len(node_feature_list)*node_feature_list[0]
#             weight1 = [x * (1-1/len(node_feature_list)) for x in weight]
#             for i in range(len(weight1)):
#                 a = a+weight1[i]*node_feature_list[i+1]
#             return a  
#         else:
#             return torch.mean(torch.stack(node_feature_list), dim=0)
        
#     def transitive_attention(self,x1, x2, input_dim, num_heads, hidden_dim):
#     # 计算查询、键和值
#         query = nn.Linear(input_dim, hidden_dim).to(self.device)
#         key = nn.Linear(input_dim, hidden_dim).to(self.device)
#         value = nn.Linear(input_dim, hidden_dim).to(self.device)
#         softmax = nn.Softmax(dim=-1).to(self.device)

#         q = query(x1)
#         k = key(x2)
#         v = value(x2)

#         # 将查询、键和值分成多个头并计算注意力得分
#         q = q.view(q.size(0), num_heads, -1, hidden_dim // num_heads).transpose(1, 2)
#         k = k.view(k.size(0), num_heads, -1, hidden_dim // num_heads).transpose(1, 2)
#         v = v.view(v.size(0), num_heads, -1, hidden_dim // num_heads).transpose(1, 2)

#         attention_scores = torch.matmul(q, k.transpose(-2, -1))

#         # 归一化得分并计算加权和
#         attention_weights = softmax(attention_scores / (hidden_dim // num_heads)**0.5)
#         attention_output = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(x1.size(0), -1, num_heads * (hidden_dim // num_heads))

#         return attention_output
    
#     def forward(self, x, trans_matrices, num_agent_tensor):
        
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)  # b 512 32 32
#         size = (1, 512, 32, 32)
#         input_dim = 1024*16*16  # 输入特征的维度
#         num_heads = 4  # 注意力头的数量
#         hidden_dim = 256 # 隐藏维度
        
#         feat_list = []
# #         attention_layer = TransitiveAttention(input_dim, num_heads, hidden_dim).to(self.device)
#         if self.compress_level > 0:
#             x4 = F.relu(self.bn_compress(self.com_compresser(x4)))
#             x4 = F.relu(self.bn_decompress(self.com_decompresser(x4)))

#         batch_size = x.size(0) // self.num_agent
#         feat_list = super().build_feat_list(x4, batch_size)
#         local_com_mat = torch.cat(tuple(feat_list), 1)
#         local_com_mat_update = torch.cat(tuple(feat_list), 1)
#         for b in range(batch_size):
#             com_num_agent = num_agent_tensor[b, 0] 
#             agent_feat_list = list() #表示
#             for nb in range(self.num_agent):
#                 agent_feat_list.append(local_com_mat[b, nb])# 将一个batch的所有智能体向量都提到一个列表中

#             for _ in range(self.gnn_iter_num):
#                 updated_feats_list = list()  #有几层gnn

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
#                     node_feature_list1 = neighbor_feat_list
#                     node_feature_list = torch.stack(node_feature_list1, dim=0)
#                     node_feature_list = node_feature_list.view(node_feature_list.size(0), -1)#(6,512*32*32)   
#                     weight = []
#                     weights = []
#                     d = node_feature_list[0]
#                     for i in range(1,len(node_feature_list)):
#                         c = node_feature_list[i]
#                         weight.append(1-torch.cosine_similarity(d, c,dim=0))
#                     for i in range(len(weight)):
#                         if weight[i] < 0:
#                             weight[i] = 0
#                     for i in range(len(weight)):
#                         weights.append(weight[i]/sum(weight))
#                     if i ==0 :
#                         mean_feat = self.weight_rsu(node_feature_list1)
#                     else:
#                         mean_feat = self.weight_ve(node_feature_list1,weights)
#                     cat_feat = torch.cat([agent_feat_list[i], mean_feat], dim=0)
#                     cat_feat = cat_feat.unsqueeze(0).unsqueeze(0)
#                     updated_feat, _ = self.convgru(cat_feat, None)
#                     updated_feat = torch.squeeze(torch.squeeze(updated_feat, 0), 0)
#                     updated_feats_list.append(updated_feat)
#                 agent_feat_list = updated_feats_list
# #                 print("agent_feat_list[0]:",agent_feat_list[0].size())
# #                 print("agent_feat_list[1]:",agent_feat_list[1].size())
                

            
# #                 input_dim = 512*32*32  # 输入特征的维度
# #                 num_heads = 4  # 注意力头的数量
# #                 hidden_dim = 512 # 隐藏维度
#                 for i in range(len(agent_feat_list)):
#                     agent_feat_list[i] = agent_feat_list[i].view(1, -1, input_dim)                
# #                 attention_layer = TransitiveAttention(input_dim, num_heads, hidden_dim).to(self.device)
#                 concatenated_features = torch.cat(agent_feat_list, dim=1)
#                 rsu = neighbor_feat_list[0].view(1, -1, input_dim) 
#                 fusion_output = self.transitive_attention(concatenated_features, rsu, input_dim, num_heads, hidden_dim)
# #                 fusion_output = fusion_output
# #                 print(fusion_output.size())
                
#                 for i in range(len(agent_feat_list)):
#                     agent_feat_list[i] = self.a(fusion_output[0][i-1]).reshape(512,32,32)
# #                 print("agent_feat_list[0]:",agent_feat_list[0].size())
# #                 print("agent_feat_list[1]:",agent_feat_list[1].size())

                
    
    
    
    
#             for k in range(com_num_agent):
#                 local_com_mat_update[b, k] = agent_feat_list[k]
#         feat_mat = super().agents_to_batch(local_com_mat_update)
#         x5 = self.down4(feat_mat)
        
        
        
        
        
#         for i in range(x5.size(0)):
#             feat_list.append(x5[i].view(1, -1, 1024*16*16))
#         concatenated_features = torch.cat(feat_list, dim=1)
#         rsu = feat_list[0]
#         fusion_output = self.transitive_attention(concatenated_features, rsu, input_dim, num_heads, hidden_dim)
#         for i in range(x5.size(0)):
#             x5[i] = x5[i]*0.8+self.a(fusion_output[0][i]).reshape(1024,16,16)*0.2
        
        
        
        
#         x = self.up1(x5, feat_mat)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits









# import torch #无rsu

# import coperception.utils.convolutional_rnn as convrnn
# from coperception.models.seg.SegModelBase import SegModelBase
# import torch.nn.functional as F
# import os
# import sys
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Linear, LSTM
# from torch.utils.data import DataLoader
# sys.path.append(os.path.dirname(sys.path[0]))
# import numpy as np

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

        
        
#     def weight_rsu(self,node_feature_list):
#         return torch.mean(torch.stack(node_feature_list), dim=0)
    
    
#     def weight_ve(self,node_feature_list,weight):
#         if weight[0]*(1-1/len(node_feature_list)) > 1/len(node_feature_list):
#             a = 1/len(node_feature_list)*node_feature_list[0]
#             weight1 = [x * (1-1/len(node_feature_list)) for x in weight]
#             for i in range(len(weight1)):
#                 a = a+weight1[i]*node_feature_list[i+1]
#             return a  
#         else:
#             return torch.mean(torch.stack(node_feature_list), dim=0)
        

    
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
#         local_com_mat = torch.cat(tuple(feat_list), 1)
#         local_com_mat_update = torch.cat(tuple(feat_list), 1)
#         for b in range(batch_size):
#             com_num_agent = num_agent_tensor[b, 0] 
#             agent_feat_list = list() #表示
#             for nb in range(self.num_agent):
#                 agent_feat_list.append(local_com_mat[b, nb])# 将一个batch的所有智能体向量都提到一个列表中

#             for _ in range(self.gnn_iter_num):
#                 updated_feats_list = list()  #有几层gnn

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
#                     node_feature_list1 = neighbor_feat_list
#                     node_feature_list = torch.stack(node_feature_list1, dim=0)
#                     node_feature_list = node_feature_list.view(node_feature_list.size(0), -1)#(6,512*32*32)   
#                     weight = []
#                     weights = []
#                     d = node_feature_list[0]
#                     for i in range(1,len(node_feature_list)):
#                         c = node_feature_list[i]
#                         weight.append(1-torch.cosine_similarity(d, c,dim=0))
#                     for i in range(len(weight)):
#                         if weight[i] < 0:
#                             weight[i] = 0
#                     for i in range(len(weight)):
#                         weights.append(weight[i]/sum(weight))
# #                     if i ==0 :
# #                         mean_feat = self.weight_rsu(node_feature_list1)
# #                     else:
#                     mean_feat = self.weight_ve(node_feature_list1,weights)
#                     cat_feat = torch.cat([agent_feat_list[i], mean_feat], dim=0)
#                     cat_feat = cat_feat.unsqueeze(0).unsqueeze(0)
#                     updated_feat, _ = self.convgru(cat_feat, None)
#                     updated_feat = torch.squeeze(torch.squeeze(updated_feat, 0), 0)
#                     updated_feats_list.append(updated_feat)
#                 agent_feat_list = updated_feats_list
#             for k in range(com_num_agent):
#                 local_com_mat_update[b, k] = agent_feat_list[k]

#         feat_mat = super().agents_to_batch(local_com_mat_update)
#         x5 = self.down4(feat_mat)
#         x = self.up1(x5, feat_mat)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits










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