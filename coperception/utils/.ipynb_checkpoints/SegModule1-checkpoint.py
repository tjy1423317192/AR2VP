import torch.nn.functional as F
import torch.nn as nn
import torch
from coperception.utils.detection_util import *
from qpsolvers import solve_qp

class SegModule(object):
    def __init__(self, model, teacher, config, optimizer, kd_flag):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.nepoch
        )
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss()
        self.teacher = teacher
        if kd_flag:
            for k, v in self.teacher.named_parameters():
                v.requires_grad = False  # fix parameters

        self.kd_flag = kd_flag

        self.com = config.com

    def resume(self, path):
        def map_func(storage, location):
            return storage.cuda()

        if os.path.isfile(path):
            if rank == 0:
                print("=> loading checkpoint '{}'".format(path))

            checkpoint = torch.load(path, map_location=map_func)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

            ckpt_keys = set(checkpoint["state_dict"].keys())
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print("caution: missing keys from checkpoint {}: {}".format(path, k))
        else:
            print("=> no checkpoint found at '{}'".format(path))

    def step(self, data, num_agent, batch_size, loss=True):
        bev_all = data["bev_seq"]
        print(bev_all.size())
        labels_all = data["labels"]
        bev_list = torch.chunk(bev_all, chunks=6, dim=0)
        labels_list = torch.chunk(labels_all, chunks=6, dim=0)
        grads_list = []
        loss_list = []
#         grad_list = []
#         for _ in range(6):
#             grad_list.append([])        
        for j in range(6):
            bev = bev_list[j]
            labels = labels_list[j]
            self.optimizer.zero_grad()
            bev = bev.permute(0, 3, 1, 2).contiguous()
#             print("bev:",bev.size())
#             print("labels:",labels.size())
#             bev = bev.repeat(6,1,1,1)
#             print(bev.size())
#             labels = labels.repeat(6,1,1)
            if not self.com:
                filtered_bev = []
                filtered_label = []
                for i in range(bev.size(0)):
                    print(bev.size(0))
                    if torch.sum(bev[i]) > 1e-4:
                        filtered_bev.append(bev[i])
                        filtered_label.append(labels[i])
                bev = torch.stack(filtered_bev, 0)
                labels = torch.stack(filtered_label, 0)
#                 print("bev_:",bev.size())
#                 print("labels_:",labels.size())
            if self.kd_flag:
                data["bev_seq_teacher"] = (
                    data["bev_seq_teacher"].permute(0, 3, 1, 2).contiguous()
                )

            if self.com:
                if self.kd_flag:
                    pred, x9, x8, x7, x6, x5, fused_layer = self.model(
                        bev, data["trans_matrices"], data["num_sensor"]
                    )
                elif self.config.flag.startswith("when2com") or self.config.flag.startswith(
                    "who2com"
                ):
                    if self.config.split == "train":
                        pred = self.model(
                            bev, data["trans_matrices"], data["num_sensor"], training=True
                        )
                    else:
                        pred = self.model(
                            bev,
                            data["trans_matrices"],
                            data["num_sensor"],
                            inference=self.config.inference,
                            training=False,
                        )
                else:
                    pred = self.model(bev, data["trans_matrices"], data["num_sensor"])
            else:
                pred = self.model(bev)

            if self.com:
                filtered_pred = []
                filtered_label = []
#                 print("bev[0]:",bev.size(0))
                for i in range(bev.size(0)):
#                     print(i)
                    if torch.sum(bev[i]) > 1e-4:
                        filtered_pred.append(pred[i])
                        filtered_label.append(labels[i])
#                 print("filtered_pred.size:",len(filtered_pred))
                if len(filtered_pred) > 0:
                    pred = torch.stack(filtered_pred, 0)
                    labels = torch.stack(filtered_label, 0)

            if not loss:
                return pred, labels

            kd_loss = (
                self.get_kd_loss(batch_size, data, fused_layer, num_agent, x5, x6, x7)
                if self.kd_flag
                else 0
            )
            if len(filtered_pred) > 0:
                loss = self.criterion(pred, labels.long()) + kd_loss

                if isinstance(self.criterion, nn.DataParallel):
                    loss = loss.mean()

                loss_data = loss.data.item()
                if np.isnan(loss_data):
                    raise ValueError("loss is nan while training")
                loss_list.append(loss)
#                 print(loss_list[j])
                loss.backward(retain_graph=True)
                split_grads = []
#                 for name, param in self.model.named_parameters():
#                     if param.grad is not None:
#                          split_grads[name] = param.grad.clone().detach()
#                 grads_list.append(split_grads)   
                for param in self.model.parameters():
                    if param.grad is not None:
                        split_grads.append(param.grad.clone().detach())
                grads_list.append(split_grads) 
                
#             else:
#                 loss_list[j] = 0
#                 grads_list.append(torch.tensor(0))
#         for i in range(len(grads_list)):
#             if grads_list[i]!=torch.tensor(0):
#                 for v in grads_list[i].values():
#                     grad_list[i].append(v)

#         grad_list = [g for g in grad_list if g]
#             else:
#                 grad_list[i].append(torch.tensor(0))
                
#         print(len(grad_list))
#         print("梯度本来有多少层",len(grad_list[0]))
#         loss_list = [g for g in grad_list if g]
#         print(len(loss_list))
#         print("loss列表：",loss_list)
#         print(len(grads_list))
        gs = self.ComputeGradient(grads_list)
#         print("改后的梯度列表长度",len(gs))
        self.optimizer.zero_grad()        
        loss = sum(loss_list)
        loss.backward()
        params = list(self.model.parameters())
        for p, g in zip(params, gs):
            p.grad = g
        self.optimizer.step()
        return pred, loss_data, gs
    
    
    
    def _min_norm_element_from2(self,v1v1, v1v2, v2v2):
            """
            Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
            d is the distance (objective) optimzed
            v1v1 = <x1,x1>
            v1v2 = <x1,x2>
            v2v2 = <x2,x2>
            """
            if v1v2 >= v1v1:
                # Case: Fig 1, third column
                gamma = 0.999
                cost = v1v1
                return gamma, cost
            if v1v2 >= v2v2:
                # Case: Fig 1, first column
                gamma = 0.001
                cost = v2v2
                return gamma, cost
            # Case: Fig 1, second column
            gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
            cost = v2v2 + gamma*(v1v2 - v2v2)
            return gamma, cost
        
        
        
#     def _min_norm_2d(self,vecs, dps):
#         """
#         Find the minimum norm solution as combination of two points
#         This is correct only in 2D
#         ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
#         """
# #         print("进入_min_norm_2d")
#         dmin = 1e8
#         for i in range(len(vecs)):
#             for j in range(i+1,len(vecs)):
#                 if (i,j) not in dps:
#                     dps[(i, j)] = 0.0
#                     for k in range(len(vecs[i])):
# #                         print(len(vecs[i]))
# #                         print("计算ing")
#                         dps[(i,j)] += torch.mul(vecs[i][k], vecs[j][k]).sum()
#                     dps[(j, i)] = dps[(i, j)]
#                 if (i,i) not in dps:
#                     dps[(i, i)] = 0.0
#                     for k in range(len(vecs[i])):
#                         dps[(i,i)] += torch.mul(vecs[i][k], vecs[i][k]).sum()
#                 if (j,j) not in dps:
#                     dps[(j, j)] = 0.0   
#                     for k in range(len(vecs[i])):
#                         dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum()
#                 c,d = self._min_norm_element_from2(dps[(i,i)], dps[(i,j)], dps[(j,j)])
#                 if d < dmin:
#                     dmin = d
#                     sol = [(i,j),c,d]
#         return sol, dps
    def _min_norm_2d(self,vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i+1,len(vecs)):
                if (i,j) not in dps:
                    dps[(i,j)] = torch.sum(torch.mul(vecs[i].view(-1), vecs[j].view(-1))).item()
                    dps[(j, i)] = dps[(i, j)]
                if (i,i) not in dps:
                    dps[(i,i)] = torch.sum(torch.mul(vecs[i].view(-1), vecs[i].view(-1))).item()
                if (j,j) not in dps:
                    dps[(j,j)] = torch.sum(torch.mul(vecs[j].view(-1), vecs[j].view(-1))).item()
                c,d = self._min_norm_element_from2(dps[(i,i)], dps[(i,j)], dps[(j,j)])
                if d < dmin:
                    dmin = d
                    sol = [(i,j),c,d]
        return sol, dps

#     def _min_norm_2d(self, vecs, dps):
#         """
#         Find the minimum norm solution as combination of two points
#         This is correct only in 2D
#         ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
#         """
#         print("进入_min_norm_2d")
#         dmin = 1e8
#         # 使用 PCA 将 vecs 中的每个列表进行降维到 128 维
# #         vecs_pca = []
# #         for vec in vecs:
# #             x = torch.tensor(vec, dtype=torch.float64)
# #             x = x.unsqueeze(0)            
# #             print(x.size())
# #             z,_,_= torch.pca_lowrank(x, q=128)
# #             print(z.size())
# #             vecs_pca.append(z)
# #             print("降维一个")
#         # 计算点积矩阵
#         for i in range(len(vecs_pca)):
#             for j in range(i + 1, len(vecs_pca)):
#                 if (i, j) not in dps:
#                     dps[(i, j)] = 0.0
#                     for k in range(vecs_pca[i].shape[1]):
#                         dps[(i, j)] += torch.dot(vecs_pca[i][:, k], vecs_pca[j][:, k])
#                     dps[(j, i)] = dps[(i, j)]
#                 if (i, i) not in dps:
#                     dps[(i, i)] = 0.0
#                     for k in range(vecs_pca[i].shape[1]):
#                         dps[(i, i)] += torch.dot(vecs_pca[i][:, k], vecs_pca[i][:, k])
#                 if (j, j) not in dps:
#                     dps[(j, j)] = 0.0
#                     for k in range(vecs_pca[j].shape[1]):
#                         dps[(j, j)] += torch.dot(vecs_pca[j][:, k], vecs_pca[j][:, k])
#                 # 计算最小范数解
#                 c, d = _min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
#                 if d < dmin:
#                     print("sol有了")
#                     dmin = d
#                     sol = [(i, j), c, d]
#         return sol, dps


    def find_min_norm_element_independent(self,vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """

        # Solution lying at the combination of two points
        dps = {}
#         print("find_min_norm_element_independent正在运行")
        init_sol, dps = self._min_norm_2d(vecs, dps)
#         print("_min_norm_2d能用")
        n = len(vecs)

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]
#         print(grad_mat)
        sol_vec = torch.ones(n, dtype=torch.float64) / n
        P = grad_mat
        A = np.ones([n], dtype=np.float64)
        q = np.zeros([n], dtype=np.float64)
        b = np.array([1.], dtype=np.float64)
        lb = 0.*np.ones([n], dtype=np.float64)
        sol_method = "quadprog"
        sol_method.encode()
        sol_vec = solve_qp(P=P, q=q, A=A, b=b, lb=lb, initvals=sol_vec.numpy(), solver=sol_method)
        return torch.tensor(sol_vec, dtype=torch.float64)    
    
#     def ComputeGradient(self,gradients):
#         # 1. Flat gradients
#         gs = []
#         gradients1 = []
#         for _ in range(len(gradients)):
#             gradients1.append([])        
# #         for i in range(len(gradients)):
# #             gradients1.append(gradients[i][:10])
# # #         print("gradients1[0]长度：",len(gradients1[0]))
# # #         largest_tensor = max(gradients1[0], key=lambda x: x.numel())
# # #         print("最大size：",largest_tensor.size())
# #         for i in range(len(gradients1)): # for each task
# #             g_task_flat = torch.cat([grad.reshape(-1) for grad in gradients1[i]], 0)
# #             gs.append(g_task_flat)
# #         # 2. Compute the weight
#         for i in range(len(gradients)):
#             for j in range(len(gradients[i])):
#                 x = torch.norm(gradients[i][j], p='fro')
#                 gradients1[i].append(x)
# #         print("gradients1长度：",len(gradients1))
# #         print("gradients1[0]长度：",len(gradients1[0]))
#         for i in range(len(gradients1)): # for each task
#             g_task_flat = torch.cat([grad.reshape(-1) for grad in gradients1[i]], 0)
#             gs.append(g_task_flat)
#         weights = self.distangle_optimize_weight_with_distance(gs)
# #         print(len(weights))
# #         print(weights)
#         # 3. Obtain the final gradient
#         d = []
# #         print("gradients[0]长度：",len(gradients[0]))
# #         print("gradients长度：",len(gradients))
#         for k in range(len(gradients[0])): # for each layer
#             g = 0
#             for i in range(len(gradients)):
#                 g += weights[i]*gradients[i][k]
#             d.append(g)
# #         for i in range(len(gradients1)):
# #             for j in range(len(gradients1[i])):
# #                 g = weights[i]*gradients1[i][j]
# #         print("d:",len(d))
#         return d



    def ComputeGradient(self,gradients):
        # 1. Flat gradients
        gs = []                   
        for i in range(len(gradients)): # for each task
            g_task_flat = torch.cat([grad.reshape(-1) for grad in gradients[i]], 0)
            gs.append(g_task_flat)
        weights = self.distangle_optimize_weight_with_distance(gs)
        d = []
        for k in range(len(gradients[0])): # for each layer
            g = 0
            for i in range(len(gradients)):
                g += weights[i]*gradients[i][k]
            d.append(g)
        return d


    def distangle_optimize_weight_with_distance(self,grads):
        """optimize the distance for all independent grad
        grads: List, all flat gradients
        """
        W = torch.ones((len(grads), len(grads)), dtype=torch.float64) / (len(grads)-1)
        W.requires_grad = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # compute the masks
        W_diag_mask = torch.eye(len(grads))
        W_mask = 1 - W_diag_mask

        optimizer = torch.optim.SGD([W], lr=0.001)

        if len(grads) > 2:
            # update off-diagonal
            for _ in range(5):
                optimizer.zero_grad()
                # compute total loss
                masked_W = self.masked_softmax(W,W_mask)
#                 print("masked_W:",masked_W.size())
#                 print("grads[0].size:",grads[0].size())
                
#                 print("停一下")
                for i in range(len(grads)):
                    if len(grads[i]) != len(grads[0]):
                        grads[i] = torch.zeros([len(grads[0])])
#                     if grads[i].size() != torch.Size([118592]):
#                         grads[i] = torch.zeros([118592])
#                 print(grads[0].size())
#                 print(grads[1].size())
#                 print(grads[4].size())
                masked_W = masked_W.double().to(device)
    #             G_combine = G_combine.to(device)
                for i in range(len(grads)):
                    grads[i] = grads[i].to(device)
                a = torch.stack(grads, dim=0)
                a = a.double().to(device)
                G_combine = torch.matmul(masked_W, a).sum(dim=0)
#                 print(G_combine.size())
    #             G_combine = torch.matmul(masked_W, cat1).sum(dim=0)
                ##########################################################
                # AGD
                maxdo_loss = 0.1 * self.asymmetric_distance(G_combine, a)
                ##########################################################
                maxdo_loss.backward()
                optimizer.step()
        # update diagonal
        W_diag = self.find_min_norm_element_independent(grads)

        if len(grads) > 2:
            W = self.masked_softmax(W, W_mask) + torch.diag(W_diag)
            return W.sum(dim=0) / (len(grads)+1)
        else:  # only for two tasks
            W = torch.diag(W_diag)
            return F.softmax(W.sum(dim=0) + 1., dim=-1)


    # def asymmetric_distance(x, y):
    #     """The proposed distance
    #     rad(x,y) -> Int >=0
    #     """
    #     dist = torch.norm(x - y, p=2, dim=-1)
    #     dist = dist / (dist + torch.norm(y, p=2, dim=-1))
    #     dist = torch.mean(dist)
    #     return dist
    def asymmetric_distance(self,x, y):
        """
        The proposed distance rad(x,y) -> Int >=0
        """
        dist = torch.norm(x - y, dim=-1)
        dist = dist / (dist + torch.norm(y, dim=-1))
        dist = torch.mean(dist)
        return dist



    def masked_softmax(self,scores, mask):
        scores = scores - torch.max(scores, dim=1, keepdim=True)[0]
        exp_scores = torch.exp(scores)
        exp_scores = exp_scores * mask
        exp_sum_scores = torch.sum(exp_scores, dim=1, keepdim=True)
        return exp_scores / (exp_sum_scores.repeat(1, exp_scores.shape[1]) + 1e-7)    
#     def step(self, data, num_agent, batch_size, loss=True):
#         bev = data["bev_seq"]
#         labels = data["labels"]
#         self.optimizer.zero_grad()
#         bev = bev.permute(0, 3, 1, 2).contiguous()
#         print("bev:",bev.size())
#         print("labels:",labels.size())
#         if not self.com:
#             filtered_bev = []
#             filtered_label = []
#             for i in range(bev.size(0)):
#                 if torch.sum(bev[i]) > 1e-4:
#                     filtered_bev.append(bev[i])
#                     filtered_label.append(labels[i])
#             bev = torch.stack(filtered_bev, 0)
#             labels = torch.stack(filtered_label, 0)
#         if self.kd_flag:
#             data["bev_seq_teacher"] = (
#                 data["bev_seq_teacher"].permute(0, 3, 1, 2).contiguous()
#             )

#         if self.com:
#             if self.kd_flag:
#                 pred, x9, x8, x7, x6, x5, fused_layer = self.model(
#                     bev, data["trans_matrices"], data["num_sensor"]
#                 )
#             elif self.config.flag.startswith("when2com") or self.config.flag.startswith(
#                 "who2com"
#             ):
#                 if self.config.split == "train":
#                     pred = self.model(
#                         bev, data["trans_matrices"], data["num_sensor"], training=True
#                     )
#                 else:
#                     pred = self.model(
#                         bev,
#                         data["trans_matrices"],
#                         data["num_sensor"],
#                         inference=self.config.inference,
#                         training=False,
#                     )
#             else:
#                 pred = self.model(bev, data["trans_matrices"], data["num_sensor"])
#         else:
#             pred = self.model(bev)

#         if self.com:
#             filtered_pred = []
#             filtered_label = []
#             for i in range(bev.size(0)):
#                 if torch.sum(bev[i]) > 1e-4:
#                     filtered_pred.append(pred[i])
#                     filtered_label.append(labels[i])
#             pred = torch.stack(filtered_pred, 0)
#             labels = torch.stack(filtered_label, 0)
#         if not loss:
#             return pred, labels

#         kd_loss = (
#             self.get_kd_loss(batch_size, data, fused_layer, num_agent, x5, x6, x7)
#             if self.kd_flag
#             else 0
#         )
#         loss = self.criterion(pred, labels.long()) + kd_loss

#         if isinstance(self.criterion, nn.DataParallel):
#             loss = loss.mean()

#         loss_data = loss.data.item()
#         if np.isnan(loss_data):
#             raise ValueError("loss is nan while training")

#         loss.backward()
#         self.optimizer.step()

#         return pred, loss_data
    def get_kd_loss(self, batch_size, data, fused_layer, num_agent, x5, x6, x7):
        if not self.kd_flag:
            return 0

        bev_seq_teacher = data["bev_seq_teacher"].type(torch.cuda.FloatTensor)
        kd_weight = data["kd_weight"]
        (
            logit_teacher,
            x9_teacher,
            x8_teacher,
            x7_teacher,
            x6_teacher,
            x5_teacher,
            x4_teacher,
        ) = self.teacher(bev_seq_teacher)
        kl_loss_mean = nn.KLDivLoss(size_average=True, reduce=True)

        target_x5 = x5_teacher.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 16 * 16, -1
        )
        student_x5 = x5.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 16 * 16, -1
        )
        kd_loss_x5 = kl_loss_mean(
            F.log_softmax(student_x5, dim=1), F.softmax(target_x5, dim=1)
        )

        target_x6 = x6_teacher.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 32 * 32, -1
        )
        student_x6 = x6.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 32 * 32, -1
        )
        kd_loss_x6 = kl_loss_mean(
            F.log_softmax(student_x6, dim=1), F.softmax(target_x6, dim=1)
        )

        target_x7 = x7_teacher.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 64 * 64, -1
        )
        student_x7 = x7.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 64 * 64, -1
        )
        kd_loss_x7 = kl_loss_mean(
            F.log_softmax(student_x7, dim=1), F.softmax(target_x7, dim=1)
        )

        target_x4 = x4_teacher.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 32 * 32, -1
        )
        student_x4 = fused_layer.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 32 * 32, -1
        )
        kd_loss_fused_layer = kl_loss_mean(
            F.log_softmax(student_x4, dim=1), F.softmax(target_x4, dim=1)
        )

        return kd_weight * (kd_loss_x5 + kd_loss_x6 + kd_loss_x7 + kd_loss_fused_layer)
