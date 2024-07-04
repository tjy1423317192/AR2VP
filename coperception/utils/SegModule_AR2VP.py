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

    def step(self, data, num_agent, batch_size,m,t, loss=True):
        bev = data["bev_seq"]
        labels = data["labels"]
#         print(bev.shape)
        bev = bev.permute(0, 3, 1, 2).contiguous()
#         self.optimizer.zero_grad()
#         print(bev.shape)
        if not self.com:
            filtered_bev = []
            filtered_label = []
            for i in range(bev.size(0)):
                if torch.sum(bev[i]) > 1e-4:
                    filtered_bev.append(bev[i])
                    filtered_label.append(labels[i])
            bev = torch.stack(filtered_bev, 0)
            labels = torch.stack(filtered_label, 0)

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
                pred,e = self.model(bev, data["trans_matrices"], data["num_sensor"])
        else:
            pred = self.model(bev)
            
            
#         print(torch.cosine_similarity(t, e[0].flatten(), dim=0))
        t = e[0].flatten()
        
        
        
        
        bev_list = bev.chunk(num_agent, dim=0)
        pred_list = pred.chunk(num_agent, dim=0)
        labels_list = labels.chunk(num_agent, dim=0)
        if self.com:
            filtered_pred = []
            filtered_label = []
            for i in range(num_agent):
                if torch.sum(torch.cat(tuple(bev_list[i]), 0))> 1e-4:
                    filtered_pred.append(pred_list[i])
                    filtered_label.append(labels_list[i])
        if not loss:
            return pred, labels
        grads_list = []
        loss_list = []
        for i in range(len(filtered_pred)):
            self.optimizer.zero_grad()
            pred = filtered_pred[i]
            labels = filtered_label[i]
            kd_loss = (
                self.get_kd_loss(batch_size, data, fused_layer, num_agent, x5, x6, x7)
                if self.kd_flag
                else 0
            )
            loss = self.criterion(pred, labels.long()) + kd_loss

            if isinstance(self.criterion, nn.DataParallel):
                loss = loss.mean()

            loss_data = loss.data.item()
#             print(loss_data)
            if np.isnan(loss_data):
                
                raise ValueError("loss is nan while training")
            loss_list.append(loss)
            loss.backward(retain_graph=True)
            split_grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    split_grads.append(param.grad.clone().detach())
            grads_list.append(split_grads)
#         print("m前：",m)
#         print(len(grads_list[0]))
        m = self.get_mem(m,grads_list)
#         print("m后：",m)
#         print(m)
#         gs = self.ComputeGradient(grads_list)
        
        gs = self.ComputeGradient(grads_list, loss_list, m)
#         print(len(gs))
        self.optimizer.zero_grad()
        loss = sum(loss_list)
        loss.backward()
        params = list(self.model.parameters())
#         print(len(params))
#         print(len(gs))
        for p, g in zip(params, gs):
            p.grad = g
#         for p in param:
#             if p.grad is not None:
#                 p.grad = gs[k]
#                 k = k+1
        self.optimizer.step()     
#         return pred, loss_data, gs,m
        return pred, loss_data,m,t

    def get_mem(self,m,gradients):
        q = 0.9
        p = 0.1
        a = m
        for i in range(len(gradients)): 
            grad_array = gradients[i]
#             print(len(grad_array))
            norm_squared = sum([torch.norm(tensor)**2 for tensor in grad_array])
            norm = torch.sqrt(norm_squared)
#             grad_norm = torch.norm(grad_array)
            a[i] = q*a[i] +p*norm
        return a
    
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


    
    

    def ComputeGradient(self,gradients, losses, gradnorm_mom):

        gs = []
        for i in range(len(gradients)): 
            g_task_flat = torch.cat([grad.reshape(-1) for grad in gradients[i]], 0)
            gs.append(g_task_flat)
        tols = self.ComputeTol(losses, gradnorm_mom)
#         print("tols:",tols)
#         print(sum(tols))
        sol = self.find_min_norm_element_with_tol(gs, tols)
#         print("sol:",sol)
        # if len(gs) > 2:
        #     print(3)
#         print(sum(sol))
        d = []
        if any(x > 0.4 for x in sol):
            for k in range(len(gradients[0])): # for each layer
                g = 0
                for i in range(len(gradients)):
                    g += gradients[i][k]/len(gradients)
                d.append(g)
        else:
            for k in range(len(gradients[0])):
                g = 0
                for i in range(len(gradients)):  # 对每个任务
                    g += sol[i] * gradients[i][k] #/ len(gradients)
                d.append(g)
        return d  
 
    def ComputeTol(self,losses, gradnorm_mom):
        # losses = [torch.from_numpy(mem_losses)] + [torch.from_numpy(loss) for loss in curr_losses] if len(mem_losses) > 0 else [torch.from_numpy(loss) for loss in curr_losses]
        tols = []
        for k in range(len(losses)):
            # assert len(losses[k]) > 0
            tols.append(gradnorm_mom[k])
#         print("tols:",tols)
        tols = torch.tensor(tols, dtype=torch.float64)
        tols = self.softmax(tols/5, 0) # Softmax Temperature 5
        return tols

    def softmax(self,x, axis=None):
        x = x - x.max(dim=axis, keepdim=True).values
        y = torch.exp(x)
        return y / y.sum(dim=axis, keepdim=True)




    def _min_norm_2d_with_tol(self,vecs, dps, tols):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = None
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = torch.sum(torch.mul(vecs[i].view(-1), vecs[j].view(-1))).item()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = torch.sum(torch.mul(vecs[i].view(-1), vecs[i].view(-1))).item()
                if (j, j) not in dps:
                    dps[(j, j)] = torch.sum(torch.mul(vecs[j].view(-1), vecs[j].view(-1))).item()

                c, d = self._min_norm_element_from2_with_tol_v2(dps[(i, i)], dps[(i, j)], dps[(j, j)], tols[i], tols[j])

                if dmin == None:
                    dmin = d
                    sol = [(i, j), c, d]
                else:
                    if d < dmin:
                        dmin = d
                        sol = [(i, j), c, d]
        return sol, dps
#     def _min_norm_2d_with_tol(self,vecs, dps, tols):
#         """
#         Find the minimum norm solution as combination of two points
#         This is correct only in 2D
#         ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
#         """
#         dmin = None
#         if not torch.is_tensor(vecs):
#             vecs = torch.tensor(vecs, dtype=torch.float64)

#         for i in range(len(vecs)):
#             for j in range(i+1,len(vecs)):
#                 if (i,j) not in dps:
#                     dps[(i,j)] = (vecs[i] * vecs[j]).sum().item()
#                     dps[(j, i)] = dps[(i, j)]
#                 if (i,i) not in dps:
#                     dps[(i,i)] = (vecs[i] * vecs[i]).sum().item()
#                 if (j,j) not in dps:
#                     dps[(j,j)] = (vecs[j] * vecs[j]).sum().item()

#                 c,d = _min_norm_element_from2_with_tol_v2(dps[(i,i)], dps[(i,j)], dps[(j,j)], tols[i], tols[j])

#                 if dmin == None:
#                     dmin = d
#                     sol = [(i,j),c,d]
#                 else:
#                     if d < dmin:
#                         dmin = d
#                         sol = [(i,j),c,d]
#         return sol, dps

    def _min_norm_element_from2_with_tol_v2(self,v1v1, v1v2, v2v2, tol1, tol2):

        gamma =  ((tol1 / (tol2*tol2 + 1e-10)) * v2v2 - (1.0 / (tol2 + 1e-10)) * v1v2) / (v1v1 + ((tol1*tol1) / (tol2*tol2 + 1e-10)) * v2v2 - (tol1/(tol2 + 1e-10)) * 2 * v1v2) 
        cost = gamma*gamma*v1v1 + \
            2*gamma*(1-gamma*tol1)*v1v2/(tol2 + 1e-10) + \
            (1.-gamma*tol1)*(1.-gamma*tol1)*v2v2 / (tol2*tol2 + 1e-10)
        return gamma, cost
    
    
#     def _next_point_with_tol_v2(self,cur_val, grad, n, tols, lr):
#         # proj_grad = grad - ( np.sum(grad) / n ) # 一定下降的方向

#         next_point = grad * lr + cur_val
#         # print(cur_val)
#         # print(next_point)
#         # print(proj_grad)
#         # print(t)
#         # print(t*proj_grad)
#         # exit()
#         # _n = next_point
#         # _n = _projection2simplex_with_tol(next_point, tols)
#         return next_point


    def _projection2simplex_with_tol(self,y, tols):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        sorted_idx = np.flip(np.argsort(y), axis=0)
        tmpsum = 0.0
        tmpsum_tol = 0.0
        tmax_f =  (np.sum(np.inner(y, tols)) - 1.0) / np.sum(np.inner(tols,tols))

        for i in sorted_idx[:-1]:        
            tmpsum += y[i] * tols[i] # plus from large to small
            tmpsum_tol += tols[i] * tols[i] # plus from large to small
            tmax =  (tmpsum - 1.) / (tmpsum_tol) #
            if tols[i] * tmax > y[i]: # 基本无法满足条件
                tmax_f = tmax
                break
#         print(type(y))
        x = tmax_f * tols
        x = x.numpy()
        output = np.maximum(y - x, np.zeros(y.shape))
        return output



    def find_min_norm_element_with_tol(self,vecs, tols):
        dps = {}
        init_sol, dps = self._min_norm_2d_with_tol(vecs, dps, tols)
        n = len(vecs)
        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]
        sol_vec = torch.ones(n, dtype=torch.float64) / n
        P = grad_mat
    #         print(P)
        A = np.ones([n], dtype=np.float64)
        q = np.zeros([n], dtype=np.float64)
        b = np.array([1.], dtype=np.float64)
        lb = 0.*np.ones([n], dtype=np.float64)
        sol_method = "osqp"
        sol_method.encode()
        sol_vec = solve_qp(P=P, q=q, A=A, b=b, lb=lb, initvals=sol_vec.numpy(), solver=sol_method)
        for i in range(len(sol_vec)):
            if sol_vec[i] < 0:
                sol_vec[i] = 0
#         print("sol_vec:",sol_vec)
#         print(sum(sol_vec))
        return sol_vec
