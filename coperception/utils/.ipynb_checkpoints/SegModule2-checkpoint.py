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
        bev = data["bev_seq"]
        labels = data["labels"]
#         print(bev.shape)
        bev = bev.permute(0, 3, 1, 2).contiguous()
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
                pred = self.model(bev, data["trans_matrices"], data["num_sensor"])
        else:
            pred = self.model(bev)
#         print(len(pred))
#         print(len(pred[0]))
        bev_list = bev.chunk(6, dim=0)
        pred_list = pred.chunk(6, dim=0)
        labels_list = labels.chunk(6, dim=0)
        if self.com:
            filtered_pred = []
            filtered_label = []
            for i in range(6):
                if torch.sum(torch.cat(tuple(bev_list), 0))> 1e-4:
                    filtered_pred.append(pred_list[i])
                    filtered_label.append(labels_list[i])
#             print("filtered_pred:",len(filtered_pred))
#             print("filtered_pred[0]:",filtered_pred[0].size())
#             pred = torch.stack(filtered_pred, 0)
#             print("pred:",pred.size())
#             labels = torch.stack(filtered_label, 0)
#             print("labels:",labels.size())
#             print("labels[0]:",labels[0].size())
        if not loss:
            return pred, labels

#         kd_loss = (
#             self.get_kd_loss(batch_size, data, fused_layer, num_agent, x5, x6, x7)
#             if self.kd_flag
#             else 0
#         )
#         loss = self.criterion(pred, labels.long()) + kd_losss

#         if isinstance(self.criterion, nn.DataParallel):
#             loss = loss.mean()

#         loss_data = loss.data.item()
#         if np.isnan(loss_data):
#             raise ValueError("loss is nan while training")

#         loss.backward()
#         self.optimizer.step()
        grads_list = []
        loss_list = []
#         print(len(filtered_pred))
#         print(len(filtered_pred[0]))
        for i in range(len(filtered_pred)):
            pred = filtered_pred[i]
            labels = filtered_label[i]
#             labels = labels.unsqueeze(0)
#             pred = pred.unsqueeze(0)
            kd_loss = (
                self.get_kd_loss(batch_size, data, fused_layer, num_agent, x5, x6, x7)
                if self.kd_flag
                else 0
            )
            loss = self.criterion(pred, labels.long()) + kd_loss

            if isinstance(self.criterion, nn.DataParallel):
                loss = loss.mean()

            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError("loss is nan while training")
            loss_list.append(loss)
#             loss.backward(retain_graph=True)
#             split_grads = []
#             for param in self.model.parameters():
#                 if param.grad is not None:
#                     split_grads.append(param.grad.clone().detach())
#             grads_list.append(split_grads)


        self.optimizer.zero_grad()        
        loss = loss_list[0]
        loss.backward(retain_graph=True)
        split_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                split_grads.append(param.grad.clone().detach())
        grads_list.append(split_grads)
        
        self.optimizer.zero_grad()
        loss = sum(loss_list[1:])
        loss.backward(retain_graph=True)
        split_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                split_grads.append(param.grad.clone().detach())
        grads_list.append(split_grads)
        
        
        gs = self.ComputeGradient(grads_list)
        
        
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
        
        
    def _min_norm_2d(self,vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e9
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
#                 print(d)
                if d < dmin:
                    dmin = d
                    sol = [(i,j),c,d]
        return sol, dps    
    
    def find_min_norm_element_independent(self,vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """

        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = self._min_norm_2d(vecs, dps)
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
        return torch.tensor(sol_vec, dtype=torch.float64)   
    
    
    
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
    
    def ComputeGradient1(self,gradients):
        d = []
        for k in range(len(gradients[0])): # for each layer
            g = 0
            for i in range(len(gradients)):
                g += gradients[i][k]/len(gradients)
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

        optimizer = torch.optim.SGD([W], lr=0.002)

        if len(grads) > 2:
            # update off-diagonal
            for _ in range(5):
                optimizer.zero_grad()
                # compute total loss
                masked_W = self.masked_softmax(W,W_mask)
                for i in range(len(grads)):
                    if len(grads[i]) != len(grads[0]):
                        grads[i] = torch.zeros([len(grads[0])])
                masked_W = masked_W.double().to(device)
                for i in range(len(grads)):
                    grads[i] = grads[i].to(device)
                a = torch.stack(grads, dim=0)
                a = a.double().to(device)
                G_combine = torch.matmul(masked_W, a).sum(dim=0)
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