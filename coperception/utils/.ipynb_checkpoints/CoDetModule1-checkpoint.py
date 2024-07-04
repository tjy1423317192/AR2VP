import torch.nn as nn
from coperception.utils.detection_util import *
from coperception.utils.min_norm_solvers import MinNormSolver
import numpy as np
import torch
import torch.nn.functional as F
from qpsolvers import solve_qp


class FaFModule(object):
    def __init__(self, model, teacher, config, optimizer, criterion, kd_flag):
        self.MGDA = config.MGDA
        if self.MGDA:
            self.encoder = model[0]
            self.head = model[1]
            self.optimizer_encoder = optimizer[0]
            self.optimizer_head = optimizer[1]
            self.scheduler_encoder = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_encoder, milestones=[50, 100, 150, 200], gamma=0.5
            )
            self.scheduler_head = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_head, milestones=[50, 100, 150, 200], gamma=0.5
            )
            self.MGDA = config.MGDA
        else:
            self.model = model
            self.kd_flag = kd_flag
            if self.kd_flag == 1:
                self.teacher = teacher
                for k, v in self.teacher.named_parameters():
                    v.requires_grad = False  # fix parameters
            self.optimizer = optimizer
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[50, 100, 150, 200], gamma=0.5
            )
        self.criterion = criterion  # {'cls_loss','loc_loss'}

        self.out_seq_len = config.pred_len
        self.category_num = config.category_num
        self.code_size = config.box_code_size
        self.loss_scale = None

        self.code_type = config.code_type
        self.loss_type = config.loss_type
        self.pred_len = config.pred_len
        self.only_det = config.only_det
        if self.code_type in ["corner_1", "corner_2", "corner_3"]:
            self.alpha = 1.0
        elif self.code_type == "faf":
            if self.loss_type == "corner_loss":
                self.alpha = 1.0
                if not self.only_det:
                    self.alpha = 1.0
            else:
                self.alpha = 0.1
        self.config = config

    def resume(self, path):
        def map_func(storage, location):
            return storage.cuda()

        if os.path.isfile(path):
            # FIXME: where does "rank" come from?
            if rank == 0:
                print("=> loading checkpoint '{}'".format(path))

            checkpoint = torch.load(path, map_location=map_func)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

            ckpt_keys = set(checkpoint["state_dict"].keys())

            # FIXME: "modeL" is not defined here
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print("caution: missing keys from checkpoint {}: {}".format(path, k))
        else:
            print("=> no checkpoint found at '{}'".format(path))




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
        dmin = 5e15
#         print("vecs[0]:",len(vecs[0]))
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
#                     print("有了sol")
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
#         print("sol_vec:",sol_vec)
        P = grad_mat
        A = np.ones([n], dtype=np.float64)
        q = np.zeros([n], dtype=np.float64)
        b = np.array([1.], dtype=np.float64)
        lb = 0.*np.ones([n], dtype=np.float64)
#         print("P:",P)
        sol_method = "osqp"
        sol_method.encode()
        sol_vec = solve_qp(P=P, q=q, A=A, b=b, lb=lb, initvals=sol_vec.numpy(), solver=sol_method)
#         print(sol_vec)
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
    
    

    def corner_loss(self, anchors, reg_loss_mask, reg_targets, pred_result):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reg_targets_list = torch.chunk(reg_targets, chunks=6, dim=0)
        pred_result_list = torch.chunk(pred_result, chunks=6, dim=0)
        anchors_list = torch.chunk(anchors, chunks=6, dim=0)
        reg_loss_mask_list = torch.chunk(reg_loss_mask, chunks=6, dim=0)
        loss_list = []
        loss_sum = 0
#         print(anchors.size())
        for i in range(6):
        # 获取第 i 个 agent 的张量
            reg_targets_i = reg_targets_list[i]
            pred_result_i = pred_result_list[i]    
            N = pred_result_i.shape[0]
            anchors = anchors_list[i].unsqueeze(-2).expand(
                anchors_list[i].shape[0],
                anchors_list[i].shape[1],
                anchors_list[i].shape[2],
                anchors_list[i].shape[3],
                reg_loss_mask_list[i].shape[-1],
                anchors_list[i].shape[-1],
            )
#             print("reg_targets:",reg_targets_i.size())
#             print("pred_result:",pred_result_i.size())
#             print(N)
            assigned_anchor = anchors[reg_loss_mask_list[i]]
            assigned_target = reg_targets_i[reg_loss_mask_list[i]]
            assigned_pred = pred_result_i[reg_loss_mask_list[i]]
                # print(assigned_anchor.shape,assigned_pred.shape,assigned_target.shape)
                # exit()
            pred_decode = bev_box_decode_torch(assigned_pred, assigned_anchor)
            target_decode = bev_box_decode_torch(assigned_target, assigned_anchor)
            pred_corners = center_to_corner_box2d_torch(
                pred_decode[..., :2], pred_decode[..., 2:4], pred_decode[..., 4:]
            )
            target_corners = center_to_corner_box2d_torch(
                target_decode[..., :2], target_decode[..., 2:4], target_decode[..., 4:]
            )
#             print("target_corners:",target_corners.size())
#             print("pred_corner:",pred_corners.size())
            loss_loc = torch.sum(torch.norm(pred_corners - target_corners, dim=-1)) / N
            loss_list.append(loss_loc)
        loss_sum = sum(loss_list)

        return loss_sum,loss_list
    
    
    def loss_calculator(
        self,
        result,
        anchors,
        reg_loss_mask,
        reg_targets,
        labels,
        N,
        motion_labels=None,
        motion_mask=None,
    ):
        loss_num = 0
        loss_cls_list = []
        cls_list = self.criterion["cls"](result["cls"], labels)
        for i in range(6):
            loss_cls_list.append(torch.sum(cls_list[i]))
        loss_cls = sum(loss_cls_list)
        loss_num += 1

        if motion_labels is not None:
            loss_motion = (
                torch.sum(self.criterion["cls"](result["state"], motion_labels)) / N
            )
            loss_num += 1

        # loss_mask_num = torch.nonzero(
        #     reg_loss_mask.view(-1, reg_loss_mask.shape[-1])
        # ).size(0)
        # print(loss_mask_num)
        # print(torch.sum(reg_targets[:,:,:,:,0][reg_loss_mask[:,:,:,:,2]]))

        if self.code_type in ["corner_1", "corner_2", "corner_3"]:
            target = reg_targets[reg_loss_mask].reshape(-1, 5, 2)
            flip_target = torch.stack(
                [target[:, 0], target[:, 3], target[:, 4], target[:, 1], target[:, 2]],
                dim=-2,
            )
            pred = result["loc"][reg_loss_mask].reshape(-1, 5, 2)
            t = torch.sum(torch.norm(pred - target, dim=-1), dim=-1)
            f = torch.sum(torch.norm(pred - flip_target, dim=-1), dim=-1)
            loss_loc = torch.sum(torch.min(t, f)) / N

        elif self.code_type == "faf":
            if self.loss_type == "corner_loss":
                if self.only_det:
                    loss_loc,loss_loc_list= self.corner_loss(
                        anchors, reg_loss_mask, reg_targets, result["loc"]
                    )
                    loss_num += 1
                elif self.config.pred_type in ["motion", "center"]:

                    # only center/motion for pred

                    loss_loc_1,loss_loc1_list = self.corner_loss(
                        anchors,
                        reg_loss_mask[..., 0][..., [0]],
                        reg_targets[..., [0], :],
                        result["loc"][..., [0], :],
                    )
                    pred_reg_loss_mask = reg_loss_mask[..., 1:, :]
                    if self.config.motion_state:
                        pred_reg_loss_mask = motion_mask  # mask out static object
                    loss_loc_2,loss_loc2_list = F.smooth_l1_loss(
                        result["loc"][..., 1:, :][pred_reg_loss_mask],
                        reg_targets[..., 1:, :][pred_reg_loss_mask],
                    )
                    loss_loc = loss_loc_1 + loss_loc_2
                    loss_num += 2

                # corners for pred
                else:
                    loss_loc = self.corner_loss(
                        anchors, reg_loss_mask, reg_targets, result["loc"]
                    )
                    loss_num += 1
            else:

                loss_loc = F.smooth_l1_loss(
                    result["loc"][reg_loss_mask], reg_targets[reg_loss_mask]
                )
                loss_num += 1

        if self.loss_scale is not None:
            if len(self.loss_scale) == 4:
                loss = (
                    self.loss_scale[0] * loss_cls
                    + self.loss_scale[1] * loss_loc_1
                    + self.loss_scale[2] * loss_loc_2
                    + self.loss_scale[3] * loss_motion
                )
            elif len(self.loss_scale) == 3:
                loss = (
                    self.loss_scale[0] * loss_cls
                    + self.loss_scale[1] * loss_loc_1
                    + self.loss_scale[2] * loss_loc_2
                )
            else:
                loss = self.loss_scale[0] * loss_cls + self.loss_scale[1] * loss_loc
        elif motion_labels is not None:
            loss = loss_cls + loss_loc + loss_motion
        else:
            loss = loss_cls + loss_loc

        if loss_num == 2:
            return loss_num, loss, loss_cls, loss_loc, loss_loc_list, loss_cls_list
        elif loss_num == 3:
            return loss_num, loss, loss_cls, loss_loc_1, loss_loc_2
        elif loss_num == 4:
            return loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion    
    
    


    def step(self, data, batch_size, num_agent):

        bev_seq = data["bev_seq"]
        labels = data["labels"]
        reg_targets = data["reg_targets"]
        reg_loss_mask = data["reg_loss_mask"]
        anchors = data["anchors"]
        trans_matrices = data["trans_matrices"]
        num_all_agents = data["num_agent"]

        # with torch.autograd.set_detect_anomaly(True):
        if self.config.flag.startswith("when2com") or self.config.flag.startswith(
            "who2com"
        ):
            if self.config.split == "train":
                result = self.model(
                    bev_seq,
                    trans_matrices,
                    num_all_agents,
                    batch_size=batch_size,
                    training=True,
                )
            else:
                result = self.model(
                    bev_seq,
                    trans_matrices,
                    num_all_agents,
                    batch_size=batch_size,
                    inference=self.config.inference,
                    training=False,
                )
        else:
            if self.kd_flag:
                result, x_8, x_7, x_6, x_5, fused_layer = self.model(
                    bev_seq, trans_matrices, num_all_agents, batch_size=batch_size
                )
            else:
                result = self.model(
                    bev_seq, trans_matrices, num_all_agents, batch_size=batch_size
                )

        if self.kd_flag:
            kd_loss = self.get_kd_loss(
                batch_size, data, fused_layer, num_agent, x_5, x_6, x_7
            )
        else:
            kd_loss = 0

        labels = labels.view(result["cls"].shape[0], -1, result["cls"].shape[-1])

        N = bev_seq.shape[0]

        loss_collect = self.loss_calculator(
            result, anchors, reg_loss_mask, reg_targets, labels, N
        )
        loss_num = loss_collect[0]
#         print("loss_num:",loss_num)
        if loss_num == 3:
            loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
        elif loss_num == 2:
            loss_num, loss, loss_cls, loss_loc, loss_loc_list, loss_cls_list = loss_collect
        elif loss_num == 4:
            loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion = loss_collect

#         loss_sum = loss_cls + loss_loc + kd_loss
        grads_list = []   
        loss2_list = []
        if self.MGDA:
            self.optimizer_encoder.zero_grad()
            self.optimizer_head.zero_grad()
            loss.backward()
            self.optimizer_encoder.step()
            self.optimizer_head.step()

        else:
            for i in range(6):
                self.optimizer.zero_grad()
                loss2 = loss_loc_list[i]+loss_cls_list[i]
                if loss2 != torch.tensor(0):
                    loss2_list.append(loss2)
                    loss2.backward(retain_graph=True)
                    split_grads = []
                    n = 0
                    for param in self.model.parameters():
                        n = n+1
                        if param.grad is not None:
                            split_grads.append(param.grad.clone().detach())
                    grads_list.append(split_grads)
#             print(len(grads_list))
            gs = self.ComputeGradient(grads_list)

            self.optimizer.zero_grad()
            loss_sum = loss2_list[0]
            loss_sum.backward()
            param = list(self.model.parameters())

#             print(len(param))
            k = 0
            for p in param:
                if p.grad is not None:
                    p.grad = gs[k]
                    k = k+1
            self.optimizer.step()

        if self.config.pred_type in ["motion", "center"] and not self.only_det:
            if self.config.motion_state:
                return (
                    loss.item(),
                    loss_cls.item(),
                    loss_loc_1.item(),
                    loss_loc_2.item(),
                    loss_motion.item(),
                    grads,
                )
            else:
                return (
                    loss.item(),
                    loss_cls.item(),
                    loss_loc_1.item(),
                    loss_loc_2.item(),
                    grads,
                )
        else:
            return loss_sum.item(), loss_cls.item(), loss_loc.item(), gs
        

    def check_shape_match(self,list1, list2):
        index_list = []
        for i, (item1, item2) in enumerate(zip(list1, list2)):
            if np.shape(item1) != np.shape(item2):
                index_list.append(i)
        if len(index_list) == 0:
            return True, index_list
        else:
            return False, index_list
    
    
    
    def get_kd_loss(self, batch_size, data, fused_layer, num_agent, x_5, x_6, x_7):
        if self.kd_flag:
            bev_seq_teacher = data["bev_seq_teacher"]
            kd_weight = data["kd_weight"]
            (
                x_8_teacher,
                x_7_teacher,
                x_6_teacher,
                x_5_teacher,
                x_3_teacher,
                x_2_teacher,
            ) = self.teacher(bev_seq_teacher)

            # for k, v in self.teacher.named_parameters():
            # 	if k != 'xxx.weight' and k != 'xxx.bias':
            # 		print(v.requires_grad)  # should be False

            # for k, v in self.model.named_parameters():
            # 	if k != 'xxx.weight' and k != 'xxx.bias':
            # 		print(v.requires_grad)  # should be False

            # -------- KD loss---------#
            kl_loss_mean = nn.KLDivLoss(size_average=True, reduce=True)

            # target_x8 = x_8_teacher.permute(0, 2, 3, 1).reshape(5 * batch_size * 256 * 256, -1)
            # student_x8 = x_8.permute(0, 2, 3, 1).reshape(5 * batch_size * 256 * 256, -1)
            # kd_loss_x8 = kl_loss_mean(F.log_softmax(student_x8, dim=1), F.softmax(target_x8, dim=1))
            # #
            target_x7 = x_7_teacher.permute(0, 2, 3, 1).reshape(
                num_agent * batch_size * 128 * 128, -1
            )
            student_x7 = x_7.permute(0, 2, 3, 1).reshape(
                num_agent * batch_size * 128 * 128, -1
            )
            kd_loss_x7 = kl_loss_mean(
                F.log_softmax(student_x7, dim=1), F.softmax(target_x7, dim=1)
            )
            #
            target_x6 = x_6_teacher.permute(0, 2, 3, 1).reshape(
                num_agent * batch_size * 64 * 64, -1
            )
            student_x6 = x_6.permute(0, 2, 3, 1).reshape(
                num_agent * batch_size * 64 * 64, -1
            )
            kd_loss_x6 = kl_loss_mean(
                F.log_softmax(student_x6, dim=1), F.softmax(target_x6, dim=1)
            )
            # #
            target_x5 = x_5_teacher.permute(0, 2, 3, 1).reshape(
                num_agent * batch_size * 32 * 32, -1
            )
            student_x5 = x_5.permute(0, 2, 3, 1).reshape(
                num_agent * batch_size * 32 * 32, -1
            )
            kd_loss_x5 = kl_loss_mean(
                F.log_softmax(student_x5, dim=1), F.softmax(target_x5, dim=1)
            )

            target_x3 = x_3_teacher.permute(0, 2, 3, 1).reshape(
                num_agent * batch_size * 32 * 32, -1
            )
            student_x3 = fused_layer.permute(0, 2, 3, 1).reshape(
                num_agent * batch_size * 32 * 32, -1
            )
            kd_loss_fused_layer = kl_loss_mean(
                F.log_softmax(student_x3, dim=1), F.softmax(target_x3, dim=1)
            )

            kd_loss = kd_weight * (
                kd_loss_x7 + kd_loss_x6 + kd_loss_x5 + kd_loss_fused_layer
            )
            # kd_loss = kd_weight * (kd_loss_x6 + kd_loss_x5 + kd_loss_fused_layer)
            # print(kd_loss)

        else:
            kd_loss = 0
        return kd_loss

    # normal case (pre & intermediate)
    def predict_all(self, data, batch_size, validation=True, num_agent=5):
        bev_seq = data["bev_seq"]
        # vis_maps = data["vis_maps"]
        trans_matrices = data["trans_matrices"]
        num_all_agents = data["num_agent"]
        # num_sensor = num_all_agents[0, 0]

        # if self.MGDA:
        #    x = self.encoder(bev_seq)
        #    result = self.head(x)
        # else:
        #    result = self.model(bev_seq, trans_matrices, num_agent_tensor, batch_size=batch_size)
        # result = self.model(bev_seq)
        if self.config.flag.startswith("when2com") or self.config.flag.startswith(
            "who2com"
        ):
            if self.config.split == "train":
                result = self.model(
                    bev_seq,
                    trans_matrices,
                    num_all_agents,
                    batch_size=batch_size,
                    training=True,
                )
            else:
                result = self.model(
                    bev_seq,
                    trans_matrices,
                    num_all_agents,
                    batch_size=batch_size,
                    inference=self.config.inference,
                    training=False,
                )
        elif self.config.flag.startswith("disco"):
            result, save_agent_weight_list = self.model(
                bev_seq, trans_matrices, num_all_agents, batch_size=batch_size
            )
        else:
            result = self.model(
                bev_seq, trans_matrices, num_all_agents, batch_size=batch_size
            )

        N = bev_seq.shape[0]

        if validation:
            labels = data["labels"]
            anchors = data["anchors"]
            reg_targets = data["reg_targets"]
            reg_loss_mask = data["reg_loss_mask"]
            motion_labels = None
            motion_mask = None

            labels = labels.view(result["cls"].shape[0], -1, result["cls"].shape[-1])

            if self.config.motion_state:
                motion_labels = data["motion_label"]
                motion_mask = data["motion_mask"]
                motion_labels = motion_labels.view(
                    result["state"].shape[0], -1, result["state"].shape[-1]
                )
            N = bev_seq.shape[0]

            loss_collect = self.loss_calculator(
                result,
                anchors,
                reg_loss_mask,
                reg_targets,
                labels,
                N,
                motion_labels,
                motion_mask,
            )
            loss_num = loss_collect[0]

            if loss_num == 3:
                loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
            elif loss_num == 2:
                loss_num, loss, loss_cls, loss_loc,loss_loc_list, loss_cls_list = loss_collect
            elif loss_num == 4:
                (
                    loss_num,
                    loss,
                    loss_cls,
                    loss_loc_1,
                    loss_loc_2,
                    loss_motion,
                ) = loss_collect

        seq_results = [[] for i in range(num_agent)]

        # global_points = [[] for i in range(num_sensor)]
        # cls_preds = [[] for i in range(num_sensor)]

        for k in range(num_agent):

            bev_seq = torch.unsqueeze(data["bev_seq"][k, :, :, :, :], 0)

            if torch.nonzero(bev_seq).shape[0] == 0:
                seq_results[k] = []
            else:
                batch_box_preds = torch.unsqueeze(result["loc"][k, :, :, :, :, :], 0)
                batch_cls_preds = torch.unsqueeze(result["cls"][k, :, :], 0)
                anchors = torch.unsqueeze(data["anchors"][k, :, :, :, :], 0)
                batch_motion_preds = None

                if not self.only_det:
                    if self.config.pred_type == "center":
                        batch_box_preds[:, :, :, :, 1:, 2:] = batch_box_preds[
                            :, :, :, :, [0], 2:
                        ]

                class_selected = apply_nms_det(
                    batch_box_preds,
                    batch_cls_preds,
                    anchors,
                    self.code_type,
                    self.config,
                    batch_motion_preds,
                )

                seq_results[k] = class_selected

                # global_points[k], cls_preds[k] = apply_box_global_transform(trans_matrices_map[k],batch_box_preds,batch_cls_preds,anchors,self.code_type,self.config,batch_motion_preds)

        # all_points_scene = numpy.concatenate(tuple(global_points), 0)
        # cls_preds_scene = torch.cat(tuple(cls_preds), 0)
        # class_selected_global = apply_nms_global_scene(all_points_scene, cls_preds_scene)

        if validation:
            if self.config.flag.startswith("disco"):
                return (
                    loss.item(),
                    loss_cls.item(),
                    loss_loc.item(),
                    seq_results,
                    save_agent_weight_list,
                )
            else:
                return loss.item(), loss_cls.item(), loss_loc.item(), seq_results
        else:
            return seq_results

    def predict_all_with_box_com(self, data, trans_matrices_map, validation=True):
        NUM_AGENT = 5
        bev_seq = data["bev_seq"]
        # vis_maps = data["vis_maps"]
        trans_matrices = data["trans_matrices"]
        num_agent_tensor = data["num_agent"]
        num_sensor = num_agent_tensor[0, 0]

        if self.MGDA:
            x = self.encoder(bev_seq)
            result = self.head(x)
        else:
            result = self.model(bev_seq, trans_matrices, num_agent_tensor, batch_size=1)

        N = bev_seq.shape[0]

        if validation:
            labels = data["labels"]
            anchors = data["anchors"]
            reg_targets = data["reg_targets"]
            reg_loss_mask = data["reg_loss_mask"]
            motion_labels = None
            motion_mask = None

            labels = labels.view(result["cls"].shape[0], -1, result["cls"].shape[-1])

            if self.config.motion_state:
                motion_labels = data["motion_label"]
                motion_mask = data["motion_mask"]
                motion_labels = motion_labels.view(
                    result["state"].shape[0], -1, result["state"].shape[-1]
                )
            N = bev_seq.shape[0]

            loss_collect = self.loss_calculator(
                result,
                anchors,
                reg_loss_mask,
                reg_targets,
                labels,
                N,
                motion_labels,
                motion_mask,
            )
            loss_num = loss_collect[0]
            if loss_num == 3:
                loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
            elif loss_num == 2:
                loss_num, loss, loss_cls, loss_loc = loss_collect
            elif loss_num == 4:
                (
                    loss_num,
                    loss,
                    loss_cls,
                    loss_loc_1,
                    loss_loc_2,
                    loss_motion,
                ) = loss_collect

        seq_results = [[] for i in range(NUM_AGENT)]
        # local_results_wo_local_nms = [[] for i in range(NUM_AGENT)]
        local_results_af_local_nms = [[] for i in range(NUM_AGENT)]

        # global_points = [[] for i in range(num_sensor)]
        # cls_preds = [[] for i in range(num_sensor)]
        global_boxes_af_localnms = [[] for i in range(num_sensor)]
        box_scores_af_localnms = [[] for i in range(num_sensor)]
        
        forward_message_size = 0
        forward_message_size_two_nms = 0

        for k in range(NUM_AGENT):
            bev_seq = torch.unsqueeze(data["bev_seq"][k, :, :, :, :], 0)

            if torch.nonzero(bev_seq).shape[0] == 0:
                seq_results[k] = []
            else:
                batch_box_preds = torch.unsqueeze(result["loc"][k, :, :, :, :, :], 0)
                batch_cls_preds = torch.unsqueeze(result["cls"][k, :, :], 0)
                anchors = torch.unsqueeze(data["anchors"][k, :, :, :, :], 0)

                if self.config.motion_state:
                    batch_motion_preds = result["state"]
                else:
                    batch_motion_preds = None

                if not self.only_det:
                    if self.config.pred_type == "center":
                        batch_box_preds[:, :, :, :, 1:, 2:] = batch_box_preds[
                            :, :, :, :, [0], 2:
                        ]

                class_selected, box_scores_pred_cls = apply_nms_det(
                    batch_box_preds,
                    batch_cls_preds,
                    anchors,
                    self.code_type,
                    self.config,
                    batch_motion_preds,
                )

                # transform all the boxes before local nms to the global coordinate
                # global_points[k], cls_preds[k] = apply_box_global_transform(trans_matrices_map[k], batch_box_preds,
                #                                                            batch_cls_preds, anchors, self.code_type,
                #                                                            self.config, batch_motion_preds)

                # transform the boxes after local nms to the global coordinate
                (
                    global_boxes_af_localnms[k],
                    box_scores_af_localnms[k],
                ) = apply_box_global_transform_af_localnms(
                    trans_matrices_map[k], class_selected, box_scores_pred_cls
                )
                # print(cls_preds[k].shape, box_scores_af_localnms[k].shape)

                forward_message_size = forward_message_size + 256 * 256 * 6 * 4 * 2
                forward_message_size_two_nms = (
                    forward_message_size_two_nms
                    + global_boxes_af_localnms[k].shape[0] * 4 * 2
                )

        # global results with one NMS
        # all_points_scene = numpy.concatenate(tuple(global_points), 0)
        # cls_preds_scene = torch.cat(tuple(cls_preds), 0)
        # class_selected_global = apply_nms_global_scene(all_points_scene, cls_preds_scene)

        # global results with two NMS
        global_boxes_af_local_nms = np.concatenate(tuple(global_boxes_af_localnms), 0)
        box_scores_af_local_nms = torch.cat(tuple(box_scores_af_localnms), 0)
        class_selected_global_af_local_nms = apply_nms_global_scene(
            global_boxes_af_local_nms, box_scores_af_local_nms
        )

        # transform the consensus global boxes to local agents (two NMS)
        back_message_size_two_nms = 0
        for k in range(num_sensor):
            local_results_af_local_nms[k], ms = apply_box_local_transform(
                class_selected_global_af_local_nms, trans_matrices_map[k]
            )
            back_message_size_two_nms = back_message_size_two_nms + ms

        sample_bandwidth_two_nms = (
            forward_message_size_two_nms + back_message_size_two_nms
        )

        # transform the consensus global boxes to local agents (One NMS)
        # back_message_size = 0
        # for k in range(num_sensor):
        #    local_results_wo_local_nms[k], ms = apply_box_local_transform(class_selected_global, trans_matrices_map[k])
        #    back_message_size = back_message_size + ms

        # sample_bandwidth = forward_message_size + back_message_size

        return (
            loss.item(),
            loss_cls.item(),
            loss_loc.item(),
            local_results_af_local_nms,
            class_selected_global_af_local_nms,
            sample_bandwidth_two_nms,
        )

    # FIXME: not in use?
    def cal_loss_scale(self, data):
        bev_seq = data["bev_seq"]
        labels = data["labels"]
        reg_targets = data["reg_targets"]
        reg_loss_mask = data["reg_loss_mask"]
        anchors = data["anchors"]
        motion_labels = None
        motion_mask = None

        with torch.no_grad():
            shared_feats = self.encoder(bev_seq)
        shared_feats_tensor = shared_feats.clone().detach().requires_grad_(True)
        result = self.head(shared_feats_tensor)
        if self.config.motion_state:
            motion_labels = data["motion_label"]
            motion_mask = data["motion_mask"]
            motion_labels = motion_labels.view(
                result["state"].shape[0], -1, result["state"].shape[-1]
            )
        self.optimizer_encoder.zero_grad()
        self.optimizer_head.zero_grad()
        grads = {}
        labels = labels.view(result["cls"].shape[0], -1, result["cls"].shape[-1])
        N = bev_seq.shape[0]

        # calculate loss
        grad_len = 0

        """
        Classification Loss
        """
        loss_cls = (
            self.alpha * torch.sum(self.criterion["cls"](result["cls"], labels)) / N
        )
        # loss_loc = torch.sum(self.criterion['loc'](result['loc'],reg_targets,mask = reg_loss_mask)) / N
        self.optimizer_encoder.zero_grad()
        self.optimizer_head.zero_grad()

        loss_cls.backward(retain_graph=True)
        grads[0] = []
        grads[0].append(shared_feats_tensor.grad.data.clone().detach())
        shared_feats_tensor.grad.data.zero_()
        grad_len += 1

        """
        Localization Loss
        """
        loc_scale = False
        # loss_mask_num = torch.nonzero(
        #     reg_loss_mask.view(-1, reg_loss_mask.shape[-1])
        # ).size(0)

        if self.code_type in ["corner_1", "corner_2", "corner_3"]:
            target = reg_targets[reg_loss_mask].reshape(-1, 5, 2)
            flip_target = torch.stack(
                [target[:, 0], target[:, 3], target[:, 4], target[:, 1], target[:, 2]],
                dim=-2,
            )
            pred = result["loc"][reg_loss_mask].reshape(-1, 5, 2)
            t = torch.sum(torch.norm(pred - target, dim=-1), dim=-1)
            f = torch.sum(torch.norm(pred - flip_target, dim=-1), dim=-1)
            loss_loc = torch.sum(torch.min(t, f)) / N

        elif self.code_type == "faf":
            if self.loss_type == "corner_loss":
                if self.only_det:
                    loss_loc = self.corner_loss(
                        anchors, reg_loss_mask, reg_targets, result["loc"]
                    )
                elif self.config.pred_type in ["motion", "center"]:

                    # only center/motion for pred

                    loss_loc_1 = self.corner_loss(
                        anchors,
                        reg_loss_mask[..., 0][..., [0]],
                        reg_targets[..., [0], :],
                        result["loc"][..., [0], :],
                    )
                    pred_reg_loss_mask = reg_loss_mask[..., 1:, :]
                    if self.config.motion_state:
                        pred_reg_loss_mask = motion_mask  # mask out static object
                    loss_loc_2 = F.smooth_l1_loss(
                        result["loc"][..., 1:, :][pred_reg_loss_mask],
                        reg_targets[..., 1:, :][pred_reg_loss_mask],
                    )

                    self.optimizer_encoder.zero_grad()
                    self.optimizer_head.zero_grad()

                    loss_loc_1.backward(retain_graph=True)
                    grads[1] = []
                    grads[1].append(shared_feats_tensor.grad.data.clone().detach())
                    shared_feats_tensor.grad.data.zero_()

                    self.optimizer_encoder.zero_grad()
                    self.optimizer_head.zero_grad()

                    loss_loc_2.backward(retain_graph=True)
                    grads[2] = []
                    grads[2].append(shared_feats_tensor.grad.data.clone().detach())
                    shared_feats_tensor.grad.data.zero_()
                    loc_scale = True
                    grad_len += 2

                # corners for pred
                else:
                    loss_loc = self.corner_loss(
                        anchors, reg_loss_mask, reg_targets, result["loc"]
                    )
            else:

                loss_loc = F.smooth_l1_loss(
                    result["loc"][reg_loss_mask], reg_targets[reg_loss_mask]
                )

            if not loc_scale:
                grad_len += 1
                self.optimizer_encoder.zero_grad()
                self.optimizer_head.zero_grad()
                loss_loc.backward(retain_graph=True)
                grads[1] = []
                grads[1].append(shared_feats_tensor.grad.data.clone().detach())
                shared_feats_tensor.grad.data.zero_()

        """
        Motion state Loss
        """
        if self.config.motion_state:
            loss_motion = (
                torch.sum(self.criterion["cls"](result["state"], motion_labels)) / N
            )

            self.optimizer_encoder.zero_grad()
            self.optimizer_head.zero_grad()

            loss_motion.backward(retain_graph=True)
            grads[3] = []
            grads[3].append(shared_feats_tensor.grad.data.clone().detach())
            shared_feats_tensor.grad.data.zero_()
            grad_len += 1

        # ---------------------------------------------------------------------
        # -- Frank-Wolfe iteration to compute scales.
        scale = np.zeros(grad_len, dtype=np.float32)
        sol, min_norm = MinNormSolver.find_min_norm_element(
            [grads[t] for t in range(grad_len)]
        )
        for i in range(grad_len):
            scale[i] = float(sol[i])

        # print(scale)
        return scale
