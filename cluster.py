# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:41:35 2022

@author: Lenovo
"""
import torch
import os
import pandas as pd
import warnings
import numpy as np

from torch import tensor
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from shapelet_network.shapelet_network import LearningShapeletsModel
from utils import ShapeletsDistanceLoss,ShapeletsSimilarityLoss,visualization3D,visualization2D
from kmeans_pytorch import kmeans
from sklearn import metrics


class LearningShapelets:
    def __init__(self, shapelets_size_and_len, train_dataset_size, knowledge, loss_func, learning_rate, learning_weight, 
                 in_channels=1, num_classes=2, dist_measure='euclidean', ucr_dataset_name='comman', verbose=0, 
                 to_cuda=False, l1=0.0, l2=0.0, t1=5, t2=1, t3=1, t4=10, shapelet_epoch=30, R=1, q=0.5, show_visualization=True):

        self.model = LearningShapeletsModel(shapelets_size_and_len=shapelets_size_and_len,
                                            num_classes=num_classes, in_channels=1, 
                                            R=R,
                                            dist_measure=dist_measure,
                                            ucr_dataset_name=ucr_dataset_name,
                                            to_cuda=to_cuda)
        
        self.to_cuda = to_cuda
        if self.to_cuda:
            self.model.cuda()
            
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.shapelets_size_and_len = shapelets_size_and_len
        self.loss_func = loss_func
        self.epsilon = 1e-7
        self.LW = learning_weight
        self.shapelet_epoch=shapelet_epoch
        self.verbose = verbose
        self.optimizer_Shapelet = None
        self.knowledge = knowledge
       # self.y_pseudo_class = torch.randint(0,num_classes,(self.num_classes,train_dataset_size), requires_grad=True,dtype=torch.float)#对伪类标签y进行初始化定义y的优化器

        if not all([l1 == 0.0, l2 == 0.0]) and not all([l1 > 0.0]):
            raise ValueError("For using the regularizer, the parameters 'k' and 'l1' must be greater than zero."
                             " Otherwise 'k', 'l1', and 'l2' must all be set to zero.")
        self.l1 = l1
        self.l2 = l2
        self.loss_dist = ShapeletsDistanceLoss(dist_measure=dist_measure, k=0)
        self.loss_sim_block = ShapeletsSimilarityLoss()
        # add a variable to indicate if regularization shall be used, just used to make code more readable
        self.use_regularizer = True if l1 > 0.0 else False
        self.learning_rate = learning_rate
        self.T1 = t1
        self.T2 = t2
        self.T3 = t3
        self.T4 = t4
        self.R = R
        self.q = q
        self.show_visualization = show_visualization
        self.gradients = []
        #print(knowledge,'knowledge')
#        self.knowledge = torch.tensor([
#                                  [2,5,0],[2,6,0],[4,1,1],[2,6,0],[4,1,1],[2,6,0],[4,1,1],[4,6,1],[3,6,0],[3,6,0],
#                                  [3,6,0],[3,6,0],[2,5,0],[4,1,1],[4,1,1],[4,2,1],[2,6,0],[4,3,0],[4,5,0],[4,6,0]
#                                  ])
        
    def set_optimizer(self, optimizer_Shapelet):
        self.optimizer_Shapelet = optimizer_Shapelet
        return
    
    def set_shapelet_weights(self, weights):
        self.model.set_shapelet_weights(weights)
        if self.optimizer_Shapelet is not None:
            warnings.warn("Updating the model parameters requires to reinitialize the optimizer. Please reinitialize"
                          " the optimizer via set_optimizer(optim)")
        return

    def set_shapelet_weights_of_block(self, i, weights):
        self.model.set_shapelet_weights_of_block(i, weights)
        if self.optimizer_Shapelet is not None:
            warnings.warn("Updating the model parameters requires to reinitialize the optimizer. Please reinitialize"
                          " the optimizer via set_optimizer(optim)")
            
    def save_gradient(self, name):
        def hook(grad):
            self.gradients.append([name, grad])
        return hook
    
    def save_gradient_1(self, node, node1):
        node.register_hook(self.save_gradient(node1))
        return
   
    def Loss_YLY(self, G, Y):
        '''
        计算伪类标签Y的损失：Loss=trace(YLY'T)
        
        参数：
        --------
        G：[样本数 n，样本数 n]
           高斯函数（X：i-X:j)
        Y: [类别数 c，样本数 n] 
           伪类标签
           
        返回值：
            int
           损失值
        '''
        x = torch.sum(G, dim=1)
        D = torch.diag(x)
        L = torch.sub(D, G)
        x = torch.mm(torch.mm(Y, L), Y.T)
        return x.trace()
        
    def calculate_chart(self, y_predict, y_pseudo_class, H, G, W, D):
        '''
        计算总的损失值=知识变量的损失+t2/2*|W'T*D-Y|+t1/2*|H|+1/2*tr(YLY'T)
        
        参数：
        ------------------------
        y_predict：预测的标签
           [样本数 n，时间序列长度 t]
           
        y_pseudo_class: 伪类标签
            [类别数 c，样本数 n] 
            
        H: shapelet之间距离的高斯函数
          [shapelet数 k, shapelet数 k]
            
        G:高斯函数（X：i-X:j)
          [样本数 n，样本数 n]
          
        W: 线性分类器
            [shapelet数 k，类别数 c] 
        
        D: 矩阵矩阵
           [shapelet数 k, 样本数 n]
           
        M: 知识分类器
           [shapelet数 k, 知识变量数 m]
           
        返回值：
        ----------------------
        loss:int
        '''
        
        self.save_gradient_1(H, 'H')
        Y_Y = self.loss_func(y_predict, y_pseudo_class)
        Loss_WX_Y = 0.5 * self.T2 * Y_Y
        Loss_H = 0.5 * self.T1 * torch.norm(H).pow(2)
        Loss_YLY = 0.5 * self.Loss_YLY(G, y_pseudo_class)
        Loss_W = 0.5 * self.T3 * torch.norm(W).pow(2)
        loss_total = Loss_YLY + Loss_WX_Y + Loss_H + Loss_W 
        return loss_total
    
    def calculate_similarity_matrix(self,a,b):
        '''a,b:[1,num,length]'''
        dist_a_b=torch.cdist(a,b,2).pow(2)
        dist_a_b=torch.squeeze(dist_a_b,dim=0)
        dist_end=torch.exp(torch.div(dist_a_b,-self.R))
        return dist_end
    
    def get_D_L(self, x, W):
        '''
        获取中间变量：
        
        参数：
        ------------------------
        x：输入时间序列样本
           [样本数 n，时间序列长度 t]
           
        W: 线性分类器
            [shapelet数 k，类别数 c] 
           
        返回值：
        ----------------------
        L: G的度矩阵-G矩阵
           [样本数 n，样本数 n]
        
        D: 矩阵矩阵
           [shapelet数 k, 样本数 n]
            
        y_predict: y的预测值
           [类别数c, 样本数 n]
            
        H: shapelet之间距离的高斯函数
          [shapelet数 k, shapelet数 k]
            
        G:高斯函数（X：i-X:j)
          [样本数 n，样本数 n]
          
        将 x=G 变为 x=G+系数乘(知识矩阵生成的G_knowledge)
        '''
        self.knowledge = self.knowledge.unsqueeze(0)
        G_knowledge = self.calculate_similarity_matrix(self.knowledge, self.knowledge)
        self.knowledge = self.knowledge.squeeze(0)
        y_predict, H, G, D = self.model(x, W) 
        G_total = self.q * G + (1-self.q) * G_knowledge
        x = torch.sum(G, dim=1)
        DG = torch.diag(x)
        L = torch.sub(DG, G)
        return L, D, y_predict, H, G_total
    
    def update_Y(self, W, D, L):
        '''
        更新伪类标签Y
        
        参数：
        ------------------------
        W: 线性分类器
            [shapelet数 k，类别数 c] 
            
        D: 矩阵矩阵
           [shapelet数 k, 样本数 n] 
           
        L: G的度矩阵-G矩阵
           [样本数 n，样本数 n]
           
        返回值：
        ----------------------
        updated_Y:更新之后的伪类标签
           [类别数c, 样本数 n]
        '''
        
        L_add_I=torch.add(
                          L,
                          self.T2 * torch.tensor(np.identity(L.shape[0]))
                          )
        L_add_I = torch.inverse(L_add_I)
        L_add_I = L_add_I.to(torch.float32)
        D = D.to(torch.float32)
        updated_Y = self.T2 * torch.mm(W.T, torch.mm(D, L_add_I))
        return updated_Y
    
    def update_W(self, D, update_Y):
        '''
        更新伪类标签 W
        
        参数：
        ------------------------
        updated_Y: 更新之后的伪类标签
           [类别数c, 样本数 n]
            
        D: 矩阵矩阵
           [shapelet数 k, 样本数 n] 
           
        返回值：
        ----------------------
        updated_W:更新之后的线性分类器参数
           [shapelet数 k，类别数 c] 
        '''
        
        X_add_I = torch.tensor(np.identity(D.shape[0]))
        X_add_I = torch.add(
                         self.T2 * torch.mm(D, D.T),
                         self.T3 * X_add_I
                         )
        X_add_I = torch.inverse(X_add_I)
        W2 = torch.mm(self.T2 * D,update_Y.T)
        X_add_I = X_add_I.to(torch.float32)
        W2 = W2.to(torch.float32)
        updated_W = torch.mm(
                          X_add_I,
                          W2
                          )
        updated_W = updated_W.detach()
        return updated_W

        
    def update(self, x, _):
        '''
        总的迭代过程
        
        参数：
        ------------------------
        x: 输入样本
           [样本数 n，时间序列长度 t]
            
        返回值：
        ----------------------
        Loss_total:每次迭代的损失值
           int
           
        y_pseudo_class:伪类标签
           [样本数 n, 类别数c]
        '''
        if self.is_first_update == True: #参数初始化
            D = self.model.transform(x)
            D = torch.squeeze(D)
            D = torch.transpose(D, 1, 0)
            #对伪类标签进行kmeans初始化
            self.cluster_ids_x, cluster_centers = kmeans(X=D.T, num_clusters=self.num_classes, distance='euclidean', device=torch.device('cuda:0'))
            self.y_pseudo_class=torch.zeros(
                                           size=(self.cluster_ids_x.shape[0],
                                                 self.cluster_ids_x.max().item()+1)
                                           ).scatter_(
                                                      1,
                                                      self.cluster_ids_x.unsqueeze(1),
                                                      1)
            self.y_pseudo_class = torch.div(
                                          self.y_pseudo_class,
                                          self.y_pseudo_class.sum(0).pow(1/2)
                                          )
            self.y_pseudo_class = self.y_pseudo_class.float() 
            self.updated_Y = torch.transpose(self.y_pseudo_class, 1, 0)
            self.updated_Y.requires_grad = False
        
        D = self.model.transform(x)
        D = torch.squeeze(D)
        D = torch.transpose(D, 1, 0)
        '''固定Y，S, 更新W'''
        self.updated_W = self.update_W(D, self.updated_Y) 
        '''固定Y，W, 更新S'''
        for i in range(self.shapelet_epoch):
            L, D, y_predict, H, G = self.get_D_L(x, self.updated_W)
            self.updated_Y = self.updated_Y.detach()
            loss_total = self.calculate_chart(
                                              y_predict,
                                              self.updated_Y,
                                              H,
                                              G,
                                              self.updated_W,
                                              D
                                              )
            loss_total.backward()
            self.optimizer_Shapelet.step()
            self.optimizer_Shapelet.zero_grad() 
        '''固定S，W，更新Y'''
        L, D, y_predict, H, G = self.get_D_L(x, self.updated_W)
        self.updated_Y = self.update_Y(self.updated_W, D, L)
        self.y_predict = y_predict.argmax(axis = 0)
        
        if self.show_visualization == True:
            D=torch.squeeze(D)
            X_S=torch.transpose(D,1,0)
            transformed_X_S = self.visualization(X_S.detach().numpy(),self.y_true)
            visualization2D(transformed_X_S, str(_)+'transformed_x')
        return loss_total


    def loss_sim(self):
        blocks = [params for params in self.model.named_parameters() if 'shapelets_blocks' in params[0]]
        loss = self.loss_sim_block(blocks)
        return loss

    def update_regularized(self, x, y):
        # get cross entropy loss and compute gradients
        y_hat = self.model(x)
        loss_ce = self.loss_func(y_hat, y)
        loss_ce.backward(retain_graph=True)

        # get shapelet distance loss and compute gradients
        dists_mat = self.model(x, 'dists')
        loss_dist = self.loss_dist(dists_mat) * self.l1
        loss_dist.backward(retain_graph=True)

        if self.l2 > 0.0:
            # get shapelet similarity loss and compute gradients
            loss_sim = self.loss_sim() * self.l2
            loss_sim.backward(retain_graph=True)

        # perform gradient upgrade step
        self.optimizer.step()
        self.optimizer.zero_grad()

        return (loss_ce.item(), loss_dist.item(), loss_sim.item()) if self.l2 > 0.0 else (
        loss_ce.item(), loss_dist.item())
        
    def visualization(self,x,y):
            #数据格式更改
        #训练x
        if self.is_first_update == True:
            self.tsne = TSNE(n_components=2)
            self.tsne.fit(x)
        #转换x
        transformed_x = self.tsne.fit_transform(x)
        transformed_x = pd.DataFrame(transformed_x)
        transformed_x = transformed_x.rename(columns={0:'dim0',1:'dim1'})
        transformed_x['label'] = y 
        return transformed_x
        

    def fit(self, X, Y, epochs=1, batch_size=256, shuffle=False, drop_last=False):
        if self.optimizer_Shapelet is None:
            raise ValueError("No optimizer set. Please initialize an optimizer via set_optimizer(optim)")

        # convert to pytorch tensors and data set / loader for training
        if not isinstance(X, torch.Tensor):
            X = tensor(X, dtype=torch.float).contiguous()
        if not isinstance(Y, torch.Tensor):
            Y = tensor(Y, dtype=torch.long).contiguous()
        if self.to_cuda:
            X = X.cuda()
            Y = Y.cuda()
        if Y.min()<0:
            Y=torch.sub(Y, Y.min())
        #print(Y,'y_label*************')
        train_ds = TensorDataset(X, Y)
        train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last)

        # set model in train mode
        self.model.train()

        losses_ce = []
        losses_dist = []
        losses_sim = []
        progress_bar = tqdm(range(epochs), disable=False if self.verbose > 0 else True)
        current_loss_ce = 0
        current_loss_dist = 0
        current_loss_sim = 0
        for _ in progress_bar:
            if _ ==0:
                self.is_first_update = True
            elif _==epochs-1:
                self.is_first_update = False
            else:
                self.is_first_update = None
            for j, (x, y) in enumerate(train_dl):
                # check if training should be done with regularizer
                if not self.use_regularizer:
                    self.y_true = y
                    current_loss_ce = self.update(x,_)
                    self.q = self.q + self.q/epochs 
                    
                    if self.is_first_update==False and self.show_visualization == True:
                        #特征可视化
                         x = torch.squeeze(x)
                         transformed_x_y = self.visualization(x.numpy(), self.y_predict.detach().numpy())
                         visualization2D(transformed_x_y, 'y_predict')
                         
                         transformed_x_y['label']=self.updated_Y.argmax(axis=0).detach().numpy()
                         visualization2D(transformed_x_y, 'pseudo_class_y')
                         
                         transformed_x_y['label']=y.numpy()
                         visualization2D(transformed_x_y, 'true_y')
                    #print(self.updated_Y.argmax(axis=0),y,metrics.rand_score(self.updated_Y.argmax(axis=0),y),
                    print('第'+str(_)+'次','self.updated_y与真实y之间的RI值:', metrics.rand_score(self.updated_Y.argmax(axis=0),y),
                          '初始化y与真实y之间的RI值:', metrics.rand_score(self.cluster_ids_x,y),'y_predict与updated_Y之间的RI值:',
                          metrics.rand_score(self.y_predict,self.updated_Y.argmax(axis=0)))
                    losses_ce.append(current_loss_ce)
                else:
                    if self.l2 > 0.0:
                        current_loss_ce, current_loss_dist, current_loss_sim = self.update_regularized(x, y)
                    else:
                        current_loss_ce, current_loss_dist = self.update_regularized(x, y)
                    losses_ce.append(current_loss_ce)
                    losses_dist.append(current_loss_dist)
                    if self.l2 > 0.0:
                        losses_sim.append(current_loss_sim)
            if not self.use_regularizer:
                progress_bar.set_description(f"Loss: {current_loss_ce}")
            else:
                if self.l1 > 0.0 and self.l2 > 0.0:
                    progress_bar.set_description(f"Loss CE: {current_loss_ce}, Loss dist: {current_loss_dist}, "
                                                 f"Loss sim: {current_loss_sim}")
                else:
                    progress_bar.set_description(f"Loss CE: {current_loss_ce}, Loss dist: {current_loss_dist}")
        return losses_ce if not self.use_regularizer else (losses_ce, losses_dist, losses_sim) if self.l2 > 0.0 else (
        losses_ce, losses_dist)
        
    def get_weight(self):
        #return torch.mm(self.updated_M,self.updated_W).numpy()
        return self.updated_W.numpy()

    def transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype = torch.float)
        if self.to_cuda:
            X = X.cuda()

        with torch.no_grad():
            shapelet_transform = self.model.transform(X)
            #print(type(shapelet_transform),shapelet_transform.shape)
        return shapelet_transform.squeeze().cpu().detach().numpy()

    def fit_transform(self, X, Y, epochs=1, batch_size=256, shuffle=False, drop_last=False):
        self.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return self.transform(X)

    def predict(self, X, batch_size=256):
        X = tensor(X, dtype=torch.float32)
        if self.to_cuda:
            X = X.cuda()
        ds = TensorDataset(X)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

        # set model in eval mode
        self.model.eval()

        """Evaluate the given data loader on the model and return predictions"""
        result = None
        with torch.no_grad():
            for x in dl:
                y_hat, H ,G, D= self.model(x[0], self.updated_W)
                y_hat = y_hat.cpu().detach().numpy()
                #print(y_hat,,'test_y,D')
                result = y_hat if result is None else np.concatenate((result, y_hat), axis=1)
        return result

    def get_shapelets(self):
        return self.model.get_shapelets().clone().cpu().detach().numpy()
