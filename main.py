# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:37:54 2022

@author: Lenovo
"""
import torch
import time
import math
import pandas as pd
import numpy as np
import torch.nn.functional as F

from torch import optim
from sklearn import metrics
from matplotlib import pyplot
from cluster import LearningShapelets
from get_data import get_data_ucr, get_data_ACS
from utils import get_weights_via_kmeans, plot_shapelets


class Main():
    def __init__(self,
                 ucr_dataset_name,
                 ucr_dataset_base_folder,
                 K=0.1,
                 Lmin=0.3,
                 learning_rate=0.01, 
                 epoch=2000, 
                 batch_size=234, 
                 t1=5, 
                 t2=1, 
                 t3=1, 
                 t4=1,
                 shapelet_epoch=30,
                 R=1,
                 q=0.5,
                 show_visualization=True):
        '''
        s_num:shapelet数量，K为shapelet数量占所有的比例
        s_length:shapelet的长度，Lmin为shapelet长度占所有的比例
        self.lr:学习率
        '''
        if ucr_dataset_name == 'ACS':
            if 'knowledge' in ucr_dataset_base_folder:
                self.x_train, self.y_train, self.x_test, self.y_test, self.knowledge = get_data_ACS(ucr_dataset_base_folder).main()
            else:
                self.x_train, self.y_train, self.x_test, self.y_test = get_data_ACS(ucr_dataset_base_folder).main()
        else:
            self.x_train, self.y_train, self.x_test, self.y_test = get_data_ucr(ucr_dataset_name, ucr_dataset_base_folder).main()
        if self.y_test.min() < 0:
            self.y_test = self.y_test - self.y_test.min()
            
        self.ucr_dataset_name = ucr_dataset_name
        self.ucr_dataset_base_folder = ucr_dataset_base_folder
        self.learning_rate = learning_rate
        self.learning_weight = 0.01
        self.epsilon = 0.9
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.shapelet_epoch = shapelet_epoch
        self.R = R
        self.q = q
        self.show_visualization = show_visualization
        
        self.K = K
        self.Lmin = Lmin
        s_num = int(self.K * self.x_train.shape[0])   
        s_lenght = int(self.Lmin * self.x_train.shape[2])  
        self.shapelets_size_and_len = {s_lenght:s_num}
        self.train_dataset_size = self.x_train.shape[0]
        self.epoch = epoch
        self.batch_size = batch_size
        
        '''当前时间'''
        t = time.localtime()
        
        '''实验结果自动记录'''
        self.record={'time':str(t.tm_year) + str(t.tm_mon) + str(t.tm_mday), 'ucr_dataset_name':ucr_dataset_name,\
                     'ucr_dataset_path':ucr_dataset_base_folder, 'knowledge':self.knowledge.shape[1], 'K': K,
                     'Lim': Lmin, 't1': self.t1, 't2': self.t2, 't3': self.t3, 't4': self.t4, 'shapelet_epoch':self.shapelet_epoch,
                     'learning_rate': self.learning_rate, 'epoch': epoch, 'lw': self.learning_weight, 
                     'batch_size':self.batch_size, 'RI':0, 'q':self.q, 'NMI':0, 'train_time':0, 'test_time':0}
    
    def initialize_model(self):
        '''
        模型初始化
        '''
        n_ts, n_channels, len_ts = self.x_train.shape
        num_classes = len(set(self.y_train))
        #loss_func=nn.CrossEntropyLoss()
        #loss_func=self.CrossEntrppyLoss_with_sum
        loss_function = self.least_square_error
        dist_measure = 'euclidean'
        learning_shapelets = LearningShapelets(shapelets_size_and_len=self.shapelets_size_and_len,
                                              train_dataset_size=self.train_dataset_size, knowledge=self.knowledge, loss_func=loss_function,
                                              in_channels=n_channels, num_classes=num_classes, learning_rate=self.learning_rate,
                                              learning_weight=self.learning_weight, to_cuda=False, verbose=1, dist_measure=dist_measure,
                                              ucr_dataset_name=self.ucr_dataset_name, t1=self.t1, t2=self.t2, 
                                              t3=self.t3, t4=self.t4, shapelet_epoch=self.shapelet_epoch, R=self.R, q=self.q, show_visualization=self.show_visualization)
        return learning_shapelets
    
    def CrossEntrppyLoss_with_sum(self, y, y_label):
        '''自己定义一个交叉熵损失函数，没有对所有样本求平均
            y_label_one_hot:是真实标签的one_hot编码
        '''
        y_label = y_label.unsqueeze(-1)
        y_label_one_hot = torch.zeros(size=(y_label.shape[0], y.shape[-1]))
        if y_label.min()<0:
            y_label = torch.sub(y_label, y_label.min())
        y_label_one_hot = y_label_one_hot.scatter_(dim=1, index=y_label.reshape(-1, 1), value=1)
        '''交叉熵损失函数'''
        loss = - torch.sum(y_label_one_hot * torch.log(F.softmax(y, dim=1)), 1)
        '''接近于0-1的可以求梯度的损失函数'''
        loss = torch.mean(loss)
        return loss
    
    def least_square_error(self, y, y_pseudo_class):
        loss = torch.norm(torch.sub(y, y_pseudo_class))
        return loss.pow(2)
    
    def initialize_shapelet_aptimizer(self, learning_shapelets):
        for i, (shapelets_size, num_shapelets) in enumerate(self.shapelets_size_and_len.items()):
            weights_block = get_weights_via_kmeans(self.x_train, shapelets_size, num_shapelets)
            learning_shapelets.set_shapelet_weights_of_block(i, weights_block)
        
        #print(list(learning_shapelets.model.parameters()),'%%%%%%%%%%')
        #optimizer = optim.Adam(learning_shapelets.model.parameters(), lr=self.lr, weight_decay=self.w, eps=self.epsilon)
        
        #optimizer_Shapelet = optim.SGD(learning_shapelets.model.parameters(),
                                      # lr = self.learning_rate, weight_decay = self.learning_weight, momentum = 0)
        
        optimizer_Shapelet = optim.Adam(learning_shapelets.model.parameters(), lr=self.learning_rate, weight_decay=self.learning_weight, eps=self.epsilon)
        learning_shapelets.set_optimizer(optimizer_Shapelet)
        return learning_shapelets
    
    def NMI(self,A,B):
        #样本点数
        total = len(A)
        A_ids = set(A)
        B_ids = set(B)
        #互信息计算
        MI = 0
        eps = 1.4e-45
        for idA in A_ids:
            for idB in B_ids:
                idAOccur = np.where(A==idA)
                idBOccur = np.where(B==idB)
                idABOccur = np.intersect1d(idAOccur,idBOccur)
                px = 1.0*len(idAOccur[0])/total
                py = 1.0*len(idBOccur[0])/total
                pxy = 1.0*len(idABOccur)/total
                MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
        # 标准化互信息
        Hx = 0
        for idA in A_ids:
            idAOccurCount = 1.0*len(np.where(A==idA)[0])
            Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
        Hy = 0
        for idB in B_ids:
            idBOccurCount = 1.0*len(np.where(B==idB)[0])
            Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
        MIhat = 2.0*MI/(Hx+Hy)
        return MIhat

    def eval_accuracy(self, model, x, y):
        
        predictions = model.predict(x)
        
        if len(predictions.shape) == 2:
            predictions = predictions.argmax(axis = 0)
            
        RI = metrics.rand_score(y, predictions)
        NMI = self.NMI(y,predictions)
        print('RI',RI,'NMI',NMI)
        #print('RI',RI)
        print(predictions, y,'**************')
       # print(f"Accuracy: {(predictions == Y).sum() / Y.size}")
        return RI,NMI
    
    def train_model(self):
        '''trainning'''
        learning_shapelets = self.initialize_model()
        learning_shapelets = self.initialize_shapelet_aptimizer(learning_shapelets)
        
        start=time.clock()
        losses = learning_shapelets.fit(self.x_train, self.y_train, epochs = self.epoch, 
                                        batch_size = 256, shuffle = False, drop_last = False)
        end=time.clock()
        self.record['train_time'] = end - start
        
        '''plot_loss_shapelet'''
        pyplot.figure(figsize=(4,4))
        pyplot.plot(losses, color='black')
        pyplot.title("Loss over training steps")
        pyplot.savefig('loss_fig.pdf')
        pyplot.show() 
        shapelets = learning_shapelets.get_shapelets()
        shapelet_transform = learning_shapelets.transform(self.x_test)
        weights = learning_shapelets.get_weight()
        record_data_plot = plot_shapelets(self.x_test, shapelets, self.y_test, shapelet_transform, 
                                          weights, self.ucr_dataset_name)
        
        '''testing'''
        start = time.clock()
        #RI,NMI=self.eval_accuracy(learning_shapelets, self.X_test, self.y_test)
        RI,NMI = self.eval_accuracy(learning_shapelets, self.x_test, self.y_test)
        NMI = 0
        end = time.clock()
        self.record['RI'], self.record['NMI'], self.record['test_time'] = RI, NMI, end-start
        
        '''to_excel'''
        record = pd.read_excel('record_auto.xlsx')
        record = record.append(self.record, ignore_index = True)
        record.to_excel('record_auto.xlsx', index = False)
        return record_data_plot

        
        
        
    
    
    
    