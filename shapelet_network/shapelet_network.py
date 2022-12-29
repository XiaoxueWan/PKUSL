# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:43:06 2022

@author: Lenovo
"""
import torch
from torch import nn
from shapelet_network.shapeletsDistBlocks import ShapeletsDistBlocks
from utils import normalize_dim

class LearningShapeletsModel(nn.Module):
    """
    Implements Learning Shapelets. Just puts together the ShapeletsDistBlocks with a
    linear layer on top.
    ----------
    shapelets_size_and_len : dict(int:int)
        keys are the length of the shapelets for a block and the values the number of shapelets for the block
    in_channels : int
        the number of input channels of the dataset
    num_classes: int
        the number of classes for classification
    dist_measure: 'string'
        the distance measure, either of 'euclidean', 'cross-correlation', or 'cosine'
    to_cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size_and_len, num_classes, in_channels=1, R=1,
                 dist_measure='euclidean', ucr_dataset_name='comman', to_cuda=True):
        super(LearningShapeletsModel, self).__init__()

        self.to_cuda = to_cuda
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.shapelets_blocks = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len=shapelets_size_and_len,
                                                    dist_measure=dist_measure, ucr_dataset_name=ucr_dataset_name, 
                                                    to_cuda=to_cuda)
        #self.linear = nn.Linear(self.num_shapelets, num_classes)
        '''r对应USSL论文里面的kernel_parameter'''
        self.R=R
        if self.to_cuda:
            self.cuda()
        self.num_classes=num_classes
        

    def forward(self, x, W, optimize='acc'):
        """
        Calculate the distances of each time series to the shapelets and stack a linear layer on top.
        m:distance_matrix
        H:shapelet similarity matrix
        G:similarity matrix of rhe shapelet transformed time series
        """
        '''D:[num_train,1,num_shapelets]'''
        '''G'''
        
        D = self.shapelets_blocks(x)
        a=torch.transpose(D,1,0)
        D=torch.squeeze(D)
        X_S=torch.transpose(D,1,0)
        G=self.calculate_similarity_matrix(a,a)
        '''距离矩阵乘以权重'''
        if optimize == 'acc':
            x = torch.mm(W.T, X_S)
        '''H'''
        a = torch.transpose(self.shapelets_blocks.get_shapelets(),0,1)
        H = self.calculate_similarity_matrix(a,a)
        return x,H,G,X_S
    
    def calculate_similarity_matrix(self,a,b):
        '''a,b:[1,num,length]'''
        dist_a_b=torch.cdist(a,b,2).pow(2)
        dist_a_b=torch.squeeze(dist_a_b,dim=0)
        dist_end=torch.exp(torch.div(dist_a_b,-self.R))
        return dist_end

    def transform(self, X):
        """
        Performs the shapelet transform with the input time series data x
        @param X: the time series data
        @type X: tensor(float) of shape (n_samples, in_channels, len_ts)
        @return: the shapelet transform of x
        @rtype: tensor(float) of shape (num_samples, num_shapelets)
        """
        return self.shapelets_blocks(X)

    def get_shapelets(self):
        """
        Return a matrix of all shapelets. The shapelets are ordered (ascending) according to
        the shapelet lengths and padded with NaN.
        @return: a tensor of all shapelets
        @rtype: tensor(float) with shape (in_channels, num_total_shapelets, shapelets_size_max)
        """
        return self.shapelets_blocks.get_shapelets()
    
    def get_y_pseudo_class(self):
        return self.shapelets_blocks.get_y_pseudo_class()

    def set_shapelet_weights(self, weights):
        """
        Set the weights of all shapelets. The shapelet weights are expected to be ordered ascending according to the
        length of the shapelets. The values in the matrix for shapelets of smaller length than the maximum
        length are just ignored.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (in_channels, num_total_shapelets, shapelets_size_max)
        @return:
        @rtype: None
        """
        start = 0
        for i, (shapelets_size, num_shapelets) in enumerate(self.shapelets_size_and_len.items()):
            end = start + num_shapelets
            self.set_shapelet_weights_of_block(i, weights[start:end, :, :shapelets_size])
            start = end

    def set_shapelet_weights_of_block(self, i, weights):
        """
        Set the weights of shapelet block i.
        @param i: The index of the shapelet block
        @type i: int
        @param weights: the weights for the shapelets of block i
        @type weights: array-like(float) of shape (in_channels, num_shapelets, shapelets_size)
        @return:
        @rtype: None
        """
        self.shapelets_blocks.set_shapelet_weights_of_block(i, weights)

    def set_weights_of_shapelet(self, i, j, weights):
        """
        Set the weights of shapelet j in shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param j: the index of the shapelet in shapelet block i
        @type j: int
        @param weights: the weights for the shapelet
        @type weights: array-like(float) of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        self.shapelets_blocks.set_shapelet_weights_of_single_shapelet(i, j, weights)