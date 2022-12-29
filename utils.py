# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:49:15 2022

@author: Lenovo
"""
import torch
import os
import random
import numpy
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec

from torch import nn
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from mpl_toolkits.mplot3d import Axes3D

def normalize_standard(X, scaler=None):
    shape = X.shape
    data_flat = X.flatten()
    if scaler is None:
        scaler = StandardScaler()
        data_transformed = scaler.fit_transform(data_flat.reshape(numpy.product(shape), 1)).reshape(shape)
    else:
        data_transformed = scaler.transform(data_flat.reshape(numpy.product(shape), 1)).reshape(shape)
    return data_transformed, scaler

def normalize_data(X, scaler=None):
    if scaler is None:
        X, scaler = normalize_standard(X)
    else:
        X, scaler = normalize_standard(X, scaler)
    return X, scaler

def Znormalize_tensor(X):
    X_mean=X.mean()
    X_std=X.std()
    return torch.div(torch.sub(X,X_mean),X_std)

def normalize_tensor(X):
    X_max=torch.max(X)
    X_min=torch.min(X)
    if X_max!=X_min:
        return torch.div(torch.sub(X,X_min),torch.sub(X_max,X_min))
    else:
        return X
    
'''
可视化工具：
    二维
'''
def visualization2D(transformed_x, text):
    '''
    x : numpy(number,dim)
    y : numpy(number)
    '''
    #画图

    fig = plt.figure(figsize=(4,4))
    font0 = {'family':'serif','weight':'bold','size':'20'}#定义图的字体
    plt.xlabel("dimention0", font0)
    plt.ylabel("dimention1", font0)
    plt.tick_params(labelsize=20)
    sns.set(font_scale=1.3)
    sns.scatterplot(data = transformed_x, hue='label', x='dim0', y='dim1')
    
    #保存图
    path='shapelets_plots/'+'visualization'+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(path+text+'times'+'.pdf', facecolor=fig.get_facecolor(), bbox_inches="tight")
    transformed_x.to_excel(path+text+'times.xlsx')
    return

'''三维'''
def visualization3D(x, y, i):
    '''
    x : numpy(number,dim)
    y : numpy(number)
    '''
    #数据格式更改
    tsne = TSNE(n_components=3)
    tsne.fit(x)
    x = tsne.fit_transform(x)
    #transformed_x = pd.DataFrame(transformed_x)
    #transformed_x = transformed_x.rename(columns={0:'dim0',1:'dim1'})
    #transformed_x['label'] = y 

    #画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    font0 = {'family':'serif','weight':'bold','size':'14'}#定义图的字体
    for i in range(x.shape[0]):
        ax.scatter(xs=x[i,0], ys=x[i,1], zs=x[i,2], s=20, color=plt.cm.Set3(y[i]/10.), marker='^')
        #Axes3D.text(x[i,0]+0.8, x[i,1]+0.8, x[i,2]+0.8, str(y[i]), zdir=x[i,0])

    #plt.xlabel("dim0", font0)
    #plt.ylabel("dim1", font0)
    #plt.tick_params(labelsize=13)
    plt.show()

    #保存图
    path='shapelets_plots/'+'visualization'+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(path+str(i)+'times'+'.pdf', facecolor=fig.get_facecolor(), bbox_inches="tight")
    return


#def normalize_dim(X,dim):
#    #the dim of X is 3
#    Y=torch.zeros(X.shape[0],X.shape[1],X.shape[2])
#    if dim==0:
#        for i in range(X.shape[1]):
#            for j in range(X.shape[2]):
#                Y[:,i,j]=normalize_tensor(X[:,i,j])
#    if dim==1:
#        for i in range(X.shape[0]):
#            for j in range(X.shape[2]):
#                Y[i,:,j]=normalize_tensor(X[i,:,j])
#    if dim==2:
#        for i in range(X.shape[0]):
#            for j in range(X.shape[1]):
#                Y[i,j,:]=normalize_tensor(X[i,j,:])
#    return Y
    
def normalize_dim2(X,dim):
    #dim:表示在哪个维度上进行归一化
    #the dim of X is 3
    Y=torch.zeros(X.shape[0],X.shape[1])
    if dim==1:
        for i in range(X.shape[1]):
            Y[:,i]=normalize_tensor(X[:,i])
    if dim==0:
        for i in range(X.shape[0]):
            Y[i,:]=normalize_tensor(X[i,:])
    return Y

def Znormalize_dim2(X,dim):
    #dim:表示在哪个维度上进行归一化
    #the dim of X is 3
    Y=torch.zeros(X.shape[0],X.shape[1])
    if dim==1:
        for i in range(X.shape[1]):
            Y[:,i]=Znormalize_tensor(X[:,i])
    if dim==0:
        for i in range(X.shape[0]):
            Y[i,:]=Znormalize_tensor(X[i,:])
    return Y

def normalize_dim(X,dim):
    #the dim of X is 3
    Y=torch.zeros(X.shape[0],X.shape[1],X.shape[2])
    if dim==0:
        for i in range(X.shape[1]):
            for j in range(X.shape[2]):
                X.data[:,i,j]=normalize_tensor(X[:,i,j])
    if dim==1:
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                X.data[i,:,j]=normalize_tensor(X[i,:,j])
    if dim==2:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X.data[i,j,:]=normalize_tensor(X[i,j,:])
    return X

def Znormalize_dim(X,dim):
    #the dim of X is 3
    Y=torch.zeros(X.shape[0],X.shape[1],X.shape[2])
    if dim==0:
        for i in range(X.shape[1]):
            for j in range(X.shape[2]):
                X.data[:,i,j]=Znormalize_tensor(X[:,i,j])
    if dim==1:
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                X.data[i,:,j]=Znormalize_tensor(X[i,:,j])
    if dim==2:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X.data[i,j,:]=Znormalize_tensor(X[i,j,:])
    return X

def sample_ts_segments(X, shapelets_size, n_segments=10000):

    """
    Sample time series segments for k-Means.
    """
    n_ts, n_channels, len_ts = X.shape
    samples_i = random.choices(range(n_ts), k=n_segments)
    
    '''因为n_channels为24，但是只想要一个通道的shapelet，所以改为了1'''
    segments = numpy.empty((n_segments, 1, shapelets_size))
    for i, k in enumerate(samples_i):
        samples_dim = random.choices(range(n_channels), k=1)
        s = random.randint(0, len_ts - shapelets_size)\
        #s=15
        segments[i] = X[k, samples_dim, s:s+shapelets_size]
    return segments

def get_weights_via_kmeans(X, shapelets_size, num_shapelets, n_segments=10000):
    """
    Get weights via k-Means for a block of shapelets.
    """
    segments = sample_ts_segments(X, shapelets_size, n_segments).transpose(0, 2, 1)
    #print(segments.shape,'segments.shape')
    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    #print(clusters.shape,'clusters.shape****************************')
    return clusters

class ShapeletsDistanceLoss(nn.Module):
    """
    Calculates the cosine similarity of a bunch of shapelets to a data set and performs global max-pooling.
    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, dist_measure='euclidean', k=6):
        super(ShapeletsDistanceLoss, self).__init__()
        if not dist_measure == 'euclidean' and not dist_measure == 'cosine':
            raise ValueError("Parameter 'dist_measure' must be either of 'euclidean' or 'cosine'.")
        if not isinstance(k, int):
            raise ValueError("Parameter 'k' must be an integer.")
        self.dist_measure = dist_measure
        self.k = k

    def forward(self, x):
        """
        Calculate the loss as the average distance to the top k best-matching time series.
        @param x: the shapelet transform
        @type x: tensor(float) of shape (batch_size, n_shapelets)
        @return: the computed loss
        @rtype: float
        """
        y_top, y_topi = torch.topk(x.clamp(1e-8), self.k, largest=False if self.dist_measure == 'euclidean' else True,
                                   sorted=False, dim=0)
        # avoid compiler warning
        y_loss = None
        if self.dist_measure == 'euclidean':
            y_loss = torch.mean(y_top)
        elif self.dist_measure == 'cosine':
            y_loss = torch.mean(1 - y_top)
        return y_loss
    
class ShapeletsSimilarityLoss(nn.Module):
    """
    Calculates the cosine similarity of each block of shapelets and averages over the blocks.
    ----------
    """
    def __init__(self):
        super(ShapeletsSimilarityLoss, self).__init__()

    def cosine_distance(self, x1, x2=None, eps=1e-8):
        """
        Calculate the cosine similarity between all pairs of x1 and x2. x2 can be left zero, in case the similarity
        between solely all pairs in x1 shall be computed.
        @param x1: the first set of input vectors
        @type x1: tensor(float)
        @param x2: the second set of input vectors
        @type x2: tensor(float)
        @param eps: add small value to avoid division by zero.
        @type eps: float
        @return: a distance matrix containing the cosine similarities
        @type: tensor(float)
        """
        x2 = x1 if x2 is None else x2
        # unfold time series to emulate sliding window
        x1 = x1.unfold(2, x2.shape[2], 1).contiguous()
        x1 = x1.transpose(0, 1)
        # normalize with l2 norm
        x1 = x1 / x1.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
        x2 = x2 / x2.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8)

        # calculate cosine similarity via dot product on already normalized ts and shapelets
        x1 = torch.matmul(x1, x2.transpose(1, 2))
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        # and average over dims to keep range between 0 and 1
        n_dims = x1.shape[1]
        x1 = torch.sum(x1, dim=1) / n_dims
        return x1

    def forward(self, shapelet_blocks):
        """
        Calculate the loss as the sum of the averaged cosine similarity of the shapelets in between each block.
        @param shapelet_blocks: a list of the weights (as torch parameters) of the shapelet blocks
        @type shapelet_blocks: list of torch.parameter(tensor(float))
        @return: the computed loss
        @rtype: float
        """
        losses = 0.
        for block in shapelet_blocks:
            shapelets = block[1]
            shapelets.retain_grad()
            sim = self.cosine_distance(shapelets, shapelets)
            losses += torch.mean(sim)
        return losses
    
def torch_dist_ts_shapelet(ts, shapelet, cuda=False):
    """
    Calculate euclidean distance of shapelet to a time series via PyTorch and returns the distance along with the position in the time series.
    """
    if not isinstance(ts, torch.Tensor):
        ts = torch.tensor(ts, dtype=torch.float)
    if not isinstance(shapelet, torch.Tensor):
        shapelet = torch.tensor(shapelet, dtype=torch.float)
    if cuda:
        ts = ts.cuda()
        shapelet = shapelet.cuda()
    shapelet=shapelet[:1,:]
    shapelet = torch.unsqueeze(shapelet, 1)
    # unfold time series to emulate sliding window
    ts = ts.unfold(1, shapelet.shape[2], 1)
    # calculate euclidean distance
    dists = torch.cdist(ts, shapelet, p=2)
    #print(dists.shape,'dists.shape')
    if dists.shape[0]>1:
        #阳极电流数据是多维的，min_single_dim是子序列与单个序列的匹配位置，min_total_dim是子序列最匹配的维度
        min_num,min_single_dim = torch.min(dists, dim=1)
        d_min, min_total_dim = torch.min(min_num, 0)
        return (min_single_dim[min_total_dim.item()].item(), min_total_dim.item())
    else:
        #公共数据
        dists = torch.sum(dists, dim=0)
        d_min,d_argmin = torch.min(dists, dim=0)
        return (d_min.item(), d_argmin.item())

def lead_pad_shapelet(shapelet, pos):
    """
    Adding leading NaN values to shapelet to plot it on a time series at the best matching position.
    """
    pad = numpy.empty(pos)
    pad[:] = numpy.NaN
    padded_shapelet = numpy.concatenate([pad, shapelet])
    return padded_shapelet

def record_shapelet_value(ucr_dataset_name,shapelets,X_test,pos,i,j):
    '''将shapelet的值保存到文档里面'''
    path='shapelet_value/'+str(ucr_dataset_name)+str(X_test.shape[0])+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    path_shapelet_value=path='shapelet_value/'+str(ucr_dataset_name)+str(X_test.shape[0])+'/'+'shapelet'+str(j)+'/'
    if not os.path.exists(path_shapelet_value):
        os.mkdir(path_shapelet_value)
    excel_shapelet=lead_pad_shapelet(shapelets[j, 0], pos)
    if X_test.shape[1]>1:
        excel_time_series=X_test[i,pos]
    else:
        excel_time_series=X_test[i]
    excel_shapelet=pd.DataFrame(excel_shapelet)
    excel_time_series=pd.DataFrame(excel_time_series)
    excel_shapelet.to_excel(path_shapelet_value+'shapelet.xlsx')
    excel_time_series.to_excel(path_shapelet_value+'time_series.xlsx')
    return

def plot_sub(i,j,fig,shapelets,X_test,record_data_plot,test_y,ucr_dataset_name):
    '''
    i:num of sample
    j:num of sub_graph
    shapelets:[num of shapelets,1, len_of_shapelets]
    X_test:[num of test,1, len of test]
    '''
    font = {'family': 'Times New Roman',
        'style': 'normal',
        'stretch': 1000,
        'weight': 'bold',
        }
    fig_ax1 = fig.add_subplot(4,int(shapelets.shape[0]/4)+1,j+1)
    plt.subplots_adjust(left=None, bottom=0.05, right=None, top=None, wspace=0.3, hspace=0.5)#wspace 子图横向间距， hspace 代表子图间的纵向距离，left 代表位于图像不同位置
    fig_ax1.text(0.01,0.01,'',fontdict=font)
    fig_ax1.set_title("shapelet"+str(j+1),fontproperties="Times New Roman",)
    if X_test.shape[1]>1:
        _, pos = torch_dist_ts_shapelet(X_test[i], shapelets[j])
        fig_ax1.plot(X_test[i, pos], color='black', alpha=0.02, )
        fig_ax1.plot(lead_pad_shapelet(shapelets[j, 0], _), color='#F03613', alpha=0.02)
        record_shapelet_value(ucr_dataset_name,shapelets,X_test,pos,i,j)
    else:
        fig_ax1.plot(X_test[i, 0], color='black', alpha=0.5)
        _, pos = torch_dist_ts_shapelet(X_test[i], shapelets[j])
        fig_ax1.plot(lead_pad_shapelet(shapelets[j, 0], pos), color='#F03613', alpha=0.5)
        record_shapelet_value(ucr_dataset_name,shapelets,X_test,pos,i,j)
    record_data_plot['fig'+str(j)]['x']=X_test[i, 0]
    record_data_plot['fig'+str(j)]['s']=shapelets[j, 0]
    record_data_plot['fig'+str(j)]['class']=test_y[i]
    record_data_plot['fig'+str(j)]['dim']=pos
    return record_data_plot

def featrue_map(shapelet_transform, y_test, weights, ucr_dataset_name, X_test, shapelet_num):
    '''设置全局字体'''
    pyplot.rcParams['font.sans-serif']='Times New Roman'
    pyplot.rcParams['font.weight']='bold'
    pyplot.rcParams['font.size']=14
    pyplot.rc('xtick',labelsize=10)
    pyplot.rc('ytick',labelsize=10)
    
    fig = pyplot.figure(facecolor='white')
    #fig.set_size_inches(20, 8)
    gs = gridspec.GridSpec(2, 2)
    fig_ax3 = fig.add_subplot(gs[:, :])
    #font0 = FontProperties(family='serif',weight='bold',size=14)
    #fig_ax3.set_title("The decision boundaries learned by the model to separate the two classes.", fontproperties=font0)
    color = {-1:'#00FF00',0: '#F03613', 1: '#7BD4CC', 2: '#00281F', 3: '#BEA42E',4:'#FFC0CB',5:'#FFF0F5',6:'#FF69B4'}
             
    dist_s1=shapelet_transform[:,shapelet_num]
    dist_s2=shapelet_transform[:,shapelet_num+1]
    fig_ax3.scatter(dist_s1, dist_s2, color=[color[l] for l in y_test])
    
    # Create a meshgrid of the decision boundaries
    xmin = numpy.min(shapelet_transform[:, shapelet_num]) - 0.1
    xmax = numpy.max(shapelet_transform[:, shapelet_num]) + 0.1
    ymin = numpy.min(shapelet_transform[:, shapelet_num+1]) - 0.1
    ymax = numpy.max(shapelet_transform[:, shapelet_num+1]) + 0.1
    xx, yy = numpy.meshgrid(numpy.arange(xmin, xmax, (xmax - xmin)/200),
                            numpy.arange(ymin, ymax, (ymax - ymin)/200))
    Z = []
    num_class=len(weights)
    for x, y in numpy.c_[xx.ravel(), yy.ravel()]:
        Z.append(numpy.argmax([weights[i][0]*x + weights[i][1]*y
                               for i in range(num_class)]))
   # Z = numpy.array(Z).reshape(xx.shape)
    #fig_ax3.contourf(xx, yy, Z / 3, cmap=viridis, alpha=0.25)
    fig_ax3.set_xlabel("shapelet"+str(shapelet_num))
    fig_ax3.set_ylabel("shapelet"+str(shapelet_num+1))
    fig_ax3.tick_params(labelsize=13)
    
    path='shapelets_plots/'+str(ucr_dataset_name)+'shapelet'+str(shapelet_num)+'_'+str(shapelet_num+1)+'feature_map'+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    pyplot.savefig(path+'.pdf',format='pdf', facecolor=fig.get_facecolor(), bbox_inches="tight")
    
    #pyplot.savefig(path+'.png', facecolor=fig.get_facecolor(), bbox_inches="tight")
    return 

def plot_shapelets(X_test, shapelets, y_test, shapelet_transform, weights, ucr_dataset_name):
    fig = pyplot.figure(facecolor='white')
    fig.set_size_inches(20, 8)
    dist={}
    nums_shapelets=shapelet_transform.shape[1]
    for i in range(nums_shapelets):
        dist[i]=shapelet_transform[:, i]
   # gs = gridspec.GridSpec(12, 8)
    #fig_ax1 = fig.add_subplot(gs[0:3, :4])
    record_data_plot={}
    for i in range(nums_shapelets):
        record_data_plot['fig'+str(i)]={}
   # fig_ax1.set_title("top of its 1 best matching time series.")
    for j in range(nums_shapelets):
        for i in numpy.argsort(dist[j])[:1]:
            record_data_plot = plot_sub(i,j,fig,shapelets,X_test,record_data_plot,y_test,ucr_dataset_name)
#    
    caption = """Shapelets learned for the pot volt dataset plotted on top of the best matching time series."""
    pyplot.figtext(0.5, -0.02, caption, wrap=True, horizontalalignment='center', fontsize=20, family='Times New Roman')
    path='shapelets_plots/'+str(ucr_dataset_name)+str(X_test.shape[0])+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    pyplot.savefig(path+'.pdf', format='pdf', facecolor=fig.get_facecolor(), bbox_inches="tight")
    #pyplot.savefig(path+'.png', facecolor=fig.get_facecolor(), bbox_inches="tight")
    pyplot.show()
    #画所有shapelets映射的特征图
    #for shapelet_num in range(0,len(shapelet_transform[1])-1,2):
    #     featrue_map(shapelet_transform, y_test, weights, ucr_dataset_name, X_test, shapelet_num)
    return record_data_plot
