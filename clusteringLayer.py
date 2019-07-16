# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 12:24:29 2018

@author: Bijaya
"""
from sklearn.cluster import KMeans
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.parameter import Parameter


class ClusteringLayer(nn.Module):
    
    def __init__(self,  inputs_dim, n_clusters, weights = None):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.input_dims = inputs_dim
        self.weights = weights
        self.alpha = 1.0
        
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        
    def init_weights(self, data):
        kmeans = KMeans(n_clusters = self.n_clusters, n_init=20)
        kmeans.fit_predict(data.data)
        self.weights = nn.Parameter(torch.Tensor(kmeans.cluster_centers_))
        #print(self.weights)
    
    def forward(self, z):
        
        if( type(self.weights) is type(None)):
            self.init_weights(z)

        # cluster
        
        
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.weights, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q
        
        
        
        
        
        
    
    