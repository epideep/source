# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 16:33:30 2018

@author: Bijaya
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from rnnAttention import RNN
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def mse_loss(inp, target):
    return torch.sum((inp - target)**2) / inp.data.nelement()

def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="leakyReLU":
            net.append(nn.LeakyReLU())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


        

class DeepClustering(nn.Module):
    def __init__(self, input1_dim, embed1_dim, input2_dim, embed2_dim, n_centroids, encode_layers=[500, 200], decode_layers=[ 200, 500], mapping_layers=[100,200, 100]):
        super(self.__class__, self).__init__()
        self.input1_dim = input1_dim
        self.embed1_dim = embed1_dim
        self.n_centroids = n_centroids
        self.input2_dim = input2_dim
        self.embed2_dim = embed2_dim
        
        self.first_encoder = buildNetwork([input1_dim]+encode_layers+[embed1_dim])
        self.first_decoder = buildNetwork([embed1_dim]+encode_layers+[input1_dim])
        
        self.first_cluster_layer = Parameter(torch.Tensor(n_centroids, embed1_dim))
        torch.nn.init.xavier_normal_(self.first_cluster_layer.data)
        
       
        
        self.second_encoder = buildNetwork([input2_dim]+encode_layers+[embed2_dim])
        self.second_decoder = buildNetwork([embed2_dim]+encode_layers+[input2_dim])
        
        
        self.second_cluster_layer = Parameter(torch.Tensor(n_centroids, embed2_dim))
        torch.nn.init.xavier_normal_(self.second_cluster_layer.data)
        
        
        self.mapper = buildNetwork([embed1_dim] + mapping_layers+[embed2_dim], activation="LeakyReLu")
        
        self.regressor = RNN(1, 20, 2, 20, embed1_dim)
        self.alpha = 1
    
    def pre_train(self, qdata, fdata):
        x1 = qdata
        x2 = fdata
        
        
        optimizer = torch.optim.Adam(self.parameters())
        for epoch in range(1000):        
            z1 = self.first_encoder(x1)
            x1_bar = self.first_decoder(z1)
    
            optimizer.zero_grad()
            loss = F.mse_loss(x1_bar, x1)
            #print("Pretrain_1:", epoch, loss)
            loss.backward()
            optimizer.step()
        for epoch in range(1000):  
            z2 = self.second_encoder(x2)
            x2_bar = self.second_decoder(z2)
            optimizer.zero_grad()
            loss = F.mse_loss(x2_bar, x2)
            #print("Pretrain_2:", epoch, loss)
            loss.backward()
            optimizer.step()
    
    
    def forward_clustering_first(self, x1):
        z1 = self.first_encoder(x1)
        x1_bar = self.first_decoder(z1)
        
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z1.unsqueeze(1) - self.first_cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x1_bar, q ,z1
    
    def forward_clustering_second(self, x2):
        z2 = self.second_encoder(x2)
        x2_bar = self.second_decoder(z2)
        
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z2.unsqueeze(1) - self.second_cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x2_bar, q , z2
        
        
        
    def loss_function(self, x1, recon_x1, x2, recon_x2, rnn_data, rnn_labels, q1, q2, emb1, emb2):

        p1 = target_distribution(q1)

        loss_val1 = F.kl_div(q1.log(), p1.detach())
        

        p2 = target_distribution(q2)
        loss_val2 = F.kl_div(q2.log(), p2.detach())
        
        #q2 = self.second_clustering.forward(emb2)
        
        pred = self.regressor.forward(rnn_data, self.mapper(emb1))
        
        pred_loss = F.mse_loss(pred, rnn_labels)
        
        translated_emb = self.mapper(emb1)
        #print(pred_loss.item())
        
        #loss = pred_loss
        loss = F.mse_loss(x1, recon_x1)+F.mse_loss(x2, recon_x2)+ F.mse_loss(translated_emb, emb2) + loss_val1.data[0] + loss_val2.data[0]+ pred_loss
        #print(F.mse_loss(x1, recon_x1).data[0], F.mse_loss(x2, recon_x2).data[0],  F.mse_loss(emb1, emb2).data[0], loss_val1.data[0], loss_val2.data[0], pred_loss.data[0])
        #print(loss)
        return loss
    
    def fit(self, qdata,  fdata, rnn_data, rnn_labels, lr = 0.001, num_epoch = 10):
        
        in_q_data = Variable(torch.Tensor(qdata), requires_grad= True)
        in_f_data = Variable(torch.Tensor(fdata), requires_grad= True)
        
        self.pre_train(in_q_data, in_f_data)
        
        kmeans = KMeans(n_clusters=self.n_centroids, n_init=10)
        z1 = self.first_encoder(in_q_data)
        kmeans.fit_predict(z1.detach().numpy())
        self.first_cluster_layer.data = torch.tensor(kmeans.cluster_centers_)
        
        z2 = self.second_encoder(in_f_data)
        kmeans.fit_predict(z2.detach().numpy())
        self.second_cluster_layer.data = torch.tensor(kmeans.cluster_centers_)
    
    
        lossfile = open('loss.csv','w')
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        
        rnn_data = Variable(torch.Tensor(rnn_data), requires_grad= True)
        
        rnn_labels = Variable(torch.Tensor(rnn_labels)).unsqueeze(1)
        
        for epoch in range(num_epoch):
            
            x1_bar, q1, z1 = self.forward_clustering_first(in_q_data)
            
            x2_bar, q2, z2 = self.forward_clustering_second(in_f_data)
            
            
            loss = self.loss_function(in_q_data,x1_bar, in_f_data, x2_bar, rnn_data, rnn_labels, q1, q2, z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(epoch, loss.item())
            lossfile.write(str(epoch)+','+str(loss.item())+'\n')
        lossfile.close()
       
    def predict(self, data, rnn_data):
        in_data = Variable(torch.Tensor(data))
        in_rnn_data = Variable(torch.Tensor(rnn_data))
        return self.regressor.forward(in_rnn_data, self.mapper(self.first_encoder(in_data)))
    
    
    def embed(self, data, rnn_data):
        in_data = Variable(torch.Tensor(data))
        return self.mapper(self.first_encoder(in_data))
    