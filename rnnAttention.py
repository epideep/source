# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 08:39:46 2018

@author: Bijaya
"""

import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_internal, emd_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #encoder
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        #attention model
        self.attn = torch.nn.Parameter(torch.randn(hidden_size , 1))#(20)*1
        
        #decoder
        self.dropout = nn.Dropout(p = 0)
        self.fc = nn.Linear(hidden_size+emd_size, num_internal)
        self.fc2 = nn.Linear(num_internal, num_internal)
        self.fc3 = nn.Linear(num_internal, num_internal)
        self.linear = torch.nn.Linear(num_internal, 1)
        self.activation = nn.LeakyReLU()
    
    def forward(self, x, emd):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        #print("*"*10)
        # Forward propagate RNN, lstm output shape: output,(h_n,c_n)
        out, (hidden, cell)  = self.lstm(x, (h0, c0))
        #print("out", out.shape)
        #print("hidden", hidden[1].shape)
        hidden_state = hidden[1].unsqueeze(0)
        hidden_state = hidden_state.squeeze(0).unsqueeze(2)
        #print("m2", merged_state.shape)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(out, hidden_state)
        #print("weight1", weights.shape)
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        #print("weight2", weights.shape)
        out = torch.bmm(torch.transpose(out, 1, 2), weights).squeeze(2)
        
        
        #print("Output/hidden size = ",out[0].size(),hidden[0][0].size(),out.size(),hidden[0].size(),hidden[1].size(),out[:,-1,:].size())
        #x.size()           : torch.Size([13, 20, 1])
        #emd.size()         : torch.Size([13, 20])
        #print("out", out.shape)      #: torch.Size([20, 20])
        #print("hidden", hidden.shape)
        #hidden[0][0].size(): torch.Size([13, 20])
        #out.size()         : torch.Size([13, 20, 20])
        #hidden[0].size()   : torch.Size([2, 13, 20])
        #hidden[1].size()   : torch.Size([2, 13, 20])
        #out[:,-1,:].size() : torch.Size([13, 20])
        
        # Attention model
        #attnExpand = self.attn.expand(x.size()[0],self.hidden_size,1)   # expand the attn from 20*1 to 13*20*1
        #out_attn = torch.bmm(out, attnExpand)# 13*20*20 * 13*20*1, before we only use 13*1*20 from the out, now we use the full 20
        #print("self.attn = ", self.attn)
        
        out = out.unsqueeze(2)
     
        # Dropout
        out = self.dropout(out)
        #print("emd", emd.size(), "out", out.size())

        out_new = torch.squeeze(out, 2)
        #out_new = torch.squeeze(out)        #last dimension is 1 and redundant
        #print("emd", emd.size(), "out_new", out_new.size())
        #print("emd", emd.shape)
        #print("out_new", out_new.shape)
        out1 = torch.cat((out_new, emd),1)
        
        
        out2 = self.activation(self.fc(out1))
        out3 = self.activation(self.fc2(out2))
        out4 = self.activation(self.fc3(out3))
        
        
        out = self.linear(out4)
        return out


class RNNTime(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_internal, emd_size, out_size):
        super(RNNTime, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #encoder
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        #attention model
        self.attn = torch.nn.Parameter(torch.randn(hidden_size , 1))#(20)*1
        
        #decoder
        self.dropout = nn.Dropout(p = 0)
        self.fc = nn.Linear(hidden_size+emd_size, num_internal)
        self.fc2 = nn.Linear(num_internal, num_internal)
        self.fc3 = nn.Linear(num_internal, num_internal)
        self.fc4 = nn.Linear(num_internal, out_size)
        self.stmax = nn.Softmax()
        self.activation = nn.LeakyReLU()
    
    def forward(self, x, emd):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        
        #print("*"*10)
        # Forward propagate RNN, lstm output shape: output,(h_n,c_n)
        out, (hidden, cell)  = self.lstm(x, (h0, c0))
        #print("out", out.shape)
        #print("hidden", hidden[1].shape)
        hidden_state = hidden[1].unsqueeze(0)
        hidden_state = hidden_state.squeeze(0).unsqueeze(2)
        #print("m2", merged_state.shape)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(out, hidden_state)
        #print("weight1", weights.shape)
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        #print("weight2", weights.shape)
        out = torch.bmm(torch.transpose(out, 1, 2), weights).squeeze(2)
        
        
        
        
        #print("Output/hidden size = ",out[0].size(),hidden[0][0].size(),out.size(),hidden[0].size(),hidden[1].size(),out[:,-1,:].size())
        #x.size()           : torch.Size([13, 20, 1])
        #emd.size()         : torch.Size([13, 20])
        #out[0].size()      : torch.Size([20, 20])
        #hidden[0][0].size(): torch.Size([13, 20])
        #out.size()         : torch.Size([13, 20, 20])
        #hidden[0].size()   : torch.Size([2, 13, 20])
        #hidden[1].size()   : torch.Size([2, 13, 20])
        #out[:,-1,:].size() : torch.Size([13, 20])
        
        # Attention model
        #attnExpand = self.attn.expand(x.size()[0],self.hidden_size,1)   # expand the attn from 20*1 to 13*20*1
        #out_attn = torch.bmm(out, attnExpand)# 13*20*20 * 13*20*1, before we only use 13*1*20 from the out, now we use the full 20
        #print("self.attn = ", self.attn)
        
        out = out.unsqueeze(2)
     
        # Dropout
        out = self.dropout(out)
        #print("emd", emd.size(), "out", out.size())

        out_new = torch.squeeze(out, 2) 

        #out_new = torch.squeeze(out)        #last dimension is 1 and redundant
        #print("emd", emd.size(), "out_new", out_new.size())
        out1 = torch.cat((out_new, emd),1)  #merge out_new and emd through the dimension 1: [20+13,20]
        out2 = self.activation(self.fc(out1))
        out3 = self.activation(self.fc2(out2))
        out4 = self.activation(self.fc3(out3))
        
        
        out = self.stmax(self.fc4(out4))
        
        return out





