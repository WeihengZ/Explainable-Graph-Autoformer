import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os
from turtle import forward

class simple_GCN(nn.Module):

    def __init__(self, adj, in_F, out_F, DEVICE):
        super(simple_GCN, self).__init__()

        self.W = nn.Parameter(torch.rand(3, in_F, out_F).float().to(DEVICE))
        self.adj = adj
    
    def forward(self, x, attn):

        '''
        input x: (B,N,T,F_in)
        attn: (B,N,N)
        adj: (N,N)
        '''

        x_new1 = torch.matmul(x, self.W[0])    # return (B,N,T,F_out)
        
        adj2 = attn * self.adj.unsqueeze(0)
        x_new2 = torch.relu(torch.einsum('bij,bjkl->bikl', adj2, x))
        x_new2 = torch.matmul(x_new2, self.W[1])    # return (B,N,T,F_out)

        adj3 = attn * torch.matmul(self.adj,self.adj).unsqueeze(0)
        x_new3 = torch.relu(torch.einsum('bij,bjkl->bikl', adj3, x))
        x_new3 = torch.matmul(x_new3, self.W[2])    # return (B,N,T,F_out)
        
        x_new = x_new1 + x_new2 + x_new3    # return (B,N,T,F_out)

        return x_new


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, pred_len, factor=1):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        
        # define the predicted length
        self.pred_len = pred_len
        # this is used to control the magnitude of the correlation function
        self.corr_factor = nn.Linear(1,1)

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        values: (B,N,T_long)
        corr: (B,N,T_long)

        return: (B,N,pred_len)
        """
        batch = values.shape[0]
        nodes = values.shape[1]
        length = values.shape[2]

        # index init, return (B,N,pred_len)
        # define the segment we want to aggregate
        init_index = torch.arange(self.pred_len, 2*self.pred_len).unsqueeze(0).unsqueeze(0).repeat(batch, nodes, 1).to(values.device)

        # return scalar, calculate number of time segment we aggregate
        top_k = int(self.factor * np.log(corr.shape[-1]))
        # return (B,N,top_k)
        weights, delay = torch.topk(corr, top_k, dim=-1)

        # calculate temporal attentions, return (B,N,top_k)
        tmp_corr = torch.softmax(weights, dim=-1)

        # aggregation
        tmp_values = torch.cat((values, torch.zeros_like(init_index), torch.zeros_like(init_index)), -1)    # return (B,N,T_long+2*pred_len)
        delays_agg = torch.zeros_like(init_index)    # return (B,N,pred_len)
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)    # return (B,N,pred_len)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)    # return (B,N,pred_len)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))    # return (B,N,pred_len)
        
        return delays_agg, delay, tmp_corr


    def forward(self, queries, keys, values):

        '''
        input
        Q: (B,N,T_short = 1 days, num_pred_len)
        K,V: (B,N,T_long = 1 weeks, num_pred_len)

        return (B,N,pred_len,F)
        '''
        # obtain the shape
        _, _, T_short, num_pred_len = queries.shape
        B, N, T_long, _ = values.shape

        # define a list to store the prediction, major time shift and attention scores
        outs = []
        delays = []
        attns = []

        for i in range(num_pred_len):

            Q = queries[...,i]    # return (B,N,T_short)
            K = keys[...,i]    # return (B,N,T_long)
            V = values[...,i]    # return (B,N,T_long)

            # expand the temporal dimension of queries
            Q = torch.cat((Q, torch.zeros(B,N,K.shape[-1] - Q.shape[-1]).to(K.device)), -1)    # return (B,N,T_long)

            # period-based dependencies (Autoformer)
            q_fft = torch.fft.rfft(Q.contiguous(), dim=-1)    # return (B,N,T_long)
            k_fft = torch.fft.rfft(K.contiguous(), dim=-1)    # return (B,N,T_long)
            res = q_fft * torch.conj(k_fft)    # return (B,N,T_long)
            corr = torch.fft.irfft(res, dim=-1)    # return (B,N,T_long)
            # scalar down the correlation score
            corr = self.corr_factor(corr.unsqueeze(-1)).squeeze(-1)   # return (B,N,T_long)

            # apply temporal aggregation, return (B,N,pred_len), (B,N,topk), (B,N,topk)
            V, delay, t_attn_score = self.time_delay_agg_full(V, corr)

            # store the information
            outs.append(V)
            delays.append(delay)
            attns.append(t_attn_score)

        return outs, delays, attns


class spatial_attns(nn.Module):

    def __init__(self, time_step, feature, DEVICE):
        super(spatial_attns, self).__init__()

        self.Q_mapping = nn.Linear(time_step*feature, time_step)
        self.K_mapping = nn.Linear(time_step*feature, time_step)
        self.W = nn.Parameter(torch.rand(time_step, time_step)).float().to(DEVICE)

    def forward(self, x):

        '''
        x: (B,N,T_short,F)
        '''
        B,N,_,_ = x.shape
        x = x.view(B,N,-1)

        Q = torch.relu(self.Q_mapping(x))
        K = torch.relu(self.K_mapping(x))

        A = torch.matmul(Q, self.W)    # return (B,N,T)

        # (B,N,T), (B,T,N) -> (B,N,N)
        A = torch.einsum('bij,bjk->bik', A, K.permute(0,2,1))

        return A



class cell(nn.Module):
    def __init__(self, pred_len, adj, num_pred_len, DEVICE):
        super(cell, self).__init__()

        # define Query, Key and Value mapping
        # for each hour in the future, we have one query and key, which will give us a specific attention score 
        # for each hour in the future, we have one "value" (hidden state)
        self.Q_mapping = nn.Sequential(nn.Linear(1,64), nn.ReLU(), nn.Linear(64, num_pred_len)) 
        self.K_mapping = nn.Sequential(nn.Linear(1,64), nn.ReLU(), nn.Linear(64, num_pred_len))
        self.V_mapping = nn.Sequential(nn.Linear(1,64), nn.ReLU(), nn.Linear(64, num_pred_len))

        # this is the module to calculate temporal attentions
        self.cross_corr = AutoCorrelation(pred_len=pred_len)

        # define one GNN layer, in feature and out feature is 1
        self.GCN = simple_GCN(adj, 1, 1, DEVICE)

        # define a layer to calculate self attention
        self.spatial_atttns = spatial_attns(pred_len, 1, DEVICE)

    def forward(self, recent, hist_ref):

        '''
        input
        recent: (B,N,T_short,F)
        hist_ref: (B,N,T_long,F)

        return list of (B,N,pred_len), (B,N,topk), (B,N,topk), (B,N,N)
        '''
        
        Q = self.Q_mapping(recent)    # return (B,N,T,num_pred_len)
        K = self.K_mapping(hist_ref)    # return (B,N,T,num_pred_len)
        V = self.V_mapping(hist_ref)    # return (B,N,T,num_pred_len)

        # return list of (B,N,pred_len), (B,N,topk), (B,N,topk)
        outs, delays, t_attn_scores = self.cross_corr(Q, K, V)    # return (B,N,pred_len,F)
        
        # define a list to store new prediction, spatial attention score
        new_out = []
        sp_attn_scores = []

        # apply spatial attention to the output
        for k in range(len(outs)):
            out = outs[k]
            sp_attn_score = self.spatial_atttns(out.unsqueeze(-1))    # return (B,N,N)

            out = self.GCN(out.unsqueeze(-1), sp_attn_score).squeeze(-1)    # return (B,N,pred_len)

            new_out.append(out)    # return list of (B,N,pred_len)
            sp_attn_scores.append(sp_attn_score)   # final return list of (B,N,N)
        
        return new_out, (delays, t_attn_scores, sp_attn_scores)


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, adj, configs, DEVICE):
        super(Model, self).__init__()

        # length of each hour
        self.pred_len = configs.pred_len    # length of the predicted sequence 

        # total number of hour for prediciton
        self.num_pred_len = configs.num_pred_len

        # main compoent to perform prediction
        self.cell = cell(pred_len=self.pred_len, adj=adj, num_pred_len=configs.num_pred_len, DEVICE=DEVICE)

        # encode time informaiton into the model
        self.t_embedding = nn.Sequential(nn.Linear(2,32), nn.ReLU(), nn.Linear(32,1))

    
    def forward(self, x_enc, t):
        '''
        input
        x_enc: (B, T=7 weeks, N)
        t: (B,T)

        return: (B, T = num_pred_len * pred_len, N)
        '''

        # time embedding, return (B,N,T) 
        B,T,N = x_enc.shape
        t = t.unsqueeze(-1).repeat(1,1,N)
        x_enc = torch.cat((x_enc.unsqueeze(-1), t.unsqueeze(-1)), -1)    # (B,T,N,2)
        x_enc = self.t_embedding(x_enc).squeeze(-1).permute(0,2,1)

        # extract decoder input: we use yesterday traffic data as decoder input, return (B,N,T)
        x_recent = x_enc[:,:,-12*24:]

        # perform prediction in one time, prediction is in the shape of (B,N,pred_len)
        out, explain = self.cell(x_recent.unsqueeze(-1), x_enc.unsqueeze(-1))    

        # cat prediciton of each segment together, return (B,N,pred_horizon)
        final_out = torch.cat(tuple(out), 2)

        return final_out.permute(0,2,1), explain

        
