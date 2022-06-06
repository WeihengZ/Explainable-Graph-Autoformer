import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        valuse (B, num_head, F, T)
        corr (B, num_head , F, T)

        return (B, num_head, F, T)
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        # find top k
        top_k = int(self.factor * math.log(length)) # train a parameter int, number of period we use

        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)   # return (B, T)

        # first calculate the mean along the batch size , then return the k largest number
        # store the low-frequency (large period) information
        # return an index of the period we use: shape = (top_k)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]   
        
        # for each datapoint in the batch, pick the period value (B, top_k)
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)

        # update corr
        # all period value is positive, larger period(lower frequency) information deserve more attention weight
        tmp_corr = torch.softmax(weights, dim=-1)

        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)   # returrn (B, num_head, F, T)
            # (B, num_head, F, T) + (B, num_head, F, T) * (B, num_head, F, T) -> (B, num_head, F, T)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)) 
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):

        '''
        input
        queries: (batch, time_length, number_of_head, key_embed_dimension)
        keys: (batch, time_length, number_of_head, key_embed_dimension)
        values: (batch, time_length, number_of_head, value_embed_dimension)

        return list = ((B, T, num_head, F), (B, T, num_head, F))

        we can actually apply attention mask on it !
        '''

        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # reshape to the same size
        if L > S:   
            # if query size > value size, expand the value size and key size to query size (used for autocorrelation in decoder)
            ''' this happens when it is used as transfer autocorrelation'''

            # expand the time step of values and keys
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()    
            
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        # permute the time dimension to last dimension, apply fft on it
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)    # the output contains only the positive frequencies (B,num_head,F,T)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)    # the output contains only the positive frequencies (B,num_head,F,T)
        res = q_fft * torch.conj(k_fft)    # (B,H,F,T) elementwise * (B,num_head,F,T) -> (B,num_head,F,T)
        corr = torch.fft.irfft(res, dim=-1)    # back to time domain (B,num_head,F,T)

        # time delay agg
        if self.training:
            # (B, T, num_head, F) -> (B, num_head, F, T) + (B, num_head, F,T) -> (B, num_head, F, T) -> (B, T, num_head, F)
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            # (B, T, num_head, F) -> (B, num_head, F, T) + (B, num_head, F,T) -> (B, num_head, F, T) -> (B, T, num_head, F)
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        # determine whether output the attention 
        # shape = ((B, T, num_head, F), (B, T, num_head, F))
        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)

class AutoCorrelation_spa_tem(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, adj, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation_spa_tem, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.adj = adj

    def time_delay_agg_training(self, values, queries, keys):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        valuse (B, num_head, F, T)
        corr (B, num_head , F, T)

        return (B, num_head, F, T)
        """
        Batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        # calculate correlation
        ''' change here '''
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)    # the output contains only the positive frequencies (B,num_head,F,T)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)    # the output contains only the positive frequencies (B,num_head,F,T)
        res = q_fft * torch.conj(k_fft)    # (B,H,F,T) elementwise * (B,num_head,F,T) -> (B,num_head,F,T)
        corr = torch.fft.irfft(res, dim=-1)    # back to time domain (B,num_head,F,T)

        # find top k
        top_k = int(self.factor * math.log(length)) # train a parameter int, number of period we use
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)   # return (B, T)
        # first calculate the mean along the batch size , then return the k largest number
        # store the low-frequency (large period) information
        # return an index of the period we use: shape = (top_k)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]

        # calculate spatial attention
        ''' change here '''
        spatial_attns = []
        q_spa = queries.view(Batch, head*channel, length)
        k_spa = keys.view(Batch, head*channel, length)
        # (B, num_head*F = num_node, T)
        for i in range(top_k):
            k_spa_pattern = torch.roll(k_spa, -int(index[i]), -1).permute(0,2,1)    # return (B, T, N)
            q_spa_pattern = q_spa    # return (B, N, T)
            spatial_attn = torch.einsum('bij,bjk->bik', q_spa_pattern, k_spa_pattern)    # return (B,N,N)
            spatial_attn.softmax(dim=-1) * self.adj.unsqueeze(0).repeat(Batch,1,1)
            spatial_attns.append(spatial_attn)
        # node aggregation
        v_spa = values.view(Batch, head*channel, length)
        node_aggs = []
        for i in range(top_k):
            node_agg = torch.einsum('bij,bjk->bik', spatial_attns[i], v_spa)    # return (B, N, T)
            node_agg = node_agg.view(Batch, head, channel, length)    # (B,H,F,T)
            node_aggs.append(node_agg)

        # for each datapoint in the batch, pick the period value (B, top_k)
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)

        # update corr
        # all period value is positive, larger period(lower frequency) information deserve more attention weight
        tmp_corr = torch.softmax(weights, dim=-1)

        # aggregation
        ''' replace the code here '''
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(node_aggs[i], -int(index[i]), -1)   # returrn (B, num_head, F, T)
            # (B, num_head, F, T) + (B, num_head, F, T) * (B, num_head, F, T) -> (B, num_head, F, T)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)) 


        # # aggregation v2
        # tmp_values = values
        # delays_agg = torch.zeros_like(values).float()
        # for i in range(top_k):
        #     pattern = torch.roll(tmp_values, -int(index[i]), -1)   # returrn (B, num_head, F, T)
        #     # (B, num_head, F, T) + (B, num_head, F, T) * (B, num_head, F, T) -> (B, num_head, F, T)
        #     delays_agg = delays_agg + pattern * \
        #                  (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)) 
        
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):

        '''
        input
        queries: (batch, time_length, number_of_head, key_embed_dimension)
        keys: (batch, time_length, number_of_head, key_embed_dimension)
        values: (batch, time_length, number_of_head, value_embed_dimension)

        return list = ((B, T, num_head, F), (B, T, num_head, F))

        we can actually apply attention mask on it !
        '''

        '''
        explanation for this function:
            extract the most valuable rolloing step, however, for each datapoint in a batch,
        it may have different period information, so we need to have exactly correlation information 
        to calculate temporal attention score. We will aggregate the temporal information to enhance
        the low frequency information.
        
        '''

        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # reshape to the same size
        if L > S:   
            # if query size > value size, expand the value size and key size to query size (used for autocorrelation in decoder)
            ''' this happens when it is used as transfer autocorrelation'''

            # expand the time step of values and keys
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()    
            
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        # permute the time dimension to last dimension, apply fft on it
        ''' put the code inside the aggregation'''
        # q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)    # the output contains only the positive frequencies (B,num_head,F,T)
        # k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)    # the output contains only the positive frequencies (B,num_head,F,T)
        # res = q_fft * torch.conj(k_fft)    # (B,H,F,T) elementwise * (B,num_head,F,T) -> (B,num_head,F,T)
        # corr = torch.fft.irfft(res, dim=-1)    # back to time domain (B,num_head,F,T)


        # time delay agg
        ''' replace the code here'''
        V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), queries, keys).permute(0, 3, 1, 2)
        # if self.training:
        #     # (B, T, num_head, F) -> (B, num_head, F, T) + (B, num_head, F,T) -> (B, num_head, F, T) -> (B, T, num_head, F)
        #     V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), queries, keys).permute(0, 3, 1, 2)
        # else:
        #     # (B, T, num_head, F) -> (B, num_head, F, T) + (B, num_head, F,T) -> (B, num_head, F, T) -> (B, T, num_head, F)
        #     V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), queries, keys).permute(0, 3, 1, 2)

        # determine whether output the attention 
        # shape = ((B, T, num_head, F), (B, T, num_head, F))
        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)

class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        '''
        queries, keys, values: shape = (B,T,hidden_dim)
        ! when we apply it as transfer correlation, queries may have longer time span

        return : list = [(B, T, hidden_dim), (B, T, num_head, F)]
        '''
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        ''' for a given correlated time series, map it to multiple head feature'''

        queries = self.query_projection(queries).view(B, L, H, -1)  # (B, T, num_head, F)
        keys = self.key_projection(keys).view(B, S, H, -1) # (B, T, num_head, F)
        values = self.value_projection(values).view(B, S, H, -1) # (B, T, num_head, F)

        # return list = [(B, T, num_head, F), (B, T, num_head, F)]
        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        # (B, T, num_head, F) -> (B, T, num_head*F)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn    # return (B, T, hidden_dim), (B, T, num_head, F)


