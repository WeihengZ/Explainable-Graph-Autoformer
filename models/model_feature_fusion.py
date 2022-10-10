from xxlimited import new
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class single_scale(nn.Module):

    def __init__(self, d_model, out_dim, patchsize):
        super(single_scale, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=out_dim, kernel_size=(1,patchsize), stride=(1,patchsize), bias=False),
            # nn.GELU(),
            # nn.Conv2d(in_channels=256, out_channels=out_dim, kernel_size=(1,1), stride=(1,1), bias=False),
        )
    
    def forward(self, x):

        '''
        input
        x: (B,N,T = 1 week,F)

        return 
        (B,N,T/patchsize,F)
        '''

        # apply convolution
        x = x.permute(0,3,1,2)    # (B, F, N, T)
        x = self.conv(x).permute(0,2,3,1)    # (B,N,T,out_dim)

        return x

class full_attns(nn.Module):

    def __init__(self, in_dim1, in_dim2, adj, hidden=32, num_head=4):
        super(full_attns, self).__init__()

        assert hidden % num_head == 0, 'hidden should be the multiplier of num_head'
        self.W_mapping1 = nn.Linear(in_dim2, 1, bias=False)
        self.W_mapping2 = nn.Linear(in_dim1, hidden, bias=False)
        self.A_mapping = nn.Linear(2*int(hidden/num_head)+1, 1, bias=False)

        self.num_heads = num_head
        self.adj = adj.unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.num_heads,1)    # (N,N,H,1)
        self.activate = nn.GELU()


    def forward(self, x):

        '''
        x: (B, N, F, T)
        '''
        B, nodes, features, timesteps = x.shape

        Q = self.W_mapping2(self.activate(self.W_mapping1(x).squeeze(-1)))   # (B,N,hidden)
        Q = Q.unsqueeze(2).repeat(1,1,nodes,1)    # (B,N,N,hidden)
        K = Q.permute(0,2,1,3)    # (B,N,N,hidden)

        # split the heads
        Q = Q.view(B, nodes, nodes, self.num_heads, -1)    # (B,N,N,H,F/H)
        K = K.view(B, nodes, nodes, self.num_heads, -1)    # (B,N,N,H,F/H)
        adj = self.adj.unsqueeze(0).repeat(B,1,1,1,1)    # (B,N,N,H,1)

        # combine and calculate scores
        combine = torch.cat((Q, K, adj), -1)    # (B,N,N,H,2*F/H+1)
        scores = self.activate(self.A_mapping(combine).squeeze(-1))    # (B,N,N,H)
        scores = scores.permute(0,3,1,2)    # (B,H,N,N)

        # for stability
        # scores = F.softmax(scores, -1)    # (B,H,N,N)

        return scores

class GCN(nn.Module):

    def __init__(self, adj, out_len, configs):
        super(GCN, self).__init__()

        feature_dim = configs.d_model
        num_head = configs.n_heads

        assert feature_dim % num_head == 0, 'feature dimension should be multiplier of number of heads'
        self.attn_cal = full_attns(in_dim1=feature_dim, in_dim2=out_len, adj=adj, hidden=32, num_head=num_head)
        self.W = nn.Linear(feature_dim, int(feature_dim/num_head), bias=False)
        
        self.num_head = num_head
        self.adj = adj.unsqueeze(0).repeat(self.num_head,1,1)    # (H,N,N)
        self.zero_vec = -9e15 * torch.ones_like(self.adj)   # (H,N,N)

        self.out_mapping = nn.Linear(feature_dim, feature_dim)

        self.activate = nn.GELU()
    
    def forward(self, x):

        '''
        input x: (B,N,T,F)
        adj: (N,N)

        return: (B,N,T,F)
        '''

        B, nodes, timesteps, feature = x.shape

        # calculate basic components
        attns = self.attn_cal.forward(x.permute(0,1,3,2))    # (B,H,N,N)
        adj = self.adj.unsqueeze(0).repeat(B,1,1,1)    # (B,H,N,N)
        mask = self.zero_vec.unsqueeze(0).repeat(B,1,1,1)    # (B,H,N,N)

        # calculate spatial attn score
        adj2 = torch.where(adj > 0, attns, mask)    # (B,H,N,N)
        adj2 = F.softmax(adj2, -1)

        # feature aggregation
        x_new = self.W(x)    # return (B,N,T,F/H)
        x_new = self.activate(torch.einsum('bhij,bjkl->bhikl', adj2, x_new))    # return (B,H,N,T,F/H)
        x_new = x_new.permute(0,2,3,4,1)    # return (B,N,T,F/H,H)
        x_new = x_new.reshape(B, nodes, timesteps, -1)    # return (B,N,T,F)
        x_new = self.out_mapping(x_new)    # return (B,N,T,F)
     
        return x_new, adj2

class AutoCorrelation(nn.Module):
    """
    
    """
    def __init__(self, device, out_len, patch, query_len, configs):
        super(AutoCorrelation, self).__init__()

        # define query length
        self.query_length = query_len

        # define the output len
        self.out_len = out_len

        # define predict length
        self.pred_len = configs.pred_len

        # define time steps
        self.time_steps = configs.time_steps

        # assert self.pred_len % self.out_len == 0, 'prediction length must be the multiplier of output length'

        # define the device
        self.device = device

        # defien patch size
        self.patch = patch
    
        # define top_k
        self.topk = configs.topk

        # define number of head
        self.num_head = configs.n_heads

        # define feature dimension
        feature_dim = configs.d_model

        # define a zeros tensor
        self.zeros = torch.zeros(configs.num_nodes, configs.time_steps, configs.d_model).to(self.device)

        # define module for "cal_QKV"
        self.Q_mapping = single_scale(d_model=feature_dim+1, out_dim=self.num_head, patchsize=patch)
        self.K_mapping = single_scale(d_model=feature_dim+1, out_dim=self.num_head, patchsize=patch)
        self.V_mapping = single_scale(d_model=feature_dim+1, out_dim=int(feature_dim/self.num_head), patchsize=1)
        self.out_mapping = nn.Linear(feature_dim, feature_dim)

        # define initial index
        self.init_index = torch.arange(self.out_len).to(self.device).unsqueeze(0).unsqueeze(0)    # (1,1,out_len)
        self.init_index = self.init_index.repeat(configs.num_nodes, int(feature_dim/self.num_head), 1)    # (N,F,out_len)

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        values: (B,N,T,d_model/H) 
        corr: (B,N,T,H)
        patchsize: scalar

        return: (B,N,1,T)
        """
        batch, nodes, T, feature_per_head  = values.shape  

        # spilt the values
        values = values.unsqueeze(-1).repeat(1, 1, 1, 1, self.num_head)    # (B,N,T,F/H,H)
         
        # return (B,N,F/H,H,T)
        values = values.permute(0,1,3,4,2)
        # return (B,N,H,T)
        corr = corr.permute(0,1,3,2)

        # calculate average corr along the feture dimension
        # return (B,N,H,top_k)
        # chunke the previous part
        k_ = np.minimum(self.topk, corr.shape[-1])
        weights, delay = torch.topk(corr, k_, dim=-1)
        
        tmp_corr = torch.softmax(weights, dim=-1)
        delay = self.patch * delay # + self.query_length

        # aggregation
        tmp_values = torch.cat((values, values), -1)    # return (B,N,F/H,H,T+T)
        indexs = self.init_index.unsqueeze(0).repeat(batch, 1, 1, 1)    # (B,N,F/H,out_len)
        
        output = []
        for j in range(self.num_head):
            
            # for each node, define zeros
            delays_agg = torch.zeros_like(indexs)    # return (B,N,F/H,out_len)

            # delay per head
            delay_per_head = delay[:,:,j,:]    # (B,N,top_k)

            # tmp_value per head
            tmp_per_head = tmp_values[...,j,:]    # (B,N,F/H,T+T)

            # corr score per head
            tmp_corr_per_head = tmp_corr[:,:,j,:]    # (B,N,top_k)

            # time delay agg
            for i in range(k_):

                delay_index = indexs + delay_per_head[...,i].unsqueeze(-1).unsqueeze(-1)   # (B,N,F/H,out_len)
                pattern = torch.gather(tmp_per_head, dim=-1, index=delay_index)   # return (B,N,F/H,out_len)

                # obtain temporal score
                tmp_corr_each = (tmp_corr_per_head[..., i]).unsqueeze(-1).unsqueeze(-1)    # (B,N,1,1)
                delays_agg = delays_agg + pattern * tmp_corr_each   # return (B,N,F/H,out_len)

            # store the output
            output.append(delays_agg)
            
        output = torch.cat(tuple(output), -2)    # return (B,N,F,out_len)
        # output = torch.mean(output, -1)    # return (B,N,F,out_len)
        output = self.out_mapping(output.permute(0,1,3,2)).permute(0,1,3,2)

        return output, delay, tmp_corr

    def cal_QKV(self, Q_in, K_in, V_in, t_stamp):
        '''
        this function is used to 
        1. calculate queries, keys and values
        2. calculate temporal correlation attention using keys and queries

        all input: (B,N,T,F)
        t_stamp: (B,T_long)
        '''
        B, N, T, f = V_in.shape
        time_stamp_pos = t_stamp.unsqueeze(1).unsqueeze(-1)    # (B,1,T_long,1)
        time_stamp_pos = time_stamp_pos.repeat(1,N,1,1)    # (B,N,T_long,1)
        time_stamp_pos = time_stamp_pos[:,:,:T,:]    # (B,N,T,1)
        Q_in = torch.cat((Q_in, time_stamp_pos), -1)
        K_in = torch.cat((K_in, time_stamp_pos), -1)
        V_in = torch.cat((V_in, time_stamp_pos), -1)

        # 
        queries = self.Q_mapping(Q_in)    # (B,N,T,H)
        keys = self.K_mapping(K_in)    # (B,N,T,H)
        values = self.V_mapping(V_in)    # (B,N,T,d_model/H)

        q = queries    # (B,N,T,H)
        k = keys    # (B,N,T,H)
        q = q.permute(0,1,3,2)    # (B,N,H,T)
        k = k.permute(0,1,3,2)    # (B,N,H,T)


        # apply fft
        q_fft = torch.fft.rfft(q.contiguous(), dim=-1)    # (B,N,H,T)
        k_fft = torch.fft.rfft(k.contiguous(), dim=-1)    # (B,N,H,T)
        res = q_fft * torch.conj(k_fft)    # (B,N,H,T)
        corr = torch.fft.irfft(res, dim=-1)    # (B,N,H,T)
        corr = corr.permute(0,1,3,2)    # (B,N,T,H)
        
        return values, corr

    def forward(self, Q_in, K_in, V_in, t):

        '''
        input
        Q_in: (B,N,T,F)
        K_in: (B,N,T,F)
        V_in: (B,N,T,F)

        return (B,N,out_len,F)
        '''
        # obtain the shape
        B, N, T, F = V_in.shape

        # apply QKV mapping, return (B,N,T,d_model/H) (B,N,T,H)
        V_new, corr = self.cal_QKV(Q_in=Q_in, K_in=K_in, V_in=V_in, t_stamp=t)

        # apply spatial-temporal delay aggregation, (B,N,F,out_len), (B,N,H,top_k), (B,N,H,top_k)
        output, delay, delay_score = self.time_delay_agg_full(V_new, corr)  

        output = output.permute(0,1,3,2)    # (B,N,out_len,F)

        # # apply spatial self-attention
        # output = self.layernorm(output)
        # output = self.fcnn(output)    # return (B,N,out_len,F)

        return output.contiguous(), delay, delay_score

class patch_atten(nn.Module):

    def __init__(self, DEVICE, adj, configs):
        super(patch_atten, self).__init__()

        self.patch1 = 3
        self.patch2 = 2
        self.patch3 = 2
        feature_dim = configs.d_model
        self.pred_len = configs.pred_len
        
        self.CAM1_mapping = nn.Sequential(nn.Linear(feature_dim * self.patch1, 32), nn.GELU(), nn.Linear(32,self.patch1))
        self.CAM2_mapping = nn.Sequential(nn.Linear(feature_dim * self.patch2, 32), nn.GELU(), nn.Linear(32,self.patch2))

        # Autocorrelation
        self.cell1 = AutoCorrelation(device=DEVICE, out_len=self.pred_len,  patch=1, query_len=int(configs.time_steps/2), configs=configs)
        self.cell2 = AutoCorrelation(device=DEVICE, out_len=int(self.pred_len/3),  patch=1, query_len=int(configs.time_steps/6), configs=configs)
        self.cell3 = AutoCorrelation(device=DEVICE, out_len=int(self.pred_len/6), patch=1, query_len=int(configs.time_steps/12), configs=configs)

        
    def patch_agg(self, x, mapping_function, patch_size):

        '''
        x: (B,N,T,F)
        '''
        # obtain the shape
        B,N,T,feature = x.shape

        # reshape return (B,N,T/p,p,F)
        x = x.view(B,N,int(T/patch_size),patch_size,feature)

        # apply CAM
        x_out = x.reshape(B,N,int(T/patch_size), -1)
        score = mapping_function(x_out)    # (B,N,T/p,p)
        score = F.softmax(score, -1)    # (B,N,T/p,p)

        # patch agg
        score = score.unsqueeze(-1)    # (B,N,T/p,p,1)
        x = score * x    # (B,N,T/p,p,F)
        x = torch.sum(x, -2)    # (B,N,T/p,F)

        return x

    def patch_forward(self, x):
        '''
        input x: (B,N,T,F)
        '''

        B,N,T,feature = x.shape

        # apply patch attention for the first time
        x0 = x
        x1 = self.patch_agg(x0, self.CAM1_mapping, self.patch1)
        x2 = self.patch_agg(x1, self.CAM2_mapping, self.patch2)
        
        return [x0, x1, x2]

    def forward(self, x, t):
        '''
        input x: (T)
        '''

        # return (T), (T/2), (T/6)
        [sp0_x0, sp0_x1, sp0_x2] = self.patch_forward(x)

        sp0_query = sp0_x0[:,:,-int(0.5*sp0_x0.shape[2]):,:]
        sp0_query = torch.cat((sp0_query, torch.zeros_like(sp0_query).to(sp0_query.device)), 2)
        sp0_x0_out, delay1, delay_score1 = self.cell1.forward(Q_in=sp0_query, K_in=sp0_x0, V_in=sp0_x0, t=t)
        sp0_x0 = x + sp0_x0_out   # (B,N,pred_len,F)
        
        sp1_query = sp0_x1[:,:,-int(0.5*sp0_x1.shape[2]):,:]
        sp1_query = torch.cat((sp1_query, torch.zeros_like(sp1_query).to(sp1_query.device)), 2)
        sp0_x1_out, delay2, delay_score2  = self.cell2.forward(Q_in=sp1_query, K_in=sp0_x1, V_in=sp0_x1, t=t)
        sp0_x1 = x + torch.repeat_interleave(sp0_x1_out, 3, dim=2)
        
        sp2_query = sp0_x2[:,:,-int(0.5*sp0_x2.shape[2]):,:]
        sp2_query = torch.cat((sp2_query, torch.zeros_like(sp2_query).to(sp2_query.device)), 2)
        sp0_x2_out, delay3, delay_score3  = self.cell3.forward(Q_in=sp2_query, K_in=sp0_x2, V_in=sp0_x2, t=t)
        sp0_x2 = x + torch.repeat_interleave(sp0_x2_out, 6, dim=2)
        

        return [sp0_x0, sp0_x1, sp0_x2], [delay1, delay2, delay3], [delay_score1, delay_score2, delay_score3]
        




class Autoformer(nn.Module):
    
    def __init__(self, DEVICE, adj, configs):
        super(Autoformer, self).__init__()

        # used in the forward function
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.input_lenght = configs.time_steps

        # used in define the layers
        self.out_len_list = configs.out_len

        # define a zeros tensor
        self.zeros = torch.zeros(configs.num_nodes, self.pred_len, configs.d_model).to(DEVICE)

        # spatial encoder  
        self.GCN_agg1 = GCN(adj=adj, out_len=configs.time_steps, configs=configs)
        self.GCN_agg2 = GCN(adj=adj, out_len=configs.time_steps, configs=configs)
        self.GCN_agg3 = GCN(adj=adj, out_len=configs.time_steps, configs=configs)
        self.GCN_agg4 = GCN(adj=adj, out_len=configs.time_steps, configs=configs)

        # patch attention
        self.patch_att1 = patch_atten(DEVICE=DEVICE, adj=adj, configs=configs)
        self.patch_att2 = patch_atten(DEVICE=DEVICE, adj=adj, configs=configs)
        self.patch_att3 = patch_atten(DEVICE=DEVICE, adj=adj, configs=configs)
        self.patch_att4 = patch_atten(DEVICE=DEVICE, adj=adj, configs=configs)
        self.patch_att5 = patch_atten(DEVICE=DEVICE, adj=adj, configs=configs)

        self.relu_fcnn = nn.Sequential(nn.Linear(configs.d_model * 13, 512), nn.Tanh(), nn.Linear(512,512), nn.Tanh(), nn.Linear(512, configs.d_model))


    def forward(self, x_enc, t):

        '''
        x_enc: (B,N,T,F)
        '''
        B,N,T,F = x_enc.shape

        # define initialization
        x_init = (x_enc[:,:,-1,:].unsqueeze(-2)).repeat(1,1,12,1)

        # apply spatial encoding
        # x_enc0 = x_enc
        # x_enc1, sp_attn1 = self.GCN_agg1.forward(x_enc0)
        # x_enc1 = x_enc1 + x_init
        # x_enc2, sp_attn2 = self.GCN_agg2.forward(x_enc1)
        # x_enc2 = x_enc2 + x_init
        # x_enc3, sp_attn3 = self.GCN_agg3.forward(x_enc2)
        # x_enc3 = x_enc3 + x_init

        # version 2
        x_enc0 = x_enc
        x_enc1, sp_attn1 = self.GCN_agg1.forward(x_enc0)
        x_enc1 = x_enc1 + x_enc0
        x_enc2, sp_attn2 = self.GCN_agg2.forward(x_enc1)
        x_enc2 = x_enc2 + x_enc1
        x_enc3, sp_attn3 = self.GCN_agg3.forward(x_enc2)
        x_enc3 = x_enc3 + x_enc2
        # x_enc1 = x_enc1 + x_init
        # x_enc2 = x_enc2 + x_init
        # x_enc3 = x_enc3 + x_init


        # apply patch attention, return (B,N,T,F)
        [sp0_x0, sp0_x1, sp0_x2], \
        [delay00, delay01, delay02],\
        [delay_score00, delay_score01, delay_score02] = self.patch_att1.forward(x=x_enc0, t=t)
        
        [sp1_x0, sp1_x1, sp1_x2], \
        [delay10, delay11, delay12],\
        [delay_score10, delay_score11, delay_score12] = self.patch_att2.forward(x=x_enc1, t=t)
        
        [sp2_x0, sp2_x1, sp2_x2], \
        [delay20, delay21, delay22],\
        [delay_score20, delay_score21, delay_score22] = self.patch_att3.forward(x=x_enc2, t=t)
        
        [sp3_x0, sp3_x1, sp3_x2], \
        [delay30, delay31, delay32],\
        [delay_score30, delay_score31, delay_score32] = self.patch_att4.forward(x=x_enc3, t=t)


        # combine all the inputs, return (B,N,pred_len,F,16)
        output = torch.cat((sp0_x0.unsqueeze(-1), sp0_x1.unsqueeze(-1), sp0_x2.unsqueeze(-1), \
            sp1_x0.unsqueeze(-1), sp1_x1.unsqueeze(-1), sp1_x2.unsqueeze(-1),\
            sp2_x0.unsqueeze(-1), sp2_x1.unsqueeze(-1), sp2_x2.unsqueeze(-1),\
            sp3_x0.unsqueeze(-1), sp3_x1.unsqueeze(-1), sp3_x2.unsqueeze(-1),
            x_init.unsqueeze(-1)), -1)

        # return (B,N,pred_len,F)
        # activate_score = torch.exp(self.relu_fcnn(output.view(B,N,self.pred_len,-1)))
        # output = activate_score.unsqueeze(-2) * output
        # output = torch.mean(output, -1)
        output = self.relu_fcnn(output.view(B,N,self.pred_len,-1))

        # record attention score
        sp_attns = [sp_attn1, sp_attn2, sp_attn3]
        te_attns_delays = [delay00, delay01, delay02, delay10, delay11, delay12,\
                           delay20, delay21, delay22, delay30, delay31, delay32]
        te_attns_score = [delay_score00, delay_score01, delay_score02, \
                          delay_score10, delay_score11, delay_score12, \
                          delay_score20, delay_score21, delay_score22, \
                          delay_score30, delay_score31, delay_score32]

        return output, [sp_attns, te_attns_delays, te_attns_score]


class Model(nn.Module):
    
    def __init__(self, adj, configs, DEVICE):
        super(Model, self).__init__()

        # define model parameters
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.out_len = configs.out_len
        self.patch_list = configs.patch_list
        self.topk = configs.topk
        self.encoder_length = configs.time_steps
        self.n_head = configs.n_heads
        self.time_steps = configs.time_steps
        self.num_nodes = configs.num_nodes


        # define t_embedding neural network
        self.encoder = nn.Sequential(nn.Linear(3, 256), nn.Tanh(), nn.Linear(256,256), nn.Tanh(), nn.Linear(256,256), nn.Tanh(), nn.Linear(256, configs.d_model))

        # define the Autoformer
        self.A1 = Autoformer(DEVICE=DEVICE, adj=adj, configs=configs)

        # out projection
        self.decoder = nn.Sequential(nn.Linear(configs.d_model, 512), nn.Tanh(), nn.Linear(512,256), nn.Tanh(), nn.Linear(256,256), nn.Tanh(), nn.Linear(256, 1))
        
    def forward(self, x_enc, t):
        '''
        input
        x_enc: (B, T=7 weeks, N)
        t: (B,T)

        return: (B, T = num_pred_len * pred_len, N)
        '''

        # always use segment of length of pred_len to initialize the future
        # return (B,pred_len,F)
        # x_enc = x_enc.unsqueeze(-1)    # (B,T,N,1)

        # apply encoder: (B,T,N,F)
        x_enc = self.encoder(x_enc)
        # permute the shape, return (B,N,T,F)
        x_enc = x_enc.permute(0,2,1,3)

        # apply the Graph Autoformer
        out, explains = self.A1(x_enc, t)

        # apply out projection
        out = self.decoder(out).squeeze(-1)    # (B,N,pred_len)

        # output, return (B,pred_len,N)
        out = out.permute(0,2,1)

        return out, explains


