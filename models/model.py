import torch
import torch.nn as nn
import torch.nn.functional as F

class time_embedding(nn.Module):

    def __init__(self, out_dim):
        super(time_embedding, self).__init__()

        self.fcnn = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32,32), nn.GELU(), nn.Linear(32, out_dim))

    def forward(self, x, t):

        '''
        input
        x: (B,T,N)
        t: (B,T)

        return 
        (B,T,N,F)
        '''

        # combine the information of x and t
        B,T,N = x.shape
        t = t.unsqueeze(-1).repeat(1,1,N)
        x = torch.cat((x.unsqueeze(-1), t.unsqueeze(-1)), -1)    # (B,T,N,2)

        # apply neural network
        x = self.fcnn(x)   # (B,T,N,F)

        return x

class multi_scale(nn.Module):

    def __init__(self):
        super(multi_scale, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,3), stride=(1,3)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,4), stride=(1,4)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,6), stride=(1,6)),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,4), stride=(1,4)),
            nn.ReLU()
        )

        # define "feature return" convolution layer 
        self.decode2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1,1), stride=(1,1))
        self.decode3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1,1), stride=(1,1))
        self.decode4 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1,1), stride=(1,1))
        self.decode5 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(1,1), stride=(1,1))
            

    def forward(self, x):

        '''
        input
        x: (B,N,T = 1 week,1)

        return 
        (B,N,T,1)
        '''

        # apply convolution
        x = x.permute(0,3,1,2)    # (B, 1, N, T)
        x_quarter = self.conv1(x)    # (B, 16, N, T/3 week)
        x_hour = self.conv2(x_quarter)    # (B, 32, N, 7*24 hours)
        x_6hour = self.conv3(x_hour)    # (B, 64, N, 7*4 segment)
        x_week = self.conv4(x_6hour)    # (B, 128, N, 7*4 segment)

        # return the feature
        x_quarter = self.decode2(x_quarter)    # (B, 1, N, T/3)
        x_hour = self.decode3(x_hour)    # (B, 1, N, T/12)
        x_6hour = self.decode4(x_6hour)    # (B, 1, N, T/72)
        x_week = self.decode5(x_week)    # (B, 1, N, T/288)

        # output, return list of (B,N,t',1)
        out =[x_quarter.permute(0,2,3,1), x_hour.permute(0,2,3,1), x_6hour.permute(0,2,3,1), x_week.permute(0,2,3,1)]

        return out

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

    def __init__(self, in_dim1, in_dim2, hidden, DEVICE):
        super(full_attns, self).__init__()

        self.Q_mapping1 = nn.Linear(in_dim2, 1, bias=False)
        self.Q_mapping2 = nn.Linear(in_dim1, hidden, bias=False)
        self.K_mapping1 = nn.Linear(in_dim2, 1, bias=False)
        self.K_mapping2 = nn.Linear(in_dim1, hidden, bias=False)
        self.W = nn.Parameter(torch.rand(hidden, hidden)).float().to(DEVICE)

    def forward(self, x):

        '''
        x: (B, attn_dim, in_dim1, in_dim2)
        '''
        B, attn_dim, feature_dim1, feature_dim2 = x.shape

        Q = self.Q_mapping2(torch.relu(self.Q_mapping1(x).squeeze(-1)))
        K = self.K_mapping2(torch.relu(self.K_mapping1(x).squeeze(-1)))
        A = torch.matmul(Q, self.W)    
        A = torch.einsum('bij,bjk->bik', A, K.permute(0,2,1))    # return (B, attn_dim, attn_dim)
        A = F.softmax(A, -1)

        return A

class GCN(nn.Module):

    def __init__(self, adj, time_step, feature_dim, DEVICE):
        super(GCN, self).__init__()

        self.attn_cal = full_attns(in_dim1=feature_dim, in_dim2=time_step, hidden=128, DEVICE=DEVICE)
        self.W = nn.Parameter(torch.rand(feature_dim, 128).float().to(DEVICE))
        self.adj = adj
        self.Wout = nn.Parameter(torch.rand(128, feature_dim).float().to(DEVICE))
    
    def forward(self, x):

        '''
        input x: (B,N,T,F)
        adj: (N,N)

        return: (B,N,T,F)
        '''

        # adj2 is the real spatial attention score !
        attn = self.attn_cal.forward(x.permute(0,1,3,2))    # (B,N,N)
        adj2 = attn * self.adj.unsqueeze(0)     # (B,N,N)


        x_new = torch.matmul(x, self.W)    # return (B,N,T,64)
        x_new = nn.GELU()(torch.einsum('bij,bjkl->bikl', adj2, x_new))    # return (B,N,T,64)
        x_new = torch.matmul(x_new, self.Wout)    # return (B,N,T,1)
     
        return x_new, adj2

class AutoCorrelation(nn.Module):
    """
    
    """
    def __init__(self, device, adj, out_len, feature_dim, patch, input_len, topk, n_head):
        super(AutoCorrelation, self).__init__()

        # define the output len
        self.out_len = out_len

        # define the device
        self.device = device

        # defien patch size
        self.patch = patch
    
        # define top_k
        self.topk = topk

        # define number of head
        self.num_head = n_head

        # define module for "cal_QKV"
        self.Q_mapping = single_scale(d_model=feature_dim, out_dim=self.num_head, patchsize=patch)
        self.K_mapping = single_scale(d_model=feature_dim, out_dim=self.num_head, patchsize=patch)
        self.V_mapping = single_scale(d_model=feature_dim, out_dim=int(feature_dim/self.num_head), patchsize=1)

        # define spatial attention module for each chosen segemnt
        self.GCN_agg = GCN(adj=adj, time_step=out_len, feature_dim=feature_dim, DEVICE=device)

        # define a fully connected neural network
        self.fcnn = nn.Sequential(nn.Linear(feature_dim, 512), nn.ReLU(), nn.Linear(512, feature_dim))

        # define initial index
        self.init_index = torch.arange(self.out_len).to(self.device)

        # define the output mapping
        self.out_mapping = nn.Linear(int(feature_dim/self.num_head), feature_dim, bias=False)

    def time_delay_agg_train(self, values, corr, patchsize):
        """
        Standard version of Autocorrelation
        values: (B,N,T,1)
        corr: (B,T/patchsize,F)
        patchsize: scalar

        return: (B,N,F,pred_len)
        """
        batch, nodes, T, feature  = values.shape    # feature == 1

        # return (B,N,1,T)
        values = values.permute(0,1,3,2)
        corr = corr.permute(0,1,3,2)

        # return (top_k)
        '''
        here is the case of batch size = 1, feature size = 1
        we assume node share the same temporal attention
        '''
        corr = torch.mean(torch.mean(torch.mean(corr, 0), 0), 0)
        weights, delay = torch.topk(corr, self.topk, dim=-1)

        # update corr, return (top_k)
        tmp_corr = torch.softmax(weights, dim=-1)

        # aggregation
        tmp_values = values    # return (B,N,1,T)
        delays_agg = torch.zeros_like(values)    # return (B,N,1,T)

        # time delay agg
        for i in range(self.topk):
            pattern = torch.roll(tmp_values, patchsize * int(delay[i]), dims=-1)    # return (B,N,1,T)
            delays_agg = delays_agg + pattern * (tmp_corr[i])   # return (B,N,1,T)

        # apply spatial self-attention
        new_values = self.GCN_agg(values.permute(0,1,3,2))    # return (B,N,T,1)

        return new_values, delay, tmp_corr

    def time_delay_agg_full2(self, values, corr):
        """
        Standard version of Autocorrelation
        values: (B,N,T,F)
        corr: (B,N,T/patchsize,H)
        patchsize: scalar

        return: (B,N,1,T)
        """
        batch, nodes, T, feature  = values.shape  

        # defien feature per head
        feature_per_head = int(feature/self.num_head)

        # spilt the values
        values = values.view(batch, nodes, T, feature_per_head, self.num_head)    # (B,N,T,F/H,H)
         
        # return (B,N,F/H,H,T)
        values = values.permute(0,1,3,4,2)
        # return (B,N,H,T/p)
        corr = corr.permute(0,1,3,2)

        # calculate average corr along the feture dimension
        # return (B,N,H,top_k)
        weights, delay = torch.topk(corr, self.topk, dim=-1)
        delay = self.patch * delay

        # calculate attention score
        # return (B,N,H,top_k)
        tmp_corr = torch.softmax(weights, dim=-1)

        # aggregation
        tmp_values = torch.cat((values, values), -1)    # return (B,N,F/H,H,T+T)
        
        output = []
        for j in range(self.num_head):
            
            # for each node, define zeros
            delays_agg = torch.zeros_like(self.init_index)    
            delays_agg = delays_agg.unsqueeze(0).unsqueeze(0).unsqueeze(0)    # return (1,1,1,out_len)
            delays_agg = delays_agg.repeat(batch, nodes, feature_per_head,1)    # return (B,N,F/H,out_len)

            # delay per head
            delay_per_head = delay[:,:,j,:]    # (B,N,top_k)

            # tmp_value per head
            tmp_per_head = tmp_values[...,j,:]    # (B,N,F/H,T+T)

            # corr score per head
            tmp_corr_per_head = tmp_corr[:,:,j,:]    # (B,N,top_k)

            # time delay agg
            for i in range(self.topk):

                delay_index = self.init_index.unsqueeze(0).unsqueeze(0).unsqueeze(0) + delay_per_head[...,i].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, feature_per_head, self.out_len)    # (B,N,F/H,out_len)
                pattern = torch.gather(tmp_per_head, dim=-1, index=delay_index)   # return (B,N,F/H,out_len)

                # obtain temporal score
                tmp_corr_each = (tmp_corr_per_head[..., i]).unsqueeze(-1).unsqueeze(-1)    # (B,N,1,1)
                delays_agg = delays_agg + pattern * tmp_corr_each   # return (B,N,F/H,out_len)

            # store the output
            output.append(delays_agg.unsqueeze(-1))
            
        output = torch.cat(tuple(output), -1)    # return (B,N,F/H,out_len, H)
        output = torch.mean(output, -1)    # return (B,N,F/H,out_len)
        output = self.out_mapping(output.permute(0,1,3,2)).permute(0,1,3,2)    # return (B,N,F,out_len)

        return output, delay, tmp_corr


    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        values: (B,N,T,F)
        corr: (B,T/patchsize,F)
        patchsize: scalar

        return: (B,N,1,T)
        """
        batch, nodes, T, feature  = values.shape    

        # return (B,N,F,T)
        values = values.permute(0,1,3,2)
        # return (B,F,T/p)
        corr = corr.permute(0,2,1)

        # index init, return (B,N,F,out_len)
        init_index = torch.arange(self.out_len).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, nodes, feature, 1).to(self.device)

        # calculate average corr along the feture dimension
        # return (B,F,top_k)
        weights, delay = torch.topk(corr, self.topk, dim=-1)

        # update corr, return (B,F,top_k)
        tmp_corr = torch.softmax(weights, dim=-1)

        # aggregation
        tmp_values = torch.cat((values, values), -1)    # return (B,N,F,T+T)
        delays_agg = torch.zeros_like(init_index)    # return (B,N,F,out_len)

        # time delay agg
        for i in range(self.topk):
            delay_each = (delay[..., i]).unsqueeze(1).unsqueeze(-1).repeat(1,nodes,1,1)    # (B,N,F,1)
            tmp_delay = init_index + self.patch * delay_each   # return (B,N,F,out_len)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)    # return (B,N,F,out_len)
            tmp_corr_each = (tmp_corr[..., i]).unsqueeze(1).unsqueeze(-1).repeat(1,nodes,1,1)    # (B,N,F,1)
            delays_agg = delays_agg + pattern * tmp_corr_each   # return (B,N,F,out_len)

        # apply spatial self-attention
        new_values = self.GCN_agg(delays_agg.permute(0,1,3,2))    # return (B,N,out_len,F)

        return new_values, delay, tmp_corr

    def cal_QKV(self, Q_in, K_in, V_in):
        '''
        this function is used to 
        1. calculate queries, keys and values
        2. calculate temporal correlation attention using keys and queries

        all input: (B,N,T,F)
        '''

        queries = self.Q_mapping(Q_in)    # (B,N,t',H)
        keys = self.K_mapping(K_in)    # (B,N,t',H)
        values = self.V_mapping(V_in)    # (B,N,T,d_model/H)

        # repeat the values, so same value is used in each head
        values = values.repeat(1,1,1,self.num_head)    # (B,N,T,d_model)

        q = queries    # (B,N,t',H)
        k = keys    # (B,N,t',H)
        q = q.permute(0,1,3,2)    # (B,N,H,t')
        k = k.permute(0,1,3,2)    # (B,N,H,t')

        # '''
        # to speed up the computation, we calculate mean feature along all the nodes
        # '''
        # q = torch.mean(q, 1)    # (B,d_model,t')
        # k = torch.mean(k, 1)    # (B,d_model,t')

        # apply fft
        q_fft = torch.fft.rfft(q.contiguous(), dim=-1)    # (B,N,H,t')
        k_fft = torch.fft.rfft(k.contiguous(), dim=-1)    # (B,N,H,t')
        res = q_fft * torch.conj(k_fft)    # (B,N,H,t')
        corr = torch.fft.irfft(res, dim=-1)    # (B,N,H,t')
        corr = corr.permute(0,1,3,2)    # (B,N,t',H)
        
        return values, corr

    def forward(self, Q_in, K_in, V_in):

        '''
        input
        Q_in: (B,N,T,F)
        K_in: (B,N,T,F)
        V_in: (B,N,T,F)

        return (B,N,out_len,F)
        '''
        # obtain the shape
        B, N, T, F = V_in.shape

        # # apply temporal position embedding
        # Q_in = self.temporal_embed.forward(Q_in)
        # K_in = self.temporal_embed.forward(K_in)
        # V_in = self.temporal_embed.forward(V_in)

        # apply QKV mapping, return (B,N,T,F), (B,N,t',H)
        V_new, corr = self.cal_QKV(Q_in=Q_in, K_in=K_in, V_in=V_in)

        # apply spatial-temporal delay aggregation, (B,N,F,out_len), (B,N,H,top_k), (B,N,H,top_k)
        output, delay, delay_score = self.time_delay_agg_full2(V_new, corr)  

        # # apply spatial position embedding, return (B,N,out_len,F)
        # output = self.spatial_embed.forward(output.permute(0,1,3,2))
        output = output.permute(0,1,3,2)

        # apply spatial self-attention
        new_values, sp_attn = self.GCN_agg(output)    # return (B,N,out_len,F), (B,N,N)

        return new_values.contiguous(), (delay, delay_score, sp_attn)

class temporal_attn_agg(nn.Module):
    def __init__(self, feature_dim):
        super(temporal_attn_agg, self).__init__()

        self.fcnn1 = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU())
        self.fcnn2 = nn.Sequential(nn.ReLU(), nn.Linear(128, feature_dim))

    def forward(self, x, attns):

        '''
        input 
        x: (B,N,T,F)
        attns (B,T,T)

        return (B,N,T,F)
        '''
        x = x.permute(0,2,1,3)    # return (B,T,N,F_in)
        x = self.fcnn1(x)    # return (B,N,T,128)
        x = torch.einsum('bij,bjkl->bikl', attns, x)    # return (B,T,N,128)
        x = x.permute(0,2,1,3)   # return (B,N,T,128)
        x = self.fcnn2(x)    # return (B,N,T,F_in)

        return x

class st_agg(nn.Module):

    def __init__(self, pred_len, feature_dim, node_number, adj, DEVICE):
        super(st_agg, self).__init__()

        # temporal agg
        self.temp_attn = full_attns(in_dim1=feature_dim, in_dim2=node_number, hidden=128, DEVICE=DEVICE)
        self.temp_agg = temporal_attn_agg(feature_dim=feature_dim)
        
        # spatial agg
        self.GCN_agg = GCN(adj=adj, time_step=pred_len, feature_dim=feature_dim, DEVICE=DEVICE)


    def forward(self, x):

        '''
        input
        x: (B,N,pred_len,F)

        return 
        (B,N,pred_len,F)
        '''
        # temporal agg
        x = x.permute(0,2,3,1)    # (B,pred_len,F,N)
        temp_attn = self.temp_attn.forward(x)    # (B, pred_len, pred_len)
        x = x.permute(0,3,1,2)    # (B,N,pred_len,F)
        x = self.temp_agg.forward(x=x, attns=temp_attn)    # (B,N,pred_len,F)

        # spatial agg
        x = self.GCN_agg.forward(x)    # (B,N,pred_len,F)

        return x

class Autoformer(nn.Module):
    def __init__(self, DEVICE, adj, configs):
        super(Autoformer, self).__init__()

        # used in the forward function
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        # used in define the layers
        out_len_list = configs.out_len
        patch_list = configs.patch_list


        # encoder
        self.cell1 = AutoCorrelation(device=DEVICE, adj=adj, out_len=out_len_list[0], feature_dim=configs.d_model, patch=patch_list[0], input_len=configs.time_steps, topk=configs.topk, n_head=configs.n_heads)
        self.cell2 = AutoCorrelation(device=DEVICE, adj=adj, out_len=out_len_list[1], feature_dim=configs.d_model, patch=patch_list[1], input_len=out_len_list[0], topk=configs.topk, n_head=configs.n_heads)

        # decoder
        self.cell3 = AutoCorrelation(device=DEVICE, adj=adj, out_len=out_len_list[2], feature_dim=configs.d_model, patch=patch_list[2], input_len=out_len_list[1], topk=configs.topk, n_head=configs.n_heads)
        self.cell4 = AutoCorrelation(device=DEVICE, adj=adj, out_len=out_len_list[3], feature_dim=configs.d_model, patch=patch_list[3], input_len=out_len_list[2], topk=configs.topk, n_head=configs.n_heads)
        self.cell5 = AutoCorrelation(device=DEVICE, adj=adj, out_len=self.pred_len, feature_dim=configs.d_model, patch=patch_list[4], input_len=out_len_list[3], topk=configs.topk, n_head=configs.n_heads)


    def forward(self, x_enc):

        '''
        x_enc: (B,N,T,F)
        '''

        B,N,T,F = x_enc.shape

        # always use perd_len segment as decoder input
        x_dec = x_enc[:, :, -self.pred_len:, :]
        x_dec = torch.cat((x_dec, torch.zeros(B, N, T-self.pred_len, self.d_model).to(x_dec.device)), 2)

        # encode information
        x_enc, explain1= self.cell1.forward(x_enc, x_enc, x_enc)    # (B,N,out_len_list[0],F)
        x_enc, explain2 = self.cell2.forward(x_enc, x_enc, x_enc)    # (B,N,out_len_list[1],F)

        # transform attention
        x_dec = x_dec[:,:,:x_enc.shape[2],:]
        x_dec_out, explain3 = self.cell3.forward(x_dec, x_enc, x_enc)    # (B,N,out_len_list[1],F)
        x_dec = x_dec_out + x_dec[:,:,:x_dec_out.shape[2],:]    # (B,N,out_len_list[1],F)

        # decode information
        x_dec_out, explain4 = self.cell4.forward(x_dec, x_dec, x_dec)    # (B,N,out_len[],F)
        x_dec = x_dec_out + x_dec[:,:,:x_dec_out.shape[2],:]    # (B,N,out_len[],F)
        x_dec_out, explain5 = self.cell5.forward(x_dec, x_dec, x_dec)    # (B,N,pred_len,F)
        x_dec = x_dec_out + x_dec[:,:,:self.pred_len,:]    # (B,N,pred_len,F)


        explains = [explain1, explain2, explain3, explain4, explain5]

        # # fully-attention decoder, (B,N,pred_len,F)
        # x_dec = self.st_agg.forward(x_dec)

        return x_dec, explains


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

        # define t_embedding neural network
        self.t_embed = time_embedding(out_dim=configs.d_model)

        # define the Autoformer
        self.A1 = Autoformer(DEVICE=DEVICE, adj=adj, configs=configs)

        # out projection
        self.out_proj = nn.Sequential(nn.Linear(configs.d_model * self.pred_len, 1024), nn.ReLU(), nn.Linear(1024, self.pred_len))
        
    def forward(self, x_enc, t):
        '''
        input
        x_enc: (B, T=7 weeks, N)
        t: (B,T)

        return: (B, T = num_pred_len * pred_len, N)
        '''

        B,T,N = x_enc.shape
 
        x_enc = self.t_embed.forward(x_enc, t)
        # permute the shape, return (B,N,T,F)
        x_enc = x_enc.permute(0,2,1,3)

        # apply the Graph Autoformer
        out, explains = self.A1(x_enc)

        # apply out projection
        out = self.out_proj(out.view(B,N,-1))    # (B,N,pred_len)

        # output, return (B,pred_len,N)
        out = out.permute(0,2,1)

        return out, explains


