__all__ = ['GNN_autoformer']

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Autoformer import Model

class self_att(nn.Module):

    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(self_att, self).__init__()
    
        self.W1 = nn.Parameter(torch.rand(num_of_timesteps).float().to(DEVICE))
        self.W2 = nn.Parameter(torch.rand(in_channels, num_of_timesteps).float().to(DEVICE))
        self.W3 = nn.Parameter(torch.rand(in_channels).float().to(DEVICE))
        self.bs = nn.Parameter(torch.rand(1, num_of_vertices, num_of_vertices).float().to(DEVICE))
        self.Vs = nn.Parameter(torch.rand(num_of_vertices, num_of_vertices).float().to(DEVICE))
        self.time_delay = nn.Parameter(torch.rand(num_of_vertices, num_of_vertices).float().to(DEVICE))
    
    def softmax_for_nonzeros(self, att_score, adj):

        '''
        input: att_score (B,N,N)
               adj (N,N) binary matrix
        '''
        B,N,_ = att_score.shape

        exp_att = torch.exp(att_score)

        exp_att = exp_att * adj.unsqueeze(0).repeat(B,1,1)

        assert not torch.isnan(exp_att).any()


        normalization_factor = torch.sum(exp_att, -1).unsqueeze(-1).repeat(1,1,N) # return (B,N,N)
        exp_att = exp_att / normalization_factor

        return exp_att

    def forward(self, x, adj):
        '''
        :param x: (batch_size, N, F_in == 1, T)
        :return: (B,N,N)
        '''
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)
        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)
        S = torch.sigmoid(product + self.bs)  # (N,N)(B, N, N)->(B,N,N)
        S_normalized = self.softmax_for_nonzeros(S, adj)

        return S_normalized    # return (B, N, N)
    

class GNN_autoformer(nn.Module):

    def __init__(self, adj, configs, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(GNN_autoformer, self).__init__()

        self.args = configs
        self.ASTGNN = self_att(DEVICE, in_channels, num_of_vertices, self.args.seq_len)
        self.Auto_former1 = Model(adj, configs)
        self.Auto_former2 = Model(adj, configs)
        self.Auto_former3 = Model(adj, configs)
        self.decoder = nn.Sequential(nn.Linear(num_of_vertices, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, num_of_vertices))
        self.device = DEVICE
        self.coeff = nn.Parameter(torch.rand(3).float().to(DEVICE))

    def forward_one_trend(self, batch_x, batch_y, id):

        '''
        input: batch_x (B,T,N)
        '''

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # forward (set mark to zeros)
        outputs = self.Auto_former1(batch_x, 0, dec_inp, 0)    # return (B,T,N)

        return outputs
    
    def forward_second_trend(self, batch_x, batch_y, id):

        '''
        input: batch_x (B,T,N)
        '''

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # forward (set mark to zeros)
        outputs = self.Auto_former2(batch_x, 0, dec_inp, 0)    # return (B,T,N)

        return outputs
    
    def forward_third_trend(self, batch_x, batch_y, id):

        '''
        input: batch_x (B,T,N)
        '''

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # forward (set mark to zeros)
        outputs = self.Auto_former3(batch_x, 0, dec_inp, 0)    # return (B,T,N)

        return outputs

    

    def forward(self, x, y, adj_list):
        '''
        input: x (B,T,N)
        '''

        # perform graph convolution
        x_expand = x.permute(0,2,1).unsqueeze(2)
        S_one_hop = self.ASTGNN(x_expand, adj_list[0])    # return (B, N, N)
        S_two_hop = self.ASTGNN(x_expand, adj_list[1])    # return (B, N, N)

        # perform graph convolution
        x_one_hop_att = torch.einsum('bij,bjk->bik', S_one_hop, x.permute(0,2,1)).permute(0,2,1)    # return (B, T, N) 
        x_two_hop_att = torch.einsum('bij,bjk->bik', S_two_hop, x.permute(0,2,1)).permute(0,2,1)    # return (B, T, N) 
        y_one_hop_att = torch.einsum('bij,bjk->bik', S_one_hop, y.permute(0,2,1)).permute(0,2,1)    # return (B, T, N) 
        y_two_hop_att = torch.einsum('bij,bjk->bik', S_two_hop, y.permute(0,2,1)).permute(0,2,1)    # return (B, T, N) 
    
        # perform forcasting
        out1 = self.forward_one_trend(x, y, 0)    # return (B,T,N)
        out2 = self.forward_second_trend(x_one_hop_att, y_one_hop_att, 1)    # return (B,T,N)
        out3 = self.forward_third_trend(x_two_hop_att, y_two_hop_att, 2)    # return (B,T,N)
        # out = torch.cat((out1, out2, out3), -1)
        out1 = self.decoder(out1)    # return (B,T,N)
        out2 = self.decoder(out2)    # return (B,T,N)
        out3 = self.decoder(out3)    # return (B,T,N)
        out = self.coeff[0] * out1 + self.coeff[1] * out2 + self.coeff[2] * out3

        return out
        



