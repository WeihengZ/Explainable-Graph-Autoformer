import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer, AutoCorrelation_spa_tem
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, adj, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len    # length of the input sequence
        self.label_len = configs.label_len    # length of the input sequence into the encoder
        self.pred_len = configs.pred_len    # length of the predicted sequence 
        self.output_attention = configs.output_attention    # True / False determine whether output the performance

        # Decomp
        kernel_size = configs.moving_avg    # a scalar: smoothing window size
        self.decomp = series_decomp(kernel_size)    # return list of [ (B, T, F), (B, T, F) ]


        ''' define your own embedding'''
        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        '''
        configs.enc_in: encoder input dimension of each time step
        configs.d_model: hidden dimension of the model of each time step
        configs.embed: str, represent the type of embedding
        configs.freq: ?
        configs.dropout: dropout probability
        
        ''' 

        # apply the fully connected neural network to encoder, couple it it other information
        self.enc_embedding = nn.Linear(configs.enc_in, configs.d_model)
        self.dec_embedding = nn.Linear(configs.enc_in, configs.d_model)
        # DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,configs.dropout)

        # Encoder
        # define multiple encoder layers, default is two
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation_spa_tem(adj, False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation_spa_tem(adj, True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation_spa_tem(adj, False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        '''
        by now, I comment part of the code of "forward" of "DataEmbedding_wo_pos" since I am not fully understand the time embedding
        '''
        
        '''
        input
        x_enc: (B, known length, N)
        x_dec: (B, alpha * known length, c_in), where 0 < alpha < 1
        x_mark_enc: (B, c_in, known length)    (static information)
        x_mark_dec: (B, c_in, alpha * known length), where 0 < alpha < 1    (static information)
        '''

        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)

        # decompose the encoder sequence
        seasonal_init, trend_init = self.decomp(x_enc)

        # derive the decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # apply the encoder
        # return list of (B,T,N), 
        enc_out, attns = self.encoder(x_enc, attn_mask=enc_self_mask)

        # apply decoder on the seasonal part
        seasonal_part, trend_part = self.decoder(seasonal_init, enc_out, trend=trend_init)

        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, T, D]
