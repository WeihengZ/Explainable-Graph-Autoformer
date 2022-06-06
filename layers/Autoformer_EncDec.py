import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    apply average pooling to single time series, default stride size is one
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        '''
        input x (B,T,F)
        return x (B,T,F)
        '''
        # padding on the both ends of time series, such that the center of the window size is on the target node
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)   # expand on t dimension
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)    # expand on t dimension
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))    # return (B, F, T) after apply avg pooling
        x = x.permute(0, 2, 1)    # return (B, T, F)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        '''
        input (B, T, F)
        return (B, T, F), (B, T, F)
        '''
        moving_mean = self.moving_avg(x)    # return (B, T, F)
        res = x - moving_mean    # return (B,T,F) which is the seasonal part of the time series
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):


        '''
        x: (B,T,N)

        '''

        # (B, T, N) -> new_x (shape = (B, T, N))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        # shortcut return (B, T, hidden_dim)
        x = x + self.dropout(new_x)
        # return seasonal information (B, T, hidden_dim)
        x, _ = self.decomp1(x)

        # obtain seasonal part of first decomposition
        y = x
        # y shape: (B, T, hidden_dim) -> (B, hidden_dim, T) -> convolution -> (B, d_ff, T)
        # here we apply 1x1 convolution, which means we train a 1 x 1 x in_channels x out_channels parameters
        # so it is equal to apply fully-connected neural network on feature dimension
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # (B, d_ff, T) -> (B, hidden_dim, T) -> transpose -> (B, T, hidden_dim)
        # after this, we basically apply 2-layers NN on feature of each timestep
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # apply shortcut again and smooth
        # (B, T, hidden_dim) -> (B, T, hidden_dim)
        res, _ = self.decomp2(x + y)


        return res, attn    # (B, T, hidden_dim), (B, T, num_head, F)


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):

        '''
        x: (B,T,hidden_dim)

        return: x: (B,T,hidden_dim)
        '''

        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):

        '''
        input x: (B,T,hidden_dim)
        input cross : (B,T,hidden_dim)
        '''


        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)

        # using decoder input as query
        # using encoder input as key and value
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)

        # again apply series-decomp -> fcnn -> series-decomp
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        # combine all the trends
        residual_trend = trend1 + trend2 + trend3
        # (B,T,hidden_dim) -> transpose -> (B,hidden_dim,T) -> (B,output_dim,T) -> (B,T,output_dim)
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)

        return x, residual_trend   # (B,T,hidden_dim), (B,T,output_dim)


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        # (B,T,hidden_dim) -> (B,T,c_out)
        if self.projection is not None:
            x = self.projection(x)

        return x, trend    # (B,T,c_out), (B,T,c_out)
