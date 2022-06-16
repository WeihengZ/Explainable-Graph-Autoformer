import argparse
import torch

def define_setup():

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # forecasting task:
    # num_week: same time slot of previous few weeks 
    # num_day: same time slot of previous few days 
    # num_hour: time slot of previous few hours 
    # pred_len: 12 * num_hours_predicted
    parser.add_argument('--num_week', type=int, default=7, help='num_week')
    parser.add_argument('--num_day', type=int, default=7, help='num_day')
    parser.add_argument('--num_hour', type=int, default=3, help='num_hour')
    parser.add_argument('--num_pred_len', type=int, default=24, help='num_hour')
    parser.add_argument('--pred_len', type=int, default=12, help='pred_length')

    # this four parameters must always equal to number of nodes
    parser.add_argument('--enc_in', type=int, default=325, help='encoder input size')
    parser.add_argument('--num_nodes', type=int, default=325, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=325, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=325, help='output size')
    parser.add_argument('--d_model', type=int, default=1, help='dimension of model')

    # this parameter needs to be equal to one to make the model explainable
    parser.add_argument('--n_heads', type=int, default=1, help='num of heads')

    # window size of smoothing the time series
    # remember to add 1 with your true window size
    parser.add_argument('--moving_avg', type=int, default=145, help='window size of moving average')

    # define model architecture
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--factor', type=int, default=1, help='factor for number of periods extracted')

    # define whether to output the attention score, default is true
    parser.add_argument('--output_attention', action='store_false', help='whether to output attention in encoder')    # default = False

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False


    return args