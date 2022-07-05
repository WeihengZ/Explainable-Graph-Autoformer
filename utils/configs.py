import argparse
import torch

def define_setup():

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # parameter for specific setting of each prediction horizon
    parser.add_argument('--pred_len', type=int, default=12, help='pred_length')
    parser.add_argument('--out_len', type=list, default=[12*24, 12*6, 12*3], help='model hyperparameter: output length of each cell')
    parser.add_argument('--patch_list', type=list, default=[12, 6, 3, 1], help='model hyperparameter: patch size of each cell')
    parser.add_argument('--num_nodes', type=int, default=325, help='number of nodes')
    parser.add_argument('--time_steps', type=int, default=7*24*12, help='length of the input time sequence, default is one week')

    # define model architecture
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--topk', type=int, default=4, help='number of segment that we extract in each temporal attention module')

    args = parser.parse_args()

    return args