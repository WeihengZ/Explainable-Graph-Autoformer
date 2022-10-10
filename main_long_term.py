import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import optim

import argparse
from models.model import Model as GNN_autoformer
from utils.data import trainingset_construct, load_graph_data, sparse_adj
from utils.train_test import train, eval

# define random seed
torch.manual_seed(0)

# define the device
device = torch.device('cuda:0')

# define set up
parser = argparse.ArgumentParser(description='Autoformer')
# parameter for specific setting of each prediction horizon
parser.add_argument('--pred_len', type=int, default=12*12, help='pred_length')
parser.add_argument('--out_len', nargs='+', type=int, default=[7*24*12, 7*24*12, 12], help='model hyperparameter: output length of each cell')
parser.add_argument('--patch_list', nargs='+', type=int, default=[12, 12, 3, 1], help='model hyperparameter: patch size of each cell')
parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
parser.add_argument('--time_steps', type=int, default=7*24*12, help='length of the input time sequence, default is one week')
# define model architecture
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--topk', type=int, default=4, help='number of segment that we extract in each temporal attention module')
# define whether to do it on Metr-la dataset
parser.add_argument('--metr', type=int, default=1)

args, unknown = parser.parse_known_args()

# load the adjacent matrix
''' dense adj mat '''
if args.metr == 0:
     sensor_ids, sensor_id_to_ind, W = load_graph_data(r'../data/PEMS_bay/adj_mx_bay.pkl')
else:
     sensor_ids, sensor_id_to_ind, W = load_graph_data(r'../data/Metr-LA/adj_mx.pkl')
''' sparse adj mat '''
# W = sparse_adj()
# normalize_scalar = np.sum(W,1)
# W /= np.expand_dims(normalize_scalar, 1)
adj = torch.from_numpy(W).float().to(device)

# define model, optimizer and loss function
model = GNN_autoformer(adj=adj ,configs=args, DEVICE=device).float().to(device)
# model.load_state_dict(torch.load('./trained_models/model_{}_epoch1.pkl'.format(args.pred_len)))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# define data loader
if args.metr == 0:
     data = (pd.read_hdf(r'./data/PEMS_bay/pems-bay.h5').to_numpy()).T    # return (N,T)
else:
     data = (pd.read_hdf(r'./data/Metr-LA/metr-la.h5').to_numpy()).T    # return (N,T)
print(data.shape)

time_step_per_week = 7*24*12
''' The first week of each dataset is only for input and not for target value'''
train_data = data[:, :4*time_step_per_week]
test_data = data[:, 3*time_step_per_week:5*time_step_per_week]
mean_val = np.mean(train_data)
std_val = np.std(train_data)
train_loader = trainingset_construct(args=args, traffic_data=train_data, batch_val=1,\
     num_data_limit=np.inf, Shuffle=True, mean=mean_val, std=std_val)
test_loader = trainingset_construct(args=args, traffic_data=test_data, batch_val=4,\
     num_data_limit=np.inf, Shuffle=False, mean=mean_val, std=std_val)

# perform training and testing
for epoch in range(5):
    torch.autograd.set_detect_anomaly(True)
    print('Current epoch', epoch)
    train_loss = train(train_loader, model, optimizer, criterion, device, mean_val, std_val, 8)
    print('training loss:', train_loss)
    ma = eval(test_loader, model, device, args, mean_val, std_val)
    if args.metr == 0:
        torch.save(model.state_dict(), r'./trained_models/model_{}_epoch{}.pkl'.format(args.pred_len, epoch))
    

    