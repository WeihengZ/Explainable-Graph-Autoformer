from locale import normalize
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import optim
import pickle

import argparse
# from models.Autoformer_new import Model as GNN_autoformer
from layers_v3.model5 import Model as GNN_autoformer
from utils.configs import define_setup
from utils.data import trainingset_construct, load_graph_data
from utils.train_test import train, eval, plot_inter_results, plot

# set the parameter for this script
parser = argparse.ArgumentParser(description='multiple tests')
parser.add_argument('--num_pred_len', type=int, default=4, help='num_pred_length')
args_this_script = parser.parse_args()


device = torch.device('cuda:1')

args = define_setup()
sensor_ids, sensor_id_to_ind, W = load_graph_data(r'../data/PEMS_bay/adj_mx_bay.pkl')

args.num_pred_len = args_this_script.num_pred_len


# perform modification to W
N = W.shape[0]
one_hop_adj = np.zeros((N,N))
one_hop_adj[W>0] = 1
two_hop_flag = one_hop_adj @ one_hop_adj
two_hop_adj = np.zeros((N,N))
two_hop_adj[two_hop_flag>0] = 1
adj_list = [torch.from_numpy(one_hop_adj).float().to(device), torch.from_numpy(two_hop_adj).float().to(device)]

# define adj matrix
normalize_scalar = np.sum(W,1)
W /= np.expand_dims(normalize_scalar, 1)
adj = torch.from_numpy(W).float().to(device)

# define model
# model = GNN_autoformer(configs=args, adj=torch.from_numpy(two_hop_adj).float().to(device)).float().to(device)
model = GNN_autoformer(adj=adj ,configs=args, DEVICE=device).float().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()


# define data loader
data = (pd.read_hdf(r'../data/PEMS_bay/pems-bay.h5').to_numpy()).T    # return (N,T)
time_len = np.size(data,1)
time_step_per_week = 7*24*12
train_data = data[:, :8*time_step_per_week]
test_data = data[:, 7*time_step_per_week:9*time_step_per_week]
# normalize the data
mean_val = np.mean(train_data)
std_val = np.std(train_data)
train_data = (train_data - mean_val) / std_val
test_data = (test_data - mean_val) / std_val

train_loader = trainingset_construct(args=args, traffic_data=train_data, batch_val=8,\
     num_data_limit=8000, label='train', Shuffle=True)
test_loader = trainingset_construct(args=args, traffic_data=test_data, batch_val=8,\
     num_data_limit=1600, label='test', Shuffle=False)


for epoch in range(50):

    print(epoch)

    ma = eval(test_loader, model, std_val, mean_val, device, args, args.num_pred_len)

    train_loss = train(train_loader, model, optimizer, criterion, device, args, epoch/40, args.num_pred_len)

    print('training loss:', train_loss)

    torch.save(model.state_dict(), './model_{}.pkl'.format(args.num_pred_len))