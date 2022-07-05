import torch
import torch.nn as nn
import numpy as np
import pandas as pd


import argparse
# from models.Autoformer_new import Model as GNN_autoformer
from models.model import Model as GNN_autoformer
from utils.configs import define_setup
from utils.data import trainingset_construct, load_graph_data
from utils.train_test import train, eval, plot, test_error, find_important_input, find_important_nodes
from utils.interpretation import find_important_nodes

# set the parameter for this script
parser = argparse.ArgumentParser(description='multiple tests')
parser.add_argument('--num_pred_len', type=int, default=6, help='num_pred_length')
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
adj = torch.from_numpy(W).float().to(device)

# define adj matrix
normalize_scalar = np.sum(W,1)
W /= np.expand_dims(normalize_scalar, 1)
adj = torch.from_numpy(W).float().to(device)

# define model
# model = GNN_autoformer(configs=args, adj=torch.from_numpy(two_hop_adj).float().to(device)).float().to(device)
model = GNN_autoformer(adj=adj ,configs=args, DEVICE=device).float().to(device)
model.load_state_dict(torch.load('./model_{}.pkl'.format(args.num_pred_len)))


# define data loader
data = (pd.read_hdf(r'../data/PEMS_bay/pems-bay.h5').to_numpy()).T    # return (N,T)
time_len = np.size(data,1)
time_step_per_week = 7*24*12
train_data = data[:, :4*time_step_per_week]
test_data = data[:, 3*time_step_per_week:5*time_step_per_week]
# normalize the data
mean_val = np.mean(train_data)
std_val = np.std(train_data)
test_data = (test_data - mean_val) / std_val

case_name = ['Sun','Mon', 'Tue', 'Wed', 'Thr', 'Fri','Sat']
# for pp in range(len(case_name)):
# find_important_input(test_data, model, device, args.num_pred_len, args, case=case_name[1], target_hour=8, node_idx=307)

# plot the atttention graph
find_important_nodes(input_data=test_data, model=model, target_hour=1, device=device, case='Mon', node_idx=0, time_slot=np.arange(model.pred_len))