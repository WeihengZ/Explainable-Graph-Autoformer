__all__ = ['trainingset_construct', 'load_graph_data', 'sparse_adj', 'scaled_Laplacian_list']

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs


def training_loader_construct(dataset, batch_num, Shuffle):

    # construct the train loader given the dataset and batch size value
    # this function can be used for all different cases 

    train_loader = DataLoader(
        dataset,
        batch_size=batch_num,
        shuffle=Shuffle,                     # change the sequence of the data every time
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader

# define training loader construction function
class MyDataset(Dataset):
    def __init__(self, traffic_data, args, num_data_limit, mean, std, transform=None):

        '''
        input
            traffic_data (N,T)
            pred_len: (scalar)
            args: 
        '''
        # extract information
        assert args.pred_len % 12 == 0, 'this script is only for short-term traffic prediction'
        pred_len = args.pred_len
        input_length = args.pred_len

        PEMS =  traffic_data   # return (N,T)
        print('traffic data shape:', PEMS.shape)

        timestep_a_week = 7*24*12
        timestep_a_day = 24*12
        time_stamp_week = np.arange(timestep_a_week).repeat(30)
        time_stamp_day = np.arange(timestep_a_day).repeat(30*7)
        t = np.sin(time_stamp_week/timestep_a_week * 2*np.pi) + np.sin(time_stamp_day/timestep_a_day * 2*np.pi)

        self.x = []
        self.y = []
        self.tx = []
        self.ty = []

        sample_steps = 1
        num_datapoints = int(np.floor((PEMS.shape[1] - timestep_a_week) / sample_steps))
        print('total number of datapoints:', num_datapoints)
        starting_point = timestep_a_week
        endding_point = PEMS.shape[1] - pred_len

        num_data = 0
        for k in range(starting_point, endding_point, sample_steps):
            if num_data < num_data_limit:

                input_data = []
                # store weekly data
                input_data.append(np.expand_dims(PEMS[:,k-timestep_a_week: k-timestep_a_week+pred_len], axis=-1))
                input_data.append(np.expand_dims(PEMS[:,k-timestep_a_day: k-timestep_a_day+pred_len], axis=-1))
                input_data.append(np.expand_dims(PEMS[:,k-pred_len: k], axis=-1))
                input_data = np.concatenate(tuple(input_data), axis=-1)    # return (N,T,F)
        
                self.x.append((input_data - mean) / std)
                self.y.append(np.array(PEMS[:, k : k + pred_len]))
                self.tx.append(t[k-pred_len: k])
                self.ty.append(t[k:k+pred_len])
                num_data += 1

        print('data created,', 'input shape:', len(self.x), 'output shape:', len(self.y))

        self.x = torch.from_numpy(np.array(self.x)).float()
        self.y = torch.from_numpy(np.array(self.y)).float()
        self.tx = torch.from_numpy(np.array(self.tx)).float()
        self.ty = torch.from_numpy(np.array(self.ty)).float()

        self.transform = transform
        
    def __getitem__(self, index):

        x = self.x[index]
        y = self.y[index]
        tx = self.tx[index]
        ty = self.ty[index]

        return x, y, tx, ty
    
    def __len__(self):

        assert len(self.x)==len(self.y), 'length of input and output are not the same'   

        return len(self.x)   

def trainingset_construct(args, traffic_data, batch_val, num_data_limit, Shuffle, mean, std):
    dataset = MyDataset(traffic_data, args, num_data_limit, mean, std)
    train_loader = training_loader_construct(dataset = dataset,batch_num = batch_val,Shuffle=Shuffle)

    return train_loader

def load_pickle(pickle_file):
        try:
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f)
        except UnicodeDecodeError as e:
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f, encoding='latin1')
        except Exception as e:
            print('Unable to load data ', pickle_file, ':', e)
            raise
        return pickle_data

def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx

def sparse_adj():
    # load the adjacency matrix
    _, _, adj = load_graph_data(r'../data/PEMS_bay/adj_mx_bay.pkl')

    # load sensor location
    sensors = pd.read_csv(r'../data/PEMS_bay/graph_sensor_locations_bay.csv', header=None).to_numpy()
    xy = sensors[:,1:3]
    x = xy[:,1]
    y = xy[:,0]

    # derive the adjacency matrix
    n = np.size(x)
    ADJ = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if adj[i,j] > 0:
                ADJ[i,j] = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)

    # extract shortest distance between any two nodes (N,N)
    shortest_dist = sp.csgraph.dijkstra(ADJ)

    # some is infinity
    shortest_dist[shortest_dist==np.inf] = 0

    print(np.sum(shortest_dist, 1))


    # define a flag matrix
    flag = np.ones((n,n))
    flag[shortest_dist==0] = 0

    # change it to adjacency matrix
    sigma = np.std(shortest_dist)
    mean = np.mean(shortest_dist)

    # define adj (N,N)
    W = np.exp(- shortest_dist**2 / sigma**2)
    print(W)
    # W[shortest_dist<mean] = 0

    return W

def scaled_Laplacian_list(W, device):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    L_tilde = (2 * L) / lambda_max - np.identity(W.shape[0])

    N = L_tilde.shape[0]

    cheb_polynomials = [torch.eye(N).to(device), torch.from_numpy(L_tilde.copy()).to(device)]
    L_tilde = torch.from_numpy(L_tilde).to(device)

    for i in range(2, 3):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials