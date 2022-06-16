__all__ = ['trainingset_construct', 'load_graph_data']

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
import numpy as np


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
    def __init__(self, traffic_data, pred_len, num_data_limit, label='train', transform=None):

        '''
        input
            traffic_data (N,T)
            pred_len: (scalar)
            args: 
        '''

        PEMS =  traffic_data   # return (N,T)
        print('traffic data shape:', PEMS.shape)

        timestep_a_week = 7*24*12
        timestep_a_day = 24*12
        time_stamp_week = np.arange(timestep_a_week).repeat(15)
        time_stamp_day = np.arange(timestep_a_day).repeat(15*7)
        t = np.sin(time_stamp_week/timestep_a_week * 2*np.pi) + np.sin(time_stamp_day/timestep_a_day * 2*np.pi)

        print('timestamp shape', t.shape)

        self.x = []
        self.y = []
        self.tx = []
        self.ty = []

        sample_steps = 1
        num_datapoints = int(np.floor((PEMS.shape[1] - 7*24*12) / sample_steps))
        print('total number of datapoints:', num_datapoints)
        starting_point = 7*24*12
        endding_point = PEMS.shape[1] - pred_len

        num_data = 0
        for k in range(starting_point, endding_point, sample_steps):
            if num_data < num_data_limit:
                self.x.append(PEMS[:,k-7*24*12:k])
                self.y.append(np.array(PEMS[:, k:k+pred_len]))
                self.tx.append(t[k-7*24*12:k])
                self.ty.append(t[k:k+pred_len])
                num_data += 1

        print('{} data created,'.format(label), 'input shape:', len(self.x), 'output shape:', len(self.y))

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

def trainingset_construct(args, traffic_data, batch_val, num_data_limit, label, Shuffle):
    dataset = MyDataset(traffic_data, args.num_pred_len*args.pred_len, num_data_limit, label)
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