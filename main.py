import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import optim
import pickle

import argparse
# from models.Autoformer_new import Model as GNN_autoformer
from models.GNN_model import GNN_autoformer

device = torch.device('cuda:0')

if "set_config" == "set_config":

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    # parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    # parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    # parser.add_argument('--model', type=str, required=True, default='Autoformer',
    #                     help='model name, options: [Autoformer, Informer, Transformer]')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=244, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=196, help='start token length')
    parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')

    # model define
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')

    ''' modify here'''
    parser.add_argument('--enc_in', type=int, default=325, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=325, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=325, help='output size')
    parser.add_argument('--d_model', type=int, default=325, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=5, help='num of heads')
    parser.add_argument('--moving_avg', type=int, default=13, help='window size of moving average')
    parser.add_argument('--freq', type=int, default=12, help='by now it is not used')
    ''' modify region stop line '''


    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')    # default = False
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if "define_dataset" == "define_dataset":

    def training_loader_construct(dataset,batch_num):

        # construct the train loader given the dataset and batch size value
        # this function can be used for all different cases 

        train_loader = DataLoader(
            dataset,
            batch_size=batch_num,
            shuffle=True,                     # change the sequence of the data every time
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader

    # define training loader construction function
    class MyDataset(Dataset):
        def __init__(self, traffic_data, seq_len, label_len, pred_len, transform=None):

            '''
            input: traffic_data (N,T)
            '''

            self.x = []
            self.y = []
            PEMS =  traffic_data   # return (N,T)
            sample_steps = 1
            num_datapoints = int(np.floor((PEMS.shape[1] - pred_len - seq_len) / sample_steps))
            for k in range(num_datapoints):
                self.x.append(PEMS[:,sample_steps*k : sample_steps*k + seq_len])
                self.y.append(PEMS[:,sample_steps*k + seq_len : sample_steps*k + seq_len + pred_len])
            self.x = torch.from_numpy(np.array(self.x)).float()
            self.y = torch.from_numpy(np.array(self.y)).float()

            self.transform = transform
            
        def __getitem__(self, index):

            x = self.x[index]
            y = self.y[index]

            return x, y
        
        def __len__(self):
            return len(self.x)   

    def trainingset_construct(args, traffic_data, batch_val):
        dataset = MyDataset(traffic_data, args.seq_len, args.label_len, args.pred_len)
        train_loader = training_loader_construct(dataset = dataset,batch_num = batch_val)

        return train_loader

if "load_adj" == "load_adj":

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

    sensor_ids, sensor_id_to_ind, W = load_graph_data(r'../data/PEMS_bay/adj_mx_bay.pkl')

    # perform modification to W
    N = W.shape[0]
    one_hop_adj = np.zeros((N,N))
    one_hop_adj[W>0] = 1
    two_hop_flag = one_hop_adj @ one_hop_adj
    two_hop_adj = np.zeros((N,N))
    two_hop_adj[two_hop_flag>0] = 1

    adj_list = [torch.from_numpy(one_hop_adj).float().to(device), torch.from_numpy(two_hop_adj).float().to(device)]


# define model
# model = GNN_autoformer(configs=args, adj=torch.from_numpy(two_hop_adj).float().to(device)).float().to(device)
model = GNN_autoformer(adj=adj_list[1] ,configs=args, DEVICE=device, in_channels=1, num_of_vertices=args.enc_in, num_of_timesteps=0).float().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.MSELoss()

# define training function
def train(loader, model, optimizer, criterion, device):

    '''
    p.s. input is (batch, #node, #time_step, #feature)
         output is (batch, #node, #time_step)
    '''

    batch_loss = 0 
    for idx, (inputs, targets) in enumerate(tqdm(loader)):

        model.train()
        optimizer.zero_grad()
        inputs = inputs.permute(0,2,1).to(device)  # (B,T,N)
        targets = targets.permute(0,2,1).to(device)    # (B,T,N)
        outputs = model(inputs, targets, adj_list)    # (B,T,N)

        # pick the predicted segment
        outputs = outputs[:, -args.pred_len:, :]
        targets = targets[:, -args.pred_len:, :]

        loss = criterion(outputs, targets) 
        loss.backward()
        optimizer.step()

        batch_loss += loss.detach().cpu().item()

    return batch_loss / (idx + 1)

# define evaluation function
@torch.no_grad()
def eval(loader, model, std, mean, device):
    # batch_rmse_loss = np.zeros(12)
    batch_mae_loss = np.zeros(args.pred_len)

    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.eval()

        inputs = (inputs).permute(0,2,1).to(device)  # (B,T,N)
        targets = targets.permute(0,2,1).to(device)  # (B,T,N)
        outputs = model(inputs, targets, adj_list)     # [B, T,N]

        # pick the predicted segment
        outputs = outputs[:, -args.pred_len:, :]
        targets = targets[:, -args.pred_len:, :]
        
        out_unnorm = outputs.detach().cpu().numpy()
        target_unnorm = targets.detach().cpu().numpy()

        mae_loss = np.zeros(args.pred_len)
        for k in range(out_unnorm.shape[1]):
            err = np.mean(np.abs(out_unnorm[:,k,:] - target_unnorm[:,k,:]) * std)
            mae_loss[k] = err
        
        batch_mae_loss += mae_loss
       
    
    print('mae loss:', batch_mae_loss / (idx + 1))

    return batch_mae_loss / (idx + 1)

# define data loader
data = (pd.read_hdf(r'../data/PEMS_bay/pems-bay.h5').to_numpy()).T    # return (N,T)
time_len = np.size(data,1)
train_data = data[:, :int(0.2*time_len)]
test_data = data[:, int(0.2*time_len):int(0.24*time_len)]
# normalize the data
mean_val = np.mean(train_data)
std_val = np.std(train_data)
train_data = (train_data - mean_val) / std_val
test_data = (test_data - mean_val) / std_val

train_loader = trainingset_construct(args, train_data, 4)
test_loader = trainingset_construct(args, test_data, 16)


for epoch in range(30):

    torch.save(model, './model.pkl')


    print(epoch)

    train_loss = train(train_loader, model, optimizer, criterion, device)

    print('training loss:', train_loss)

    ma = eval(test_loader, model, std_val, mean_val, device)