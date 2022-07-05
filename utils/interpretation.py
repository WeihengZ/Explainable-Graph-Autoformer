from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from utils.data import trainingset_construct, load_graph_data

def return_one_time(init, delay, delay_score, sp_attn, last_out_len, device):

    '''
    this function is used to update the attention scores backward one layer
    '''

    '''
    init: (N, next_out_len)
    delay: (1,N,H,top_k)
    delay_score: (1,N,H,top_k)
    sp_attn: (1,N,N)
    last_out_len: scalar
    '''

    '''
    return: return_init == (N, last_out_len)
    '''

    # obtain the shape
    N, next_out_len = init.shape
    _, _, H, topk = delay.shape

    # obtain return shape
    return_init = torch.zeros(N,last_out_len).to(device)

    # return the spatial attention
    sp_attn = sp_attn[0,:, :]    # return (N,N)
    sp_attn /= torch.sum(sp_attn,1).unsqueeze(-1).repeat(1,N)     # return (N,N)
    new_init = torch.zeros_like(init)
    # for each node, distribut the attention back to the neighbor node
    for i in range(N):
        for j in range(N):
            new_init[i,:] += init[j,:] * sp_attn[j,i]
    
    # return the temporal attention, (N,2*last_out_len)
    return_init_double = torch.cat((return_init, return_init),-1)
    for i in range(H):
        delay_per_head = delay[0,:,i,:]    # return (N,topk)
        corr_per_head = delay_score[0,:,i,:]     # return (N,topk)
        # apply on each node
        for j in range(N):
            for k in range(topk):
                return_init_double[j, delay_per_head[j,k]:delay_per_head[j,k]+next_out_len] += corr_per_head[j,k] * new_init[j,:]
    
    # obtain the pervious output, (N, last_out_len)
    return_init = return_init_double[:,:last_out_len] + return_init_double[:,last_out_len:2*last_out_len]
    # each head has same importance
    return_init /= H

    return return_init

def attn_cal(explain, out_len_list, device, node_idx, time_slot):

    '''
    this function is used to calculate the important part for the prediction
    '''

    '''
    explain: output of the model, list of the attention scores
    out_len_list: list of the output length
    device: cpu or gpu
    node_idx: (1D) index of the nodes that we focus on
    time_slot: (1D) index of the timesteps that we focus on
    '''

    '''
    return: init == (N, out_len_list[0])
    '''

    # define the number of nodes
    N = np.size(node_idx)
    # define the number of 
    pred_len = np.size(time_slot)

    # extract attention distribution from the explainable item
    delays = []
    delay_scores = []
    spatial_attns = []
    for i in range(len(explain)):
        # delay = (1,N,H,top_k), delay_score = (1,N,H,top_k), sp_attn = (1,N,N)
        delay, delay_score, sp_attn = explain[i]
        delays.append(delay)
        delay_scores.append(delay_score)
        spatial_attns.append(sp_attn)

    # define part of the prediction that we want to analyze
    # specify the nodes and timesteps
    init = torch.zeros(N, pred_len).to(device)    # return (N, pred_len)
    init[node_idx, time_slot] = 1

    # backward to find the important part
    for i in range(len(explain)):
        init = return_one_time(init, delays[-i], delay_scores[-i], spatial_attns[-i], out_len_list[-i], device)

    return init

def forward(input_data, model, target_hour, device, case):

    '''
    this function is used to forward the model to calculate attention scores
    '''

    '''
    a. input_data: shape == (N, T), where N is number of nodes and T is the number of timesteps
            data is stored in the order of Sun->Mon->...->Fri->Sat
            The data is recorded every 5 minutes and T >= 7*24*12
                [:, :7*24*12] is the graph data of first week
                [:, :24*12] is the graph data of first day
                [:, :12] is the graph data of first hour
                [:, :1] is the graph data of first 5 minutes
            
    b. model: trained model.
            future_prediction, list of attention distribution = model(historical_input, historical_timestamp) 
    c. target_hour: the first hour of the predicted horizon
    d. device: cpu or cuda
    e. case: one of the ['Sun', 'Mon', 'Tue', 'Wed', 'Thr', 'Fri', 'Sat']
    '''

    '''
    return: explain (list of attention scores)
    '''

    # extract the shape
    N,T = input_data.shape

    # define basic time lengths and the time stamp
    timestep_a_week = 7*24*12
    timestep_a_day = 24*12

    # define the input data index
    index = np.arange(7*12*24) + (target_hour - 1) * 12
    if case != 'Sun':
        if case == 'Mon':
            index += 12*24
        if case == 'Tue':
            index += 2*12*24
        if case == 'Wed':
            index += 3*12*24
        if case == 'Thr':
            index += 4*12*24
        if case == 'Fri':
            index += 5*12*24
        if case == 'Sat':
            index += 6*12*24

    # define the timestamp for the model input
    time_stamp_week = np.arange(timestep_a_week).repeat(2)
    time_stamp_day = np.arange(timestep_a_day).repeat(2*7)
    time_stamp = np.sin(time_stamp_week/timestep_a_week * 2*np.pi) + np.sin(time_stamp_day/timestep_a_day * 2*np.pi)

    # define the input and timestamp
    inputs = input_data[:,index]    # (N,T=7*24*12)
    inputs = torch.from_numpy(inputs).unsqueeze(0).permute(0,2,1)    # (1,T,N)
    inputs = inputs.float().to(device)
    t = torch.from_numpy(time_stamp[index]).unsqueeze(0)    # (1,T)
    t = t.float().to(device)

    # forward the model to obtain the attention scores for analysis
    _, explain = model(inputs, t)    # return (1,N,T)

    return explain

def filter_nodes(percent, node_id):
    '''
    this function is used to filter out a part of the nearest nodes
    '''
    '''
    node_id: specific node index that we analyze
    percent: in the unit of %, which is the percentage of the nodes of that we want to keep
    '''
    '''
    return: flag_matrix == (N,N), flag_matrix[i,j] == 1 if the node pair are far away enough
    '''

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

    # for each nodes, find the neighbor node satisfying the requirement
    index_cutoff = int(percent * n)
    dist = shortest_dist[node_id,:]
    idx = np.argsort(dist)
    idx = idx[-index_cutoff:]

    return idx


        
''' ------------------------ '''

def find_important_nodes(input_data, model, target_hour, device, case, node_idx, time_slot):

    # node_idx should be only one for this function
    assert np.size(node_idx) == 1, 'this function can only apply to one node'

    explain = forward(input_data, model, target_hour, device, case)

    # return (N,T)
    out_len_list = [7*24*12] + model.out_len
    attn_dist = attn_cal(explain=explain, out_len_list=out_len_list, device=device, node_idx=node_idx, time_slot=time_slot)
    attn_dist = attn_dist.cpu().detach().numpy()

    # return the flag index (1D)
    far_node_index = filter_nodes(percent=0.5, node_id=node_idx)

    # load the sensor location
    sensors = pd.read_csv(r'../data/PEMS_bay/graph_sensor_locations_bay.csv', header=None).to_numpy()
    xy = sensors[:,1:3]

    # plot the nodes attention for the far nodes
    fig = plt.figure()
    attn = np.sum(attn_dist, -1)    # return (N)
    plt.scatter(xy[far_node_index,1],xy[far_node_index,0],c=attn[far_node_index], cmap='Reds')
    plt.plot(xy[node_idx,1],xy[node_idx,0], 'bo', markerfacecolor='none', label='target_node')
    plt.xlim([-122.1, -121.8])
    plt.ylim([37.225,37.45])
    plt.xticks([])
    plt.yticks([])
    plt.title('{}:00'.format(target_hour))
    plt.colorbar()
    plt.savefig(r'./explain/graph.png')




    



