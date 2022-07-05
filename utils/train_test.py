from cProfile import label
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from utils.data import trainingset_construct, load_graph_data
import time

# define training function
def train(loader, model, optimizer, criterion, device, args):

    '''
    p.s. input is (batch, #node, #time_step, #feature)
         output is (batch, #node, #time_step)
    '''

    batch_loss = 0 
    for idx, (inputs, targets, tx, ty) in enumerate(tqdm(loader)):

        model.train()
        optimizer.zero_grad()

        update_step = 8

        inputs = inputs.permute(0,2,1).to(device)  # (B,T,N)
        targets = targets.permute(0,2,1).to(device)    # (B,T,N)
        tx = tx.to(device)    # (B,T)
        ty = ty.to(device)    # (B,T)
        outputs = model.forward(inputs, tx)[0]    # (B,T,N)

        # pick the predicted segment

        outputs = outputs[:, :args.pred_len, :]
        targets = targets[:, :args.pred_len, :]


        loss = criterion(outputs, targets) / update_step
        loss.backward()
        if idx % update_step == 0:
            optimizer.step()
            model.zero_grad()

        batch_loss += loss.detach().cpu().item()

    return batch_loss / (idx + 1)



@torch.no_grad()
def eval(loader, model, device, args):
    # batch_rmse_loss = np.zeros(12)
    batch_mae_loss = np.zeros(args.pred_len)

    # save the results
    new_delays = []
    new_t_attns = []
    new_sp_attns = []

    for idx, (inputs, targets, tx, ty) in enumerate(tqdm(loader)):
        model.eval()

        inputs = (inputs).permute(0,2,1).to(device)  # (B,T,N)
        targets = targets.permute(0,2,1).to(device)  # (B,T,N)
        tx = tx.to(device)    # (B,T)
        outputs, _ = model.forward(inputs, tx)     # [B,T,N]

        outputs = outputs[:, :args.pred_len, :]
        targets = targets[:, :args.pred_len, :]
        
        out_unnorm = outputs.detach().cpu().numpy()
        target_unnorm = targets.detach().cpu().numpy()

        mae_loss = np.zeros(args.pred_len)
        for k in range(out_unnorm.shape[1]):
            err = np.mean(np.abs(out_unnorm[:,k,:] - target_unnorm[:,k,:]))
            mae_loss[k] = err
        
        batch_mae_loss += mae_loss

    print('mae loss:', batch_mae_loss / (idx + 1))

    return batch_mae_loss / (idx + 1)


@torch.no_grad()
def test_error(loader, model, std, mean, device, args):
    # batch_rmse_loss = np.zeros(12)
    node_mae_loss = np.zeros(args.enc_in)

    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.eval()

        inputs = (inputs).permute(0,2,1).to(device)  # (B,T,N)
        targets = targets.permute(0,2,1).to(device)  # (B,T,N)
        outputs = model(inputs)[0]     # [B, T,N]

        # pick the predicted segment
        outputs = outputs[:, -args.pred_len:, :]    # return (B,T,N)
        targets = targets[:, -args.pred_len:, :]
        
        out_unnorm = outputs.detach().cpu().numpy()
        target_unnorm = targets.detach().cpu().numpy()

        for k in range(args.enc_in):
            err = np.mean(np.abs(out_unnorm[:,:,k] - target_unnorm[:,:,k]) * std)
            node_mae_loss[k] += err

    batch_mae_loss = node_mae_loss / (idx + 1)

    # load sensor location
    sensors = pd.read_csv(r'../data/PEMS_bay/graph_sensor_locations_bay.csv', header=None).to_numpy()
    xy = sensors[:,1:3]

    fig = plt.figure()
    plt.scatter(xy[:,1],xy[:,0],c=batch_mae_loss, cmap='Reds')
    plt.colorbar()
    plt.title('Error distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(r'./tests/node_error_dist.png')


    print('mae loss:', batch_mae_loss / (idx + 1))

    return batch_mae_loss 


@torch.no_grad()
def plot(loader, model, std, mean, device, args, node_id, num_forward):

    for idx, (inputs, targets, tx, ty) in enumerate(tqdm(loader)):
        model.eval()

        if idx == 12:

            inputs = (inputs).permute(0,2,1).to(device)  # (B,T,N)
            targets = targets.permute(0,2,1).to(device)  # (B,T,N)
            tx = tx.to(device)    
            outputs = model.forward(inputs, tx, num_forward)[0]     # [B, T,N]

            # pick the predicted segment
            outputs = outputs[0, :args.num_pred_len*args.pred_len, node_id]*std + mean  # return (T)
            targets = targets[0, :args.num_pred_len*args.pred_len, node_id]*std + mean     # return (T)
            
            out_unnorm = outputs.detach().cpu().numpy()
            target_unnorm = targets.detach().cpu().numpy()

            # plot
            fig = plt.figure()
            x = (np.arange(args.pred_len*args.num_pred_len) + 1) / 12
            plt.plot(x, target_unnorm, label='grond truth')
            plt.plot(x, out_unnorm, label='prediction')
            plt.legend(loc=0)
            plt.grid()
            plt.ylabel('speed (mph)')
            plt.xlabel('time (hour)')

            plt.ylim(20,80)
            plt.savefig(r'./tests/{}_{}.png'.format(node_id, num_forward))

        if idx >= 51:
            break

    return 0



@torch.no_grad()
def plot_temp_attn_score(input_data, model, device, num_hour, args, case, target_hour):
    '''
    We know that data are sorted in the order of [Sun, Mon, ..., Fri, Sat]
    The number of time steps each day is 12*24 
    '''

    '''
    input
    input_data: (N,T)

    '''
    N,T = input_data.shape

    index = np.arange(7*12*24) + (target_hour - num_hour)*12
    day_id = 0
    if case != 'Sun':
        if case == 'Mon':
            index += 12*24
            day_id += 1
        if case == 'Tue':
            index += 2*12*24
            day_id += 2
        if case == 'Wed':
            index += 3*12*24
            day_id += 3
        if case == 'Thr':
            index += 4*12*24
            day_id += 4
        if case == 'Fri':
            index += 5*12*24
            day_id += 5
        if case == 'Sat':
            index += 6*12*24
            day_id += 6

    timestep_a_week = 7*24*12
    timestep_a_day = 24*12
    time_stamp_week = np.arange(timestep_a_week).repeat(15)
    time_stamp_day = np.arange(timestep_a_day).repeat(15*7)
    time_stamp = np.sin(time_stamp_week/timestep_a_week * 2*np.pi) + np.sin(time_stamp_day/timestep_a_day * 2*np.pi)
    
    # store information
    factor = 12
    attn_dist = np.zeros((N,168*factor))

    for i in range(1):
        inputs = input_data[:,index + i * timestep_a_week]    # (N,T)
        inputs = torch.from_numpy(inputs).unsqueeze(0).permute(0,2,1)    # (1,T,N)
        inputs = inputs.float().to(device)
        t = torch.from_numpy(time_stamp[index]).unsqueeze(0)    # (1,T)
        t = t.float().to(device)

        target = input_data[:, int(index[-1] + (num_hour-1)*args.pred_len) :int(index[-1] + (num_hour)*args.pred_len)]
    
        # forward the model
        outputs, explain = model(inputs, t)    # return (1,N,T)

        # extract information
        delay = explain[0][-1].squeeze(0).detach().cpu().numpy()    # (N,168)
        t_atten_score = explain[1][-1].squeeze(0).detach().cpu().numpy()    # (N,168)

        # store the information
        N,topk = delay.shape
        for j in range(N):
            for k in range(topk):
                # each item is (N,top_k)
                attn_dist[j,delay[j,k]] += t_atten_score[j,k]

        # make a plot
        fig = plt.figure()
        x = np.arange(168 * factor) + (target_hour - num_hour) * 12
        # attn_score = np.mean(attn_dist, 0)   # (168)
        attn_score = np.zeros(168 * factor)
        attn_score[-1] = 1
        attn_score = attn_score / np.amax(attn_score)

        # plot the prediction timestep
        plt.plot((168+target_hour)*factor, 0, 'o', label='predicted hour')

        plt.plot(x, attn_score, c='red', label='{} hours ahead prediction attention distribution'.format(num_hour)) 
        plt.vlines(x=(np.array([0,24,48,72,96,120,144,168])) * factor, ymin=0, ymax=1.2, colors='k', linestyles='dashed')
        
        days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thr', 'Fri', 'Sat']
        
        plt.xticks(ticks = (np.array([0,24,48,72,96,120,144,168]) + target_hour) * factor,
                   labels=['{}:00 \n {}'.format(target_hour, days[day_id % 7]), '{}:00 \n {}'.format(target_hour, days[(day_id+1) % 7 ]),
        '{}:00 \n {}'.format(target_hour, days[(day_id+2) % 7]), '{}:00 \n {}'.format(target_hour, days[(day_id+3) % 7]), '{}:00 \n {}'.format(target_hour, days[(day_id+4) % 7]),
        '{}:00 \n {}'.format(target_hour, days[(day_id+5) % 7]),'{}:00 \n {}'.format(target_hour, days[(day_id+6) % 7]), '{}:00 \n {}'.format(target_hour, 'future')])
        
        
        for k in range(day_id,day_id+7):
            pos = k - day_id
            plt.text(x=(6 + 24*pos) * factor, y=1.1, s=days[k % 7], c='red')
        
        plt.title('Case of {} {}:00 prediction ({} hours future)'.format(case, target_hour, 1))
        plt.ylabel('Attention score')
        plt.legend(loc=0)
        plt.savefig(r'./case_explain/atten_dist_{}_hours_case_{}_target_{}'.format(args.num_pred_len, case, target_hour))

def plot_spatial_attns(node_id, node_attns, args):

    # load sensor location
    sensors = pd.read_csv(r'../data/PEMS_bay/graph_sensor_locations_bay.csv', header=None).to_numpy()
    xy = sensors[:,1:3]

    # extract the final distribution
    node_attn = node_attns[-1]
    node_attn = np.mean(node_attn.cpu().detach().numpy(), -1)

    fig = plt.figure(figsize=(20,4))
    plt.subplot(1,4,1)
    plt.scatter(xy[:,1],xy[:,0],c=np.mean(node_attns[0].cpu().detach().numpy(),-1), cmap='Reds')
    plt.plot(xy[node_id,1],xy[node_id,0], 'bo', markerfacecolor='none', label='target_node')
    plt.xlim([-122.1, -121.8])
    plt.ylim([37.225,37.45])
    plt.colorbar()
    plt.subplot(1,4,2)
    plt.scatter(xy[:,1],xy[:,0],c=np.mean(node_attns[1].cpu().detach().numpy(),-1), cmap='Reds')
    plt.plot(xy[node_id,1],xy[node_id,0], 'bo', markerfacecolor='none', label='target_node')
    plt.xlim([-122.1, -121.8])
    plt.ylim([37.225,37.45])
    plt.colorbar()
    plt.subplot(1,4,3)
    plt.scatter(xy[:,1],xy[:,0],c=np.mean(node_attns[2].cpu().detach().numpy(),-1), cmap='Reds')
    plt.plot(xy[node_id,1],xy[node_id,0], 'bo', markerfacecolor='none', label='target_node')
    plt.xlim([-122.1, -121.8])
    plt.ylim([37.225,37.45])
    plt.colorbar()
    plt.subplot(1,4,4)
    plt.scatter(xy[:,1],xy[:,0],c=np.mean(node_attns[3].cpu().detach().numpy(),-1), cmap='Reds')
    plt.plot(xy[node_id,1],xy[node_id,0], 'bo', markerfacecolor='none', label='target_node')
    plt.xlim([-122.1, -121.8])
    plt.ylim([37.225,37.45])
    plt.colorbar()
    plt.savefig(r'./explain/dist_evolution_node_{}.png'.format(node_id))


    fig = plt.figure(figsize=(20,4))
    plt.subplot(1,3,1)
    plt.scatter(xy[:,1],xy[:,0],c=node_attn, cmap='Reds')
    plt.plot(xy[node_id,1],xy[node_id,0], 'bo', markerfacecolor='none', label='target_node')
    plt.xlim([-122.1, -121.8])
    plt.ylim([37.225,37.45])
    plt.colorbar()
    plt.title('spatial attention distribution of node {}'.format(node_id))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=0)

    plt.subplot(1,3,2)
    plt.hist(node_attn, bins=50)
    plt.xlabel('Attention scores')
    plt.ylabel('Frequency')

    plt.subplot(1,3,3)
    idx = np.argsort(node_attn)
    idx_cutout = int(325 * 0.05)
    idx = idx[-idx_cutout:]
    node_attn_update = node_attn[idx]    # return idx

    plt.scatter(xy[idx,1],xy[idx,0],c=node_attn_update, cmap='Reds')
    plt.plot(xy[node_id,1],xy[node_id,0], 'bo', markerfacecolor='none', label='target_node')
    plt.xlim([-122.1, -121.8])
    plt.ylim([37.225,37.45])
    plt.colorbar()
    plt.title('spatial attention distribution of node {}'.format(node_id))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=0)



    plt.savefig(r'./explain/{}_node_{}_spatial_attn_dist.png'.format(args.num_pred_len, node_id))


@torch.no_grad()
def find_important_input(input_data, model, device, num_hour, args, case, target_hour, node_idx):
    '''
    We know that data are sorted in the order of [Sun, Mon, ..., Fri, Sat]
    The number of time steps each day is 12*24 
    '''

    '''
    input
    input_data: (N,T)

    '''
    N,T = input_data.shape

    index = np.arange(7*12*24) + (target_hour - num_hour)*12
    day_id = 0
    if case != 'Sun':
        if case == 'Mon':
            index += 12*24
            day_id += 1
        if case == 'Tue':
            index += 2*12*24
            day_id += 2
        if case == 'Wed':
            index += 3*12*24
            day_id += 3
        if case == 'Thr':
            index += 4*12*24
            day_id += 4
        if case == 'Fri':
            index += 5*12*24
            day_id += 5
        if case == 'Sat':
            index += 6*12*24
            day_id += 6

    timestep_a_week = 7*24*12
    timestep_a_day = 24*12
    time_stamp_week = np.arange(timestep_a_week).repeat(15)
    time_stamp_day = np.arange(timestep_a_day).repeat(15*7)
    time_stamp = np.sin(time_stamp_week/timestep_a_week * 2*np.pi) + np.sin(time_stamp_day/timestep_a_day * 2*np.pi)

    # forward
    for i in range(1):
        inputs = input_data[:,index + i * timestep_a_week]    # (N,T)
        inputs = torch.from_numpy(inputs).unsqueeze(0).permute(0,2,1)    # (1,T,N)
        inputs = inputs.float().to(device)
        t = torch.from_numpy(time_stamp[index]).unsqueeze(0)    # (1,T)
        t = t.float().to(device)

        target = input_data[:, int(index[-1] + (num_hour-1)*args.pred_len) :int(index[-1] + (num_hour)*args.pred_len)]
    
        # forward the model
        outputs, explain = model(inputs, t)    # return (1,N,T)

        # return (N,T)
        input_time_len = 7 * timestep_a_day 
        attn_dists = attn_cal(N=N, pred_len=args.pred_len, explain=explain, out_len_list=[12*24, 12*8], origin_timestep=input_time_len, device=device, node_idx=node_idx)

        # return it back to numpy, (N,T)
        attn_dist = attn_dists[-1]
        attn_dist = attn_dist.cpu().detach().numpy()



        # make a plot for spatial attention
        plot_spatial_attns(node_id=node_idx, node_attns=attn_dists, args=args)


        # make a plot
        fig = plt.figure()
        x = np.arange(input_time_len) + (target_hour - num_hour) * 12
        attn_score = np.mean(attn_dist, 0)   # (timestep_a_week)
        attn_score = attn_score / np.amax(attn_score)

        # define a factor
        factor = 12

        # plot the prediction timestep
        plt.plot(input_time_len + (target_hour)*factor, 0, 'o', label='predicted hour')

        plt.plot(x, attn_score, c='red', label='{} hours ahead prediction attention distribution'.format(num_hour)) 
        plt.vlines(x=(np.array([0,24,48,72,96,120,144,168])) * factor, ymin=0, ymax=1.2, colors='k', linestyles='dashed')
        
        days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thr', 'Fri', 'Sat']
        
        plt.xticks(ticks = (np.array([0,24,48,72,96,120,144,168]) + target_hour) * factor,
                   labels=['{}:00 \n {}'.format(target_hour, days[day_id % 7]), '{}:00 \n {}'.format(target_hour, days[(day_id+1) % 7 ]),
        '{}:00 \n {}'.format(target_hour, days[(day_id+2) % 7]), '{}:00 \n {}'.format(target_hour, days[(day_id+3) % 7]), '{}:00 \n {}'.format(target_hour, days[(day_id+4) % 7]),
        '{}:00 \n {}'.format(target_hour, days[(day_id+5) % 7]),'{}:00 \n {}'.format(target_hour, days[(day_id+6) % 7]), '{}:00 \n {}'.format(target_hour, 'future')])
        
        
        for k in range(day_id,day_id+7):
            pos = k - day_id
            plt.text(x=(6 + 24*pos) * factor, y=1.1, s=days[k % 7], c='red')
        
        plt.title('Case of {} {}:00 prediction ({} hours future)'.format(case, target_hour, 1))
        plt.ylabel('Attention score')
        plt.legend(loc=0)
        plt.savefig(r'./case_explain/atten_dist_{}_hours_case_{}_target_{}_node_{}'.format(args.num_pred_len, case, target_hour, node_idx)) 

def attn_cal(N, pred_len, explain, out_len_list, origin_timestep, device, node_idx):

    # extract information
    # (1,N,H,top_k), (1,N,H,top_k), (1,N,N)
    delay1, delay_score1, sp_attn1 = explain[0]
    delay2, delay_score2, sp_attn2 = explain[1]
    delay3, delay_score3, sp_attn3 = explain[2]
    delay4, delay_score4, sp_attn4 = explain[3]

    # calculate important part
    # define one node index
    init = torch.zeros(N, pred_len).to(device)    # return (N, pred_len)
    init[node_idx,:] = 1

    # apply backward
    init4 = return_one_time(init, delay4, delay_score4, sp_attn4, out_len_list[-1], device)
    init3 = return_one_time(init4, delay3, delay_score3, sp_attn3, out_len_list[-2], device)
    init2 = return_one_time(init3, delay2, delay_score2, sp_attn2, out_len_list[-2], device)
    init1 = return_one_time(init2, delay1, delay_score1, sp_attn1, origin_timestep, device)    # (return N,T)

    return [init4,init3, init2, init1]

def return_one_time(init, delay, delay_score, sp_attn, last_out_len, device):
    '''
    init: (N, next_out_len)
    delay: (1,N,H,top_k)
    delay_score: (1,N,H,top_k)
    sp_attn: (1,N,N)
    last_out_len: scalar
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

def forward(input_data, model, target_hour, device, args, node_idx, case, num_hour=1):

    # extract the shape
    N,T = input_data.shape

    # define the input data
    index = np.arange(7*12*24) + (target_hour - num_hour) * 12
    day_id = 0
    if case != 'Sun':
        if case == 'Mon':
            index += 12*24
            day_id += 1
        if case == 'Tue':
            index += 2*12*24
            day_id += 2
        if case == 'Wed':
            index += 3*12*24
            day_id += 3
        if case == 'Thr':
            index += 4*12*24
            day_id += 4
        if case == 'Fri':
            index += 5*12*24
            day_id += 5
        if case == 'Sat':
            index += 6*12*24
            day_id += 6

    #
    timestep_a_week = 7*24*12
    timestep_a_day = 24*12
    time_stamp_week = np.arange(timestep_a_week).repeat(15)
    time_stamp_day = np.arange(timestep_a_day).repeat(15*7)
    time_stamp = np.sin(time_stamp_week/timestep_a_week * 2*np.pi) + np.sin(time_stamp_day/timestep_a_day * 2*np.pi)

    # forward
    for i in range(1):
        inputs = input_data[:,index + i * timestep_a_week]    # (N,T)
        inputs = torch.from_numpy(inputs).unsqueeze(0).permute(0,2,1)    # (1,T,N)
        inputs = inputs.float().to(device)
        t = torch.from_numpy(time_stamp[index]).unsqueeze(0)    # (1,T)
        t = t.float().to(device)

        target = input_data[:, int(index[-1] + (num_hour-1)*args.pred_len) :int(index[-1] + (num_hour)*args.pred_len)]
    
        # forward the model
        outputs, explain = model(inputs, t)    # return (1,N,T)

        # return (N,T)
        input_time_len = 7 * timestep_a_day 
        attn_dists = attn_cal(N=N, pred_len=args.pred_len, explain=explain, out_len_list=[12*24, 12*8], origin_timestep=input_time_len, device=device, node_idx=node_idx)

        # return it back to numpy, (N,T)
        attn_dist = attn_dists[-1]
        attn_dist = attn_dist.cpu().detach().numpy()
    
    return attn_dist

def find_important_nodes(input_data, model, target_hour, device, args, case, node_id):

    # load sensor location
    sensors = pd.read_csv(r'../data/PEMS_bay/graph_sensor_locations_bay.csv', header=None).to_numpy()
    xy = sensors[:,1:3]

    fig = plt.figure(figsize=(6*4,4*4))
    for t_hour in range(24):
        A = forward(input_data, model, t_hour, device, args, node_id, 'Mon')   # (N,T)
        
        # # extract wednesday data
        # As = np.mean(A[:,3*24*12 - t_hour*12: 4*24*12 - t_hour*12], -1)    # (N)
        # extract attention of last 24 hours
        As = np.mean(A[:,:24*12], -1)    # (N)
        
        # find top 5% nodes
        idx = np.argsort(As)
        idx_cutout = int(325 * 0.05)
        idx = idx[-idx_cutout:]
        node_attn_update = As[idx]

        print('plotting subplot', t_hour)
        ax = plt.subplot(4,6,t_hour+1)
        plt.scatter(xy[idx,1],xy[idx,0],c=node_attn_update, cmap='Reds')
        plt.plot(xy[node_id,1],xy[node_id,0], 'bo', markerfacecolor='none', label='target_node')
        plt.xlim([-122.1, -121.8])
        plt.ylim([37.225,37.45])
        plt.xticks([])
        plt.yticks([])
        ax.title.set_text('{}:00'.format(t_hour))
        plt.colorbar()
    
    plt.tight_layout()
    # fig.suptitle('important nodes of prediction of node {}'.format(node_id))
    plt.savefig(r'./case_explain/attn_last_node_{}.png'.format(node_id))






    


