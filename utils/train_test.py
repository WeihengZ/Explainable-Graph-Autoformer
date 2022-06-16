from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from utils.data import trainingset_construct, load_graph_data

# define training function
def train(loader, model, optimizer, criterion, device, args, epoch_ratio, num_forward):

    '''
    p.s. input is (batch, #node, #time_step, #feature)
         output is (batch, #node, #time_step)
    '''

    batch_loss = 0 
    for idx, (inputs, targets, tx, ty) in enumerate(tqdm(loader)):

        model.train()
        optimizer.zero_grad()

        inputs = inputs.permute(0,2,1).to(device)  # (B,T,N)
        targets = targets.permute(0,2,1).to(device)    # (B,T,N)
        tx = tx.to(device)    # (B,T)
        ty = ty.to(device)    # (B,T)
        outputs = model.forward(inputs, tx)[0]    # (B,T,N)

        # pick the predicted segment
        outputs = outputs[:, :num_forward*args.pred_len, :]
        targets = targets[:, :num_forward*args.pred_len, :]


        loss = criterion(outputs, targets) 
        loss.backward()
        optimizer.step()

        batch_loss += loss.detach().cpu().item()

    return batch_loss / (idx + 1)



@torch.no_grad()
def eval(loader, model, std, mean, device, args, num_forward):
    # batch_rmse_loss = np.zeros(12)
    batch_mae_loss = np.zeros(args.num_pred_len*args.pred_len)

    # save the results
    delays = []
    t_attns = []
    sp_attns = []

    for idx, (inputs, targets, tx, ty) in enumerate(tqdm(loader)):
        model.eval()

        inputs = (inputs).permute(0,2,1).to(device)  # (B,T,N)
        targets = targets.permute(0,2,1).to(device)  # (B,T,N)
        tx = tx.to(device)    # (B,T)
        outputs, explain_item = model.forward(inputs, tx)     # [B,T,N]

        # # for explainable analysis: model 3
        # for i in range(args.num_pred_len):
        #     delays.append(explain_item[0][i].squeeze(-2).detach().cpu().numpy())
        #     t_attns.append(explain_item[1][i].squeeze(-2).detach().cpu().numpy())
        #     sp_attns.append(explain_item[2][i].squeeze(-2).detach().cpu().numpy())

        # for model 5 explanation
        new_delays = []
        new_t_attns = []
        new_sp_attns = []
        delays = explain_item[0]
        t_attns = explain_item[1]
        sp_attns = explain_item[2]
        for i in range(len(delays)):
            new_delays.append(delays[i].detach().cpu().numpy())
            new_t_attns.append(t_attns[i].detach().cpu().numpy())
            new_sp_attns.append(sp_attns[i].detach().cpu().numpy())
        delays = new_delays
        t_attns = new_t_attns
        sp_attns = new_sp_attns

        # pick the predicted segment
        outputs = outputs[:, :num_forward*args.pred_len, :]
        targets = targets[:, :num_forward*args.pred_len, :]
        
        out_unnorm = outputs.detach().cpu().numpy()
        target_unnorm = targets.detach().cpu().numpy()

        mae_loss = np.zeros(args.num_pred_len*args.pred_len)
        for k in range(out_unnorm.shape[1]):
            err = np.mean(np.abs(out_unnorm[:,k,:] - target_unnorm[:,k,:]) * std)
            mae_loss[k] = err
        
        batch_mae_loss += mae_loss
    
    # for each node, find the most possible delay
    print(delays[0].shape)
    B,N,topk = delays[0].shape
    delay_period = np.zeros((N,args.num_pred_len,2016))
    for tt in range(len(delays)):
        data = delays[tt]
        for i in range(B):
            for j in range(N):
                for k in range(topk):
                    # each item is (B,N,top_k)
                    delay_period[j,int(tt%args.num_pred_len),data[i,j,k]] += t_attns[tt][i,j,k]
    
    # plot the major delay period for node average
    avg_period = np.mean(delay_period, 0)    # return (args.num_pred_len， 168)
    fig = plt.figure()
    for pp in range(args.num_pred_len):
        avg_period_plot = avg_period[pp] /  np.amax(avg_period[pp])
        x = np.arange(np.size(avg_period_plot))
        plt.vlines(x=np.array([0,24,48,72,96,120,144,168]) * 12, ymin=0, ymax=1.2, colors='k', linestyles='dashed')
        plt.plot(x, avg_period_plot, label='attnetion distribution for future {} hours'.format(pp+1))
        plt.xticks(ticks=np.array([0,24,48,72,96,120,144,168]) * 12, labels=[0,24,48,72,96,120,144,168])
        plt.text(x=6*12, y=1.1, s='Sun', c='red')
        plt.text(x=30*12, y=1.1, s='Mon', c='red')
        plt.text(x=54*12, y=1.1, s='Tue', c='red')
        plt.text(x=78*12, y=1.1, s='Wed', c='red')
        plt.text(x=102*12, y=1.1, s='Thr', c='red')
        plt.text(x=126*12, y=1.1, s='Fri', c='red')
        plt.text(x=150*12, y=1.1, s='Sat', c='red')
        plt.xlabel('Hours')
        plt.ylabel('Attention score')

        plt.legend(loc=0)
    plt.savefig(r'./explain/average_period_{}'.format(args.num_pred_len))

    # plot temporal without classifying the temporal steps
    avg_period = np.mean(np.mean(delay_period, 0),0)    # return (168)
    fig = plt.figure()
    avg_period_plot = avg_period /  np.amax(avg_period)
    x = np.arange(np.size(avg_period_plot))
    plt.vlines(x=np.array([0,24,48,72,96,120,144,168]) * 12, ymin=0, ymax=1.2, colors='k', linestyles='dashed')
    plt.plot(x, avg_period_plot, label='attnetion distribution for future {} hours'.format(pp+1))
    plt.xticks(ticks=np.array([0,24,48,72,96,120,144,168]) * 12, labels=[0,24,48,72,96,120,144,168])
    plt.text(x=6*12, y=1.1, s='Sun', c='red')
    plt.text(x=30*12, y=1.1, s='Mon', c='red')
    plt.text(x=54*12, y=1.1, s='Tue', c='red')
    plt.text(x=78*12, y=1.1, s='Wed', c='red')
    plt.text(x=102*12, y=1.1, s='Thr', c='red')
    plt.text(x=126*12, y=1.1, s='Fri', c='red')
    plt.text(x=150*12, y=1.1, s='Sat', c='red')
    
    plt.xlabel('Hours')
    plt.ylabel('Attention score')
    plt.legend(loc=0)
    plt.savefig(r'./explain/average_period_all_{}'.format(args.num_pred_len))

    # pick node 40 and 307 to observe spatial attn score
    node_id = 40
    num_nodes = sp_attns[0].shape[1]
    rr = np.zeros((num_nodes, num_nodes))
    for i in range(len(sp_attns)):
        rr += np.mean(sp_attns[i],0)
    rr /= len(sp_attns)
    node_attn = rr[node_id]    # return (num_node)

    # load sensor location
    sensors = pd.read_csv(r'../data/PEMS_bay/graph_sensor_locations_bay.csv', header=None).to_numpy()
    xy = sensors[:,1:3]

    # load adj matrix
    sensor_ids, sensor_id_to_ind, W = load_graph_data(r'../data/PEMS_bay/adj_mx_bay.pkl')
    node_adj = W[node_id]

    fig = plt.figure()
    plt.scatter(xy[:,1],xy[:,0],c=node_attn*node_adj, cmap='Reds')
    plt.plot(xy[node_id,1],xy[node_id,0], 'bo', markerfacecolor='none', label='target_node')
    plt.colorbar()
    plt.title('spatial attention distribution of node {}'.format(node_id))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=0)
    plt.savefig(r'./explain/{}_node_{}_error_dist.png'.format(args.num_pred_len, node_id))

    node_id = 307
    num_nodes = sp_attns[0].shape[1]
    rr = np.zeros((num_nodes, num_nodes))
    for i in range(len(sp_attns)):
        rr += np.mean(sp_attns[i],0)
    rr /= len(sp_attns)
    node_attn = rr[node_id]    # return (num_node)

    node_adj = W[node_id]

    fig = plt.figure()
    plt.scatter(xy[:,1],xy[:,0],c=node_attn*node_adj, cmap='Reds')
    plt.plot(xy[node_id,1],xy[node_id,0], 'bo', markerfacecolor='none', label='target_node')
    plt.colorbar()
    plt.title('spatial attention distribution of node {}'.format(node_id))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=0)
    plt.savefig(r'./explain/{}_node_{}_error_dist.png'.format(args.num_pred_len, node_id))






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


