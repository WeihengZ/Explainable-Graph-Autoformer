from cProfile import label
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from utils.data import trainingset_construct, load_graph_data
import time

# define training function
def train(loader, model, optimizer, criterion, device, mean, std, update_step):

    '''
    p.s. input is (batch, #node, #time_step, #feature)
         output is (batch, #node, #time_step)
    '''

    batch_loss = 0 
    for idx, (inputs, targets, tx, ty) in enumerate(tqdm(loader)):

        model.train()
        optimizer.zero_grad()

        update_step = update_step

        if inputs.dim() == 3:
            inputs = inputs.permute(0,2,1).to(device)  # (B,T,N)
        if inputs.dim() == 4:
            inputs = inputs.permute(0,2,1,3).to(device)  # (B,T,N,F)

        targets = targets.permute(0,2,1).to(device)    # (B,T,N)

        # create a mask for missing data
        mask = torch.where(targets<1, torch.zeros_like(targets), torch.ones_like(targets))

        tx = tx.to(device)    # (B,T)
        ty = ty.to(device)    # (B,T)
        t_stamp = torch.cat((tx,ty),1)
        outputs = model.forward(inputs, t_stamp)[0]    # (B,T,N)
        outputs = outputs * std + mean

        loss = criterion(mask * outputs, mask * targets) / update_step
        loss.backward()
        if idx % update_step == 0:
            optimizer.step()
            model.zero_grad()

        batch_loss += loss.detach().cpu().item()

    return batch_loss / (idx + 1)



@torch.no_grad()
def eval(loader, model, device, args, mean, std):
    # batch_rmse_loss = np.zeros(12)
    batch_mae_loss = np.zeros(args.pred_len)

    for idx, (inputs, targets, tx, ty) in enumerate(tqdm(loader)):
        model.eval()

        if inputs.dim() == 3:
            inputs = inputs.permute(0,2,1).to(device)  # (B,T,N)
        if inputs.dim() == 4:
            inputs = inputs.permute(0,2,1,3).to(device)  # (B,T,N,F)
            
        targets = targets.permute(0,2,1).to(device)  # (B,T,N)

        # create a mask for missing data
        mask = torch.where(targets<1, torch.zeros_like(targets), torch.ones_like(targets))

        tx = tx.to(device)    # (B,T)
        ty = ty.to(device)    # (B,T)
        t_stamp = torch.cat((tx,ty),1)
        outputs = model.forward(inputs, t_stamp)[0]    # (B,T,N)

        outputs = outputs * std + mean
        
        out_unnorm = (mask * outputs).detach().cpu().numpy()
        target_unnorm = (mask * targets).detach().cpu().numpy()

        mae_loss = np.zeros(args.pred_len)
        for k in range(out_unnorm.shape[1]):
            err = np.mean(np.abs(out_unnorm[:,k,:] - target_unnorm[:,k,:]))
            mae_loss[k] = err
        
        batch_mae_loss += mae_loss

    print('mae loss:', batch_mae_loss / (idx + 1))

    return batch_mae_loss / (idx + 1)


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


