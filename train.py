import os
import sys
import math
import argparse
import numpy as np
import torch
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import itertools
from model import *

which_gpu = "7"
os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu

# random.seed(33)
# np.random.seed(33)
# torch.manual_seed(33)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(33)
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

def train_test_split(data):
    X_aqi_train = data[:436]
    X_aqi_val = data[436:468]
    X_aqi_test = data[468:500]
    return X_aqi_train, X_aqi_val, X_aqi_test

def load_X_batch(X_aqi_train,X_meo_train,X_aqi_ex_train,X_meo_ex_train):
    X_aqi_batch = X_aqi_train[i*batch_size:i*batch_size+batch_size].cuda() # (B, T_in, N_aqi, F)
    X_meo_batch = X_meo_train[i*batch_size:i*batch_size+batch_size].cuda()
    X_aqi_ex_batch = X_aqi_ex_train[i*batch_size:i*batch_size+batch_size].cuda()
    X_meo_ex_batch = X_meo_ex_train[i*batch_size:i*batch_size+batch_size].cuda()
#     print(X_aqi_batch.shape,X_meo_batch.shape,X_aqi_ex_batch.shape,X_meo_ex_batch.shape)
    return (X_aqi_batch,X_meo_batch,X_aqi_ex_batch,X_meo_ex_batch)

def load_y_batch(y_aqi_train,y_meo_train,y_nwp_train,y_aqi_mask_train,y_meo_mask_train,y_aqi_ex_train,y_meo_ex_train):
    y_aqi_batch = y_aqi_train[i*batch_size:i*batch_size+batch_size].cuda() # (B, N_aqi, 48, F)
    y_meo_batch = y_meo_train[i*batch_size:i*batch_size+batch_size].cuda()
    y_nwp_batch = y_nwp_train[i*batch_size:i*batch_size+batch_size].cuda()
    y_aqi_mask_batch = y_aqi_mask_train[i*batch_size:i*batch_size+batch_size].cuda() # (B, N_aqi, 48, F)
    y_meo_mask_batch = y_meo_mask_train[i*batch_size:i*batch_size+batch_size].cuda()
    y_aqi_ex_batch = y_aqi_ex_train[i*batch_size:i*batch_size+batch_size].cuda()
    y_meo_ex_batch = y_meo_ex_train[i*batch_size:i*batch_size+batch_size].cuda()
    return (y_aqi_batch,y_meo_batch,y_nwp_batch,y_aqi_mask_batch,y_meo_mask_batch,y_aqi_ex_batch,y_meo_ex_batch)

def loss_mse(y_pred, y_true, y_mask):
    y_mse = torch.sub(y_pred, y_true)
    y_mse = torch.mul(y_mse, y_mse)
    y_mse = torch.mul(y_mse, y_mask)
    y_mse = torch.div(torch.sum(y_mse), torch.sum(y_mask))
    return y_mse

def mae(pred, label, mask):
    mae = np.sum(abs(label-pred)*mask)
    num = np.sum(mask)
    mae = mae/num
    return mae

def smape(pred, label, mask):
    smape = np.sum(2.0*(np.abs(pred - label) / (np.abs(pred) + np.abs(label)))*mask)
    num = np.sum(mask)
    smape = smape/num
    return smape

def get_max(x, y, z):
    if x > y:
        max_value = x
    else:
        max_value = y
        
    if max_value < z:
        max_value = z
    else:
        pass 
    return max_value

def softmax(m_weight, t_weight, s_weight):
    max_weight = get_max(m_weight, t_weight, s_weight)
    m_weight = m_weight-max_weight
    t_weight = t_weight-max_weight
    s_weight = s_weight-max_weight
    total = torch.exp(m_weight)+torch.exp(t_weight)+torch.exp(s_weight)
    m_weight = torch.exp(m_weight)/total
    t_weight = torch.exp(t_weight)/total
    s_weight = torch.exp(s_weight)/total
    return m_weight, t_weight, s_weight

# loss.detach().cpu().numpy()

N_aqi = 35 # the number of air quality stations
N_meo = 18 # the number of weather stations

# N_aqi = 76
# N_meo = 11

alpha = 0.2 # alpha for the leaky_relu
epochs = 100
hid_dim = 32
patience = 5
dropout = 0.5
batch_size = 32
epsilon = 10000
learning_rate = 0.0001

if __name__ == '__main__':
    
    # load data
    X_aqi = np.load("data_sample/X_aqi_sample.npy") # (N, T_in, N_aqi, F)
    y_aqi = np.load("data_sample/y_aqi_sample.npy") # (N, T_out, N_aqi, F)
    y_aqi_mask = np.load("data_sample/y_aqi_mask_sample.npy") # (N, T_out, N_aqi, F)
    X_meo = np.load("data_sample/X_meo_sample.npy")
    y_meo = np.load("data_sample/y_meo_sample.npy")
    y_nwp = np.load("data_sample/y_nwp_sample.npy")
    y_meo_mask = np.load("data_sample/y_meo_mask_sample.npy")
    context_feat = np.load("data_sample/context_feat.npy") # contextual features
    adj = np.load("data_sample/adj.npy") # (N, N)
    
    print('Using sample data...')
    
    # calculate the sparse adjacency matrix
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] < epsilon:
                adj[i, j] = 1
            else:
                adj[i, j] = 0
                
    adj_norm = adj / 10000

    adj = torch.FloatTensor(adj).cuda()
    adj_norm = torch.FloatTensor(adj_norm).cuda()
    
#     y_aqi = y_aqi.transpose((0, 2, 1, 3)) # (N, N_aqi, T_out, F)
#     y_aqi_mask = y_aqi_mask.transpose((0, 2, 1, 3)) # (N, N_aqi, T_out, F)
#     print('after transpose: ', y_aqi.shape)
#     y_meo = y_meo.transpose((0, 2, 1, 3))
#     y_meo_mask = y_meo_mask.transpose((0, 2, 1, 3))
    
    # train test split
    X_aqi_train, X_aqi_val, X_aqi_test = train_test_split(X_aqi)
    y_aqi_train, y_aqi_val, y_aqi_test = train_test_split(y_aqi)
    y_aqi_mask_train, y_aqi_mask_val, y_aqi_mask_test = train_test_split(y_aqi_mask)
    
    X_meo_train, X_meo_val, X_meo_test = train_test_split(X_meo)
    y_meo_train, y_meo_val, y_meo_test = train_test_split(y_meo)
    y_nwp_train, y_nwp_val, y_nwp_test = train_test_split(y_nwp)
    y_meo_mask_train, y_meo_mask_val, y_meo_mask_test = train_test_split(y_meo_mask)
    
    # preprocess data
    X_aqi_ex_train = torch.LongTensor(X_aqi_train[:, :, :, 6:]).cuda()
    X_aqi_ex_val = torch.LongTensor(X_aqi_val[:, :, :, 6:]).cuda()
    X_aqi_ex_test = torch.LongTensor(X_aqi_test[:, :, :, 6:]).cuda()
    y_aqi_ex_train = torch.LongTensor(y_aqi_train[:, :, :, 6:]).cuda()
    y_aqi_ex_val = torch.LongTensor(y_aqi_val[:, :, :, 6:]).cuda()
    y_aqi_ex_test = torch.LongTensor(y_aqi_test[:, :, :, 6:]).cuda()
    X_aqi_train = torch.FloatTensor(X_aqi_train[:, :, :, :6]).cuda()
    X_aqi_val = torch.FloatTensor(X_aqi_val[:, :, :, :6]).cuda()
    X_aqi_test = torch.FloatTensor(X_aqi_test[:, :, :, :6]).cuda()
    
    X_meo_ex_train = torch.LongTensor(X_meo_train[:, :, :, 4:]).cuda()
    X_meo_ex_val = torch.LongTensor(X_meo_val[:, :, :, 4:]).cuda()
    X_meo_ex_test = torch.LongTensor(X_meo_test[:, :, :, 4:]).cuda()
    y_meo_ex_train = torch.LongTensor(y_meo_train[:, :, :, 4:]).cuda()
    y_meo_ex_val = torch.LongTensor(y_meo_val[:, :, :, 4:]).cuda()
    y_meo_ex_test = torch.LongTensor(y_meo_test[:, :, :, 4:]).cuda()
    X_meo_train = torch.FloatTensor(X_meo_train[:, :, :, :4]).cuda()
    X_meo_val = torch.FloatTensor(X_meo_val[:, :, :, :4]).cuda()
    X_meo_test = torch.FloatTensor(X_meo_test[:, :, :, :4]).cuda()

    y_aqi_train = torch.FloatTensor(y_aqi_train).cuda()
    y_aqi_val = torch.FloatTensor(y_aqi_val).cuda()
    y_aqi_test = torch.FloatTensor(y_aqi_test).cuda()
    y_meo_train = torch.FloatTensor(y_meo_train).cuda()
    y_meo_val = torch.FloatTensor(y_meo_val).cuda()
    y_meo_test = torch.FloatTensor(y_meo_test).cuda()
    y_nwp_train = torch.FloatTensor(y_nwp_train).cuda()
    y_nwp_val = torch.FloatTensor(y_nwp_val).cuda()
    y_nwp_test = torch.FloatTensor(y_nwp_test).cuda()
    y_aqi_mask_train = torch.FloatTensor(y_aqi_mask_train).cuda()
    y_aqi_mask_val = torch.FloatTensor(y_aqi_mask_val).cuda()
    y_aqi_mask_test = torch.FloatTensor(y_aqi_mask_test).cuda()
    y_meo_mask_train = torch.FloatTensor(y_meo_mask_train).cuda()
    y_meo_mask_val = torch.FloatTensor(y_meo_mask_val).cuda()
    y_meo_mask_test = torch.FloatTensor(y_meo_mask_test).cuda()
    
    context_feat = torch.FloatTensor(context_feat).cuda()
    
    num_iter_train = int(X_aqi_train.shape[0] / batch_size)
#     num_iter_val = int(X_aqi_val.shape[0] / batch_size)
#     num_iter_test = int(X_aqi_test.shape[0] / batch_size)
    
    # MasterGNN
    generator = Generator(dropout=0.5, alpha=0.2, hid_dim=hid_dim, t_out=48)
    dm = macroDiscriminator()
    ds = spatialDiscriminator()
    dt = temporalDiscriminator()
    generator.cuda()
    dm.cuda()
    ds.cuda()
    dt.cuda()
    
#     optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
#     optimizer_dm = torch.optim.Adam(dm.parameters(), lr=learning_rate)
#     optimizer_ds = torch.optim.Adam(ds.parameters(), lr=learning_rate)
#     optimizer_dt = torch.optim.Adam(dt.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.Adam(itertools.chain(dm.parameters(), dt.parameters(), ds.parameters()), lr=learning_rate*3)
    
    criterion = nn.BCELoss()

    best_epoch = 0
    min_mae = 1e15
    st_epoch = best_epoch
    for epoch in range(st_epoch, epochs):
        st_time = time.time()
        
        # shuffle data
        shuffle_index = np.random.permutation(X_aqi_train.shape[0])
        X_aqi_train = X_aqi_train[shuffle_index]
        X_meo_train = X_meo_train[shuffle_index]
        X_aqi_ex_train = X_aqi_ex_train[shuffle_index]
        X_meo_ex_train = X_meo_ex_train[shuffle_index]
        y_aqi_train = y_aqi_train[shuffle_index]
        y_meo_train = y_meo_train[shuffle_index]
        y_nwp_train = y_nwp_train[shuffle_index]
        y_aqi_ex_train = y_aqi_ex_train[shuffle_index]
        y_meo_ex_train = y_meo_ex_train[shuffle_index]
        y_aqi_mask_train = y_aqi_mask_train[shuffle_index]
        y_meo_mask_train = y_meo_mask_train[shuffle_index]
        
        # training
        print('training......')
        cnt = 0
        train_discloss = 0
        train_aqiloss,train_temploss,train_humiloss,train_windloss = 0,0,0,0
        train_aqimae,train_tempmae,train_humimae,train_windmae = 0,0,0,0
        train_aqismape,train_tempsmape,train_humismape,train_windsmape = 0,0,0,0
        train_gdloss,train_gmloss,train_gtloss,train_gsloss = 0,0,0,0
        for i in range(num_iter_train):
            X_batch = load_X_batch(X_aqi_train,X_meo_train,X_aqi_ex_train,X_meo_ex_train)
            y_batch = load_y_batch(y_aqi_train,y_meo_train,y_nwp_train, \
                                   y_aqi_mask_train,y_meo_mask_train,y_aqi_ex_train,y_meo_ex_train)
            
            out_aqi,out_temp,out_humi,out_wind = generator(X_batch,y_batch,context_feat,adj,adj_norm)
            
            X_aqi, X_meo, X_aqi_ex, X_meo_ex = X_batch
            y_aqi, y_meo, y_nwp, y_aqi_mask, y_meo_mask, y_aqi_ex, y_meo_ex = y_batch

            # input of micro discriminator
            X_aqi_true = X_aqi[:, :, :, 0] # (B, 72, N_aqi)
            X_meo_true = torch.cat((X_meo[:,:,:,0],X_meo[:,:,:,1],X_meo[:,:,:,2]), axis=2) # (B, 72, 3*N_meo)
            input_real = torch.cat((X_aqi_true,X_meo_true),axis=2) # (B, 72, N_aqi+3*N_meo)
            out_real = torch.cat((y_aqi[:,:,:,0],y_meo[:,:,:,0],y_meo[:,:,:,1],y_meo[:,:,:,2]),axis=2) # (B, 48, N_aqi+3*N_meo)
            out_fake = torch.cat((out_aqi.detach(),out_temp.detach(),out_humi.detach(),out_wind.detach()),axis=1)
            out_fake = torch.transpose(out_fake, 1, 2) # (B, 48, N_aqi+3*N_meo)
            mdisc_real = torch.cat((input_real, out_real), axis=1) # (B, 120, N_aqi+3*N_meo)
            mdisc_fake = torch.cat((input_real, out_fake), axis=1) # (B, 120, N_aqi+3*N_meo)
            
            # input of temporal discriminator
            tdisc_real_aqi = mdisc_real[:,:,:N_aqi] # (B, 120, N_aqi)
            tdisc_real_meo = mdisc_real[:,:,N_aqi:] # (B, 120, 3*N_meo)
            tdisc_fake_aqi = mdisc_fake[:,:,:N_aqi] # (B, 120, N_aqi)
            tdisc_fake_meo = mdisc_fake[:,:,N_aqi:] # (B, 120, 3*N_meo)
            
            # input of spatial discriminator
            sdisc_fake_aqi = torch.transpose(out_aqi.detach(), 1, 2).unsqueeze(3) # (B, T_out, N_aqi, 1)
            sdisc_fake_temp = torch.transpose(out_temp.detach(), 1, 2).unsqueeze(3) # (B, T_out, N_meo, 1)
            sdisc_fake_humi = torch.transpose(out_humi.detach(), 1, 2).unsqueeze(3) # (B, T_out, N_meo, 1)
            sdisc_fake_wind = torch.transpose(out_wind.detach(), 1, 2).unsqueeze(3) # (B, T_out, N_meo, 1)
            sdisc_fake_meo = torch.cat((sdisc_fake_temp,sdisc_fake_humi,sdisc_fake_wind),axis=3) # (B, T_out, N_meo, 3)
            sdisc_real_aqi = y_aqi[:, :, :, 0].unsqueeze(3) # (B, T_out, N_aqi, 1)
            sdisc_real_meo = y_meo[:, :, :, :3] # (B, T_out, N_meo, 3)

            real_mlabel = Variable(torch.ones(batch_size,1)).cuda() # define the real sample label
            fake_mlabel = Variable(torch.zeros(batch_size,1)).cuda() # define the fake sample label
            real_mout, mmetric_real = dm(mdisc_real)
            fake_mout, mmetric_fake = dm(mdisc_fake)
            dm_loss_mreal = criterion(real_mout, real_mlabel)
            dm_loss_mfake = criterion(fake_mout, fake_mlabel)
            dm_loss = dm_loss_mreal + dm_loss_mfake

            real_tlabel = Variable(torch.ones(batch_size*(N_aqi+N_meo),1)).cuda() # define the real sample label
            fake_tlabel = Variable(torch.zeros(batch_size*(N_aqi+N_meo),1)).cuda() # define the fake sample label
            real_tout, tmetric_real = dt(tdisc_real_aqi, tdisc_real_meo)
            fake_tout, tmetric_fake = dt(tdisc_fake_aqi, tdisc_fake_meo)
            dt_loss_treal = criterion(real_tout, real_tlabel)
            dt_loss_tfake = criterion(fake_tout, fake_tlabel)
            dt_loss = dt_loss_treal + dt_loss_tfake
            
            real_slabel = Variable(torch.ones(batch_size*48,1)).cuda() # define the real sample label
            fake_slabel = Variable(torch.zeros(batch_size*48,1)).cuda() # define the fake sample label
            real_sout, smetric_real = ds(sdisc_real_aqi, sdisc_real_meo, adj, adj_norm)
            fake_sout, smetric_fake = ds(sdisc_fake_aqi, sdisc_fake_meo, adj, adj_norm)
            ds_loss_sreal = criterion(real_sout, real_slabel)
            ds_loss_sfake = criterion(fake_sout, fake_slabel)
            ds_loss = ds_loss_sreal + ds_loss_sfake
            
#             m_weight = torch.mul(mmetric_real.detach(), mmetric_fake.detach())
#             m_weight = torch.sum(m_weight, 1)
            m_weight = torch.sub(mmetric_real.detach(), mmetric_fake.detach())
            m_weight = torch.mul(m_weight, m_weight)
            m_weight = torch.sum(m_weight, 1)
            m_weight = torch.mean(m_weight)
#             t_weight = torch.mul(tmetric_real.detach(), tmetric_fake.detach())
#             t_weight = torch.sum(t_weight, 2)
            t_weight = torch.sub(tmetric_real.detach(), tmetric_fake.detach())
            t_weight = torch.mul(t_weight, t_weight)
            t_weight = torch.sum(t_weight, 2)
            t_weight = torch.mean(t_weight)
#             s_weight = torch.mul(smetric_real.detach(), smetric_fake.detach())
#             s_weight = torch.sum(s_weight, 2)
            s_weight = torch.sub(smetric_real.detach(), smetric_fake.detach())
            s_weight = torch.mul(s_weight, s_weight)
            s_weight = torch.sum(s_weight, 1)
            s_weight = torch.mean(s_weight)
            
            d_loss = dm_loss+dt_loss+ds_loss
            
            train_discloss += d_loss.item()
            
            # generator loss
            gm_loss = criterion(fake_mout, real_mlabel)
            gt_loss = criterion(fake_tout, real_tlabel)
            gs_loss = criterion(fake_sout, real_slabel)
            
            y_aqi = torch.transpose(y_aqi, 1, 2) # (B, N, T, F)
            y_aqi_mask = torch.transpose(y_aqi_mask, 1, 2) # (B, N, T, F)
            y_meo = torch.transpose(y_meo, 1, 2) # (B, N, T, F)
            y_meo_mask = torch.transpose(y_meo_mask, 1, 2) # (B, N, T, F)

            B = out_aqi.shape[0]
            out_aqi = out_aqi*59.02+57.23
            label_aqi = y_aqi[:, :, :, 0].contiguous()*59.02+57.23
            loss_aqi = loss_mse(out_aqi, label_aqi, y_aqi_mask[:,:,:,0])

            out_temp = out_temp*12.08+11.52
            label_temp = y_meo[:, :, :, 0].contiguous()*12.08+11.52
            loss_temp = loss_mse(out_temp, label_temp, y_meo_mask[:,:,:,0])

            out_humi = out_humi*25.69+48.75
            label_humi = y_meo[:, :, :, 1].contiguous()*25.69+48.75
            loss_humi = loss_mse(out_humi, label_humi, y_meo_mask[:,:,:,1])

            out_wind = out_wind*1.39+1.91
            label_wind = y_meo[:, :, :, 2].contiguous()*1.39+1.91
            loss_wind = loss_mse(out_wind, label_wind, y_meo_mask[:,:,:,2])
            
#             label_aqi = y_aqi[:, :, :, 0].contiguous()
#             loss_aqi = loss_mse(out_aqi, label_aqi, y_aqi_mask[:,:,:,0])

#             label_temp = y_meo[:, :, :, 0].contiguous()
#             loss_temp = loss_mse(out_temp, label_temp, y_meo_mask[:,:,:,0])

#             label_humi = y_meo[:, :, :, 1].contiguous()
#             loss_humi = loss_mse(out_humi, label_humi, y_meo_mask[:,:,:,1])

#             label_wind = y_meo[:, :, :, 2].contiguous()
#             loss_wind = loss_mse(out_wind, label_wind, y_meo_mask[:,:,:,2])   
    
            m_weight, t_weight, s_weight = softmax(m_weight, t_weight, s_weight)
            gloss = loss_aqi+loss_temp+loss_humi+loss_wind
            gdloss = m_weight*gm_loss+t_weight*gt_loss+s_weight*gs_loss
            loss_generator = gloss+gdloss
            
            train_aqiloss += loss_aqi.item()
            train_temploss += loss_temp.item()
            train_humiloss += loss_humi.item()
            train_windloss += loss_wind.item()
            train_gdloss += gdloss.item()
            train_gmloss += gm_loss.item()
            train_gtloss += gt_loss.item()
            train_gsloss += gs_loss.item()
            
            train_aqimae += mae(out_aqi.detach().cpu().numpy(), label_aqi.detach().cpu().numpy(), \
                                y_aqi_mask[:,:,:,0].cpu().numpy())
            train_tempmae += mae(out_temp.detach().cpu().numpy(),label_temp.detach().cpu().numpy(), \
                                 y_meo_mask[:,:,:,0].cpu().numpy())
            train_humimae += mae(out_humi.detach().cpu().numpy(),label_humi.detach().cpu().numpy(), \
                                 y_meo_mask[:,:,:,1].cpu().numpy())
            train_windmae += mae(out_wind.detach().cpu().numpy(),label_wind.detach().cpu().numpy(), \
                                 y_meo_mask[:,:,:,2].cpu().numpy())
            train_aqismape += smape(out_aqi.detach().cpu().numpy(),label_aqi.detach().cpu().numpy(), \
                                    y_aqi_mask[:,:,:,0].cpu().numpy())
            train_tempsmape += smape(out_temp.detach().cpu().numpy(),label_temp.detach().cpu().numpy(), \
                                     y_meo_mask[:,:,:,0].cpu().numpy())
            train_humismape += smape(out_humi.detach().cpu().numpy(),label_humi.detach().cpu().numpy(), \
                                     y_meo_mask[:,:,:,1].cpu().numpy())
            train_windsmape += smape(out_wind.detach().cpu().numpy(),label_wind.detach().cpu().numpy(), \
                                     y_meo_mask[:,:,:,2].cpu().numpy())
            
            #  Train Discriminator and Generator
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
            
            d_loss.backward(retain_graph=True)
            loss_generator.backward()
            
            optimizer_d.step()
            optimizer_g.step()
            
            #  Train Discriminator
#             optimizer_d.zero_grad()
#             d_loss.backward(retain_graph=True)
#             optimizer_d.step()
            
            if cnt%100 == 99:
                print('Epoch: {:04d}'.format(epoch+1),
                      'Step: {:06d}'.format(cnt+1),
                      'discriminator loss: {:.4f}'.format(train_discloss/100)
                     )
                train_discloss = 0

            #  Train Generator
#             optimizer_g.zero_grad()
#             loss_generator.backward()
#             optimizer_g.step()

            if cnt%5 == 1:
                print('Epoch: {:04d}'.format(epoch+1),
                      'Step: {:06d}'.format(cnt+1),
                      'aqi mae: {:.4f}'.format(train_aqimae/5),
                      'temp mae: {:.4f}'.format(train_tempmae/5),
                      'humi mae: {:.4f}'.format(train_humimae/5),
                      'wind mae: {:.4f}'.format(train_windmae/5),
                      'aqi smape: {:.4f}'.format(train_aqismape/5),
                      'temp smape: {:.4f}'.format(train_tempsmape/5),
                      'humi smape: {:.4f}'.format(train_humismape/5),
                      'wind smape: {:.4f}'.format(train_windsmape/5),
                      'dloss: {:.4f}'.format(train_gdloss/5),
                      'mloss: {:.4f}'.format(train_gmloss/5),
                      'tloss: {:.4f}'.format(train_gtloss/5),
                      'sloss: {:.4f}'.format(train_gsloss/5)
                     )
                train_aqiloss,train_temploss,train_humiloss,train_windloss = 0,0,0,0
                train_aqimae,train_tempmae,train_humimae,train_windmae = 0,0,0,0
                train_aqismape,train_tempsmape,train_humismape,train_windsmape = 0,0,0,0
                train_gdloss,train_gmloss,train_gtloss,train_gsloss = 0,0,0,0
            cnt += 1
        
#         torch.save(generator, 'model/mastergnn.pkl')
        
        # validating
        cnt = 0
        val_aqimae,val_tempmae,val_humimae,val_windmae = 0,0,0,0
        val_aqismape,val_tempsmape,val_humismape,val_windsmape = 0,0,0,0
        with torch.no_grad():
            i = 0
            print('validating......')
#             for i in range(num_iter_val):
            X_batch = load_X_batch(X_aqi_val,X_meo_val,X_aqi_ex_val,X_meo_ex_val)
            y_batch = load_y_batch(y_aqi_val,y_meo_val,y_nwp_val,y_aqi_mask_val,y_meo_mask_val,y_aqi_ex_val,y_meo_ex_val)

            out_aqi,out_temp,out_humi,out_wind = generator(X_batch,y_batch,context_feat,adj,adj_norm)

            X_aqi, X_meo, X_aqi_ex, X_meo_ex = X_batch
            y_aqi, y_meo, y_nwp, y_aqi_mask, y_meo_mask, y_aqi_ex, y_meo_ex = y_batch

            y_aqi = torch.transpose(y_aqi, 1, 2) # (B, N, T, F)
            y_aqi_mask = torch.transpose(y_aqi_mask, 1, 2) # (B, N, T, F)
            y_meo = torch.transpose(y_meo, 1, 2) # (B, N, T, F)
            y_meo_mask = torch.transpose(y_meo_mask, 1, 2) # (B, N, T, F)

            B = out_aqi.shape[0]
            out_aqi = out_aqi*59.02+57.23
            label_aqi = y_aqi[:, :, :, 0].contiguous()*59.02+57.23

            out_temp = out_temp*12.08+11.52
            label_temp = y_meo[:, :, :, 0].contiguous()*12.08+11.52

            out_humi = out_humi*25.69+48.75
            label_humi = y_meo[:, :, :, 1].contiguous()*25.69+48.75

            out_wind = out_wind*1.39+1.91
            label_wind = y_meo[:, :, :, 2].contiguous()*1.39+1.91

            val_aqimae += mae(out_aqi.detach().cpu().numpy(), label_aqi.detach().cpu().numpy(), \
                              y_aqi_mask[:,:,:,0].cpu().numpy())
            val_tempmae += mae(out_temp.detach().cpu().numpy(),label_temp.detach().cpu().numpy(), \
                               y_meo_mask[:,:,:,0].cpu().numpy())
            val_humimae += mae(out_humi.detach().cpu().numpy(),label_humi.detach().cpu().numpy(), \
                               y_meo_mask[:,:,:,1].cpu().numpy())
            val_windmae += mae(out_wind.detach().cpu().numpy(),label_wind.detach().cpu().numpy(), \
                               y_meo_mask[:,:,:,2].cpu().numpy())
            val_aqismape += smape(out_aqi.detach().cpu().numpy(),label_aqi.detach().cpu().numpy(), \
                                  y_aqi_mask[:,:,:,0].cpu().numpy())
            val_tempsmape += smape(out_temp.detach().cpu().numpy(),label_temp.detach().cpu().numpy(), \
                                   y_meo_mask[:,:,:,0].cpu().numpy())
            val_humismape += smape(out_humi.detach().cpu().numpy(),label_humi.detach().cpu().numpy(), \
                                   y_meo_mask[:,:,:,1].cpu().numpy())
            val_windsmape += smape(out_wind.detach().cpu().numpy(),label_wind.detach().cpu().numpy(), \
                                   y_meo_mask[:,:,:,2].cpu().numpy())
            cnt += 1
        val_aqimae = val_aqimae/cnt
        val_tempmae = val_tempmae/cnt
        val_humimae = val_humimae/cnt
        val_windmae = val_windmae/cnt
        val_aqismape = val_aqismape/cnt
        val_tempsmape = val_tempsmape/cnt
        val_humismape = val_humismape/cnt
        val_windsmape = val_windsmape/cnt

        # testing
        cnt = 0
        test_aqimae,test_tempmae,test_humimae,test_windmae = 0,0,0,0
        test_aqismape,test_tempsmape,test_humismape,test_windsmape = 0,0,0,0
        with torch.no_grad():
            i = 0
            print('testing......')
#             for i in range(num_iter_test):
            X_batch = load_X_batch(X_aqi_test,X_meo_test,X_aqi_ex_test,X_meo_ex_test)
            y_batch = load_y_batch(y_aqi_test,y_meo_test,y_nwp_test, \
                                   y_aqi_mask_test,y_meo_mask_test,y_aqi_ex_test,y_meo_ex_test)

            out_aqi,out_temp,out_humi,out_wind = generator(X_batch,y_batch,context_feat,adj,adj_norm)

            X_aqi, X_meo, X_aqi_ex, X_meo_ex = X_batch
            y_aqi, y_meo, y_nwp, y_aqi_mask, y_meo_mask, y_aqi_ex, y_meo_ex = y_batch

            y_aqi = torch.transpose(y_aqi, 1, 2) # (B, N, T, F)
            y_aqi_mask = torch.transpose(y_aqi_mask, 1, 2) # (B, N, T, F)
            y_meo = torch.transpose(y_meo, 1, 2) # (B, N, T, F)
            y_meo_mask = torch.transpose(y_meo_mask, 1, 2) # (B, N, T, F)

            B = out_aqi.shape[0]
            out_aqi = out_aqi*59.02+57.23
            label_aqi = y_aqi[:, :, :, 0].contiguous()*59.02+57.23

            out_temp = out_temp*12.08+11.52
            label_temp = y_meo[:, :, :, 0].contiguous()*12.08+11.52

            out_humi = out_humi*25.69+48.75
            label_humi = y_meo[:, :, :, 1].contiguous()*25.69+48.75

            out_wind = out_wind*1.39+1.91
            label_wind = y_meo[:, :, :, 2].contiguous()*1.39+1.91

            test_aqimae += mae(out_aqi.detach().cpu().numpy(), label_aqi.detach().cpu().numpy(), \
                              y_aqi_mask[:,:,:,0].cpu().numpy())
            test_tempmae += mae(out_temp.detach().cpu().numpy(),label_temp.detach().cpu().numpy(), \
                               y_meo_mask[:,:,:,0].cpu().numpy())
            test_humimae += mae(out_humi.detach().cpu().numpy(),label_humi.detach().cpu().numpy(), \
                               y_meo_mask[:,:,:,1].cpu().numpy())
            test_windmae += mae(out_wind.detach().cpu().numpy(),label_wind.detach().cpu().numpy(), \
                               y_meo_mask[:,:,:,2].cpu().numpy())
            test_aqismape += smape(out_aqi.detach().cpu().numpy(),label_aqi.detach().cpu().numpy(), \
                                  y_aqi_mask[:,:,:,0].cpu().numpy())
            test_tempsmape += smape(out_temp.detach().cpu().numpy(),label_temp.detach().cpu().numpy(), \
                                   y_meo_mask[:,:,:,0].cpu().numpy())
            test_humismape += smape(out_humi.detach().cpu().numpy(),label_humi.detach().cpu().numpy(), \
                                   y_meo_mask[:,:,:,1].cpu().numpy())
            test_windsmape += smape(out_wind.detach().cpu().numpy(),label_wind.detach().cpu().numpy(), \
                                   y_meo_mask[:,:,:,2].cpu().numpy())
            cnt += 1
        test_aqimae = test_aqimae/cnt
        test_tempmae = test_tempmae/cnt
        test_humimae = test_humimae/cnt
        test_windmae = test_windmae/cnt
        test_aqismape = test_aqismape/cnt
        test_tempsmape = test_tempsmape/cnt
        test_humismape = test_humismape/cnt
        test_windsmape = test_windsmape/cnt

#         if(val_mae < min_mae):
#             min_mae = val_mae
#             best_epoch = epoch + 1
#             best_mae = test_mae.copy()
#             best_mse = test_mse.copy()
#             best_loss = test_loss.copy()
       
        print("Epoch: {}".format(epoch+1))
#         print("Train loss: {}".format(loss_train))
        print('Validation metric','aqi_mae',val_aqimae,'temp_mae',val_tempmae,'humi_mae',val_humimae,'wind_mae',val_windmae, \
              'aqi_smape',val_aqismape,'temp_smape',val_tempsmape,'humi_smape',val_humismape,'wind_smape',val_windsmape)
        print('Testing metric','aqi_mae',test_aqimae,'temp_mae',test_tempmae,'humi_mae',test_humimae,'wind_mae',test_windmae, \
              'aqi_smape',test_aqismape,'temp_smape',test_tempsmape,'humi_smape',test_humismape,'wind_smape',test_windsmape)
#         print(best_mae, best_mse, best_loss, 'Best Epoch-{}'.format(best_epoch))
        print('time: {:.4f}s'.format(time.time() - st_time))

        # early stop
#         if(epoch+1 - best_epoch >= patience):
#             sys.exit(0)
