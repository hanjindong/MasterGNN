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

N_aqi = 35 # the number of air quality stations
N_meo = 18 # the number of weather stations

class HeterGraphAttentionLayer(nn.Module):
    """
    context-aware heterogeneous graph attention layer.
    """
    def __init__(self, in_aqi_features, in_meo_features, out_features, dropout, alpha):
        super(HeterGraphAttentionLayer, self).__init__()
        self.in_aqi_features = in_aqi_features+8   # input dim
        self.in_meo_features = in_meo_features+10   # input dim
        self.out_features = out_features   # output dim
        self.dropout = dropout    # dropout rate
        self.alpha = alpha     # leakyrelu alpha
        
        #embedding layer
        self.aqi_idEmbed = nn.Embedding(35, 2)
        self.aqi_monthEmbed = nn.Embedding(13, 2)
        self.aqi_weekdayEmbed = nn.Embedding(7, 2)
        self.aqi_hourEmbed = nn.Embedding(24, 2)
        self.meo_windEmbed = nn.Embedding(9, 2)
        self.meo_idEmbed = nn.Embedding(18, 2)
        self.meo_monthEmbed = nn.Embedding(13, 2)
        self.meo_weekdayEmbed = nn.Embedding(7, 2)
        self.meo_hourEmbed = nn.Embedding(24, 2)
        
        # heterogeneous station graph trainable parameters
        self.W_xa = nn.Parameter(torch.zeros(size=(self.in_aqi_features-8, out_features))) 
        nn.init.xavier_uniform_(self.W_xa.data, gain=1.414)  # initilization
        self.W_xm = nn.Parameter(torch.zeros(size=(self.in_meo_features-10, out_features))) 
        nn.init.xavier_uniform_(self.W_xm.data, gain=1.414)  # initilization
        
        self.W_ua = nn.Parameter(torch.zeros(size=(self.in_aqi_features, out_features))) 
        nn.init.xavier_uniform_(self.W_ua.data, gain=1.414)  # initilization
        self.W_um = nn.Parameter(torch.zeros(size=(self.in_meo_features, out_features))) 
        nn.init.xavier_uniform_(self.W_um.data, gain=1.414)  # initilization
        self.a_aa = nn.Parameter(torch.zeros(size=(2*out_features+121, 1)))
        nn.init.xavier_uniform_(self.a_aa.data, gain=1.414)   # initilization
        self.a_am = nn.Parameter(torch.zeros(size=(2*out_features+121, 1)))
        nn.init.xavier_uniform_(self.a_am.data, gain=1.414)   # initilization
        self.a_ma = nn.Parameter(torch.zeros(size=(2*out_features+121, 1)))
        nn.init.xavier_uniform_(self.a_ma.data, gain=1.414)   # initilization
        self.a_mm = nn.Parameter(torch.zeros(size=(2*out_features+121, 1)))
        nn.init.xavier_uniform_(self.a_mm.data, gain=1.414)   # initilization
        
        # homogeneous aqi graph trainable parameters
        self.W_ha = nn.Parameter(torch.zeros(size=(self.in_aqi_features, out_features)))  
        nn.init.xavier_uniform_(self.W_ha.data, gain=1.414)  # initilization
        self.a_ha = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a_ha.data, gain=1.414)   # initilization
        
        # homogeneous weather graph trainable parameters
        self.W_hm = nn.Parameter(torch.zeros(size=(self.in_meo_features, out_features)))  
        nn.init.xavier_uniform_(self.W_hm.data, gain=1.414)  # initilization
        self.a_hm = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a_hm.data, gain=1.414)   # initilization
        
        # leakyrelu layer
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, aqi_inp, meo_inp, aqi_ex, meo_ex, context_feat, adj, adj_norm):
        """
        aqi_inp, meo_inp: input features.
        adj: heterogeneous station graph.
        adj_norm: normalized heterogeneous station graph.
        """
        # embedding layer
        
        aqi_id = self.aqi_idEmbed(aqi_ex[:, :, 0])
        aqi_month = self.aqi_monthEmbed(aqi_ex[:, :, 1])
        aqi_weekday = self.aqi_weekdayEmbed(aqi_ex[:, :, 2])
        aqi_hour = self.aqi_hourEmbed(aqi_ex[:, :, 3])
        meo_wind = self.meo_windEmbed(meo_ex[:, :, 0])
        meo_id = self.meo_idEmbed(meo_ex[:, :, 1])
        meo_month = self.meo_monthEmbed(meo_ex[:, :, 2])
        meo_weekday = self.meo_weekdayEmbed(meo_ex[:, :, 3])
        meo_hour = self.meo_hourEmbed(meo_ex[:, :, 4])
#         print(aqi_ex.shape, meo_ex.shape)
#         print(aqi_inp.shape, meo_inp.shape, self.W_xa.shape, self.W_xm.shape)
        
        attri_aqi = torch.matmul(aqi_inp, self.W_xa) # (B, N_aqi, out_features)
        attri_meo = torch.matmul(meo_inp, self.W_xm) # (B, N_meo, out_features)
        heter_attri = torch.cat([attri_aqi, attri_meo], axis=1)
        
        aqi_inp = torch.cat([aqi_inp, aqi_id, aqi_month, aqi_weekday, aqi_hour], dim=2)
        meo_inp = torch.cat([meo_inp, meo_wind, meo_id, meo_month, meo_weekday, meo_hour], dim=2)

        batch_size = aqi_inp.shape[0]
        
        # heterogeneous graph attention
        heter_aqi = torch.matmul(aqi_inp, self.W_ua) # (B, N, out_features)
        heter_meo = torch.matmul(meo_inp, self.W_um) # (B, N, out_features)
        heter_feat = torch.cat([heter_aqi, heter_meo], axis=1)
        heter_feat = torch.cat([heter_feat, context_feat.unsqueeze(0).repeat(batch_size, 1, 1)], axis=2)
        N_aqi = heter_aqi.size()[1]
        N_meo = heter_meo.size()[1]
        N = N_aqi+N_meo
        heter_input = torch.cat([heter_feat.repeat(1, 1, N).view(batch_size, N*N, -1), \
                               heter_feat.repeat(1, N, 1)], dim=2)
        heter_input = heter_input.view(batch_size, N, -1, 2*self.out_features+120) # (B, N, N, 2*out_features)
        adj_norm = adj_norm.unsqueeze(0)
        adj_norm = adj_norm.unsqueeze(3)
        adj_norm = adj_norm.repeat(batch_size, 1, 1, 1) # (B, N, N, 1)
        heter_input = torch.cat([heter_input, adj_norm], axis=3)
        heter_input_aa = heter_input[:, :N_aqi, :N_aqi, :]
        heter_input_am = heter_input[:, :N_aqi, -1*N_meo:, :]
        heter_input_ma = heter_input[:, -1*N_meo:, :N_aqi, :]
        heter_input_mm = heter_input[:, -1*N_meo:, -1*N_meo:, :]
#         adj_aa = adj[:N_aqi, :N_aqi]
#         adj_am = adj[:N_aqi, -1*N_meo:]
#         adj_ma = adj[-1*N_meo:, :N_aqi]
#         adj_mm = adj[-1*N_meo:, -1*N_meo:]
        e_aa = self.leakyrelu(torch.matmul(heter_input_aa, self.a_aa).squeeze(3))
        e_am = self.leakyrelu(torch.matmul(heter_input_am, self.a_am).squeeze(3))
        e_ma = self.leakyrelu(torch.matmul(heter_input_ma, self.a_ma).squeeze(3))
        e_mm = self.leakyrelu(torch.matmul(heter_input_mm, self.a_mm).squeeze(3))
        e_1 = torch.cat([e_aa, e_am], axis=2)
        e_2 = torch.cat([e_ma, e_mm], axis=2)
        e = torch.cat([e_1, e_2], axis=1)
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj>0, e, zero_vec)   # (B, N, N)
        attention = F.softmax(attention, dim=2)    # (B, N, N)
        heter_out = torch.matmul(attention, heter_attri)  # (B, N, N).(B, N, out_features) => (B, N, out_features)
        heter_aqi = heter_out[:, :N_aqi, :]
        heter_meo = heter_out[:, N_aqi:, :]
        
#         # homogeneous aqi graph attention
#         homo_aqi = torch.matmul(aqi_inp, self.W_ha)   # (B, N, out_features)
#         N_aqi = homo_aqi.size()[1]
#         adj_aqi = adj[:N_aqi, :N_aqi]
#         aqi_input = torch.cat([homo_aqi.repeat(1, 1, N_aqi).view(batch_size, N_aqi*N_aqi, -1), \
#                                homo_aqi.repeat(1, N_aqi, 1)], dim=2)
#         aqi_input = aqi_input.view(batch_size, N_aqi, -1, 2*self.out_features) # (B, N, N, 2*out_features)
#         e = self.leakyrelu(torch.matmul(aqi_input, self.a_ha).squeeze(3)) # (B, N, N)
#         zero_vec = -1e12 * torch.ones_like(e)
#         attention = torch.where(adj_aqi>0, e, zero_vec)   # (B, N, N)
#         attention = F.softmax(attention, dim=2)    # (B, N, N)
# #         attention = F.dropout(attention, self.dropout, training=self.training)   # dropout layer
#         homo_aqi = torch.matmul(attention, homo_aqi)  # (B, N, N).(B, N, out_features) => (B, N, out_features)
    
#         # homogeneous weather graph attention
#         homo_meo = torch.matmul(meo_inp, self.W_hm)   # [B, N, out_features]
#         N_meo = homo_meo.size()[1]
#         adj_meo = adj[-1*N_meo:, -1*N_meo:]
#         meo_input = torch.cat([homo_meo.repeat(1, 1, N_meo).view(batch_size, N_meo*N_meo, -1), \
#                                homo_meo.repeat(1, N_meo, 1)], dim=2)
#         meo_input = meo_input.view(batch_size, N_meo, -1, 2*self.out_features) # (B, N, N, 2*out_features)
#         e = self.leakyrelu(torch.matmul(meo_input, self.a_hm).squeeze(3)) # (B, N, N)
#         zero_vec = -1e12 * torch.ones_like(e)
#         attention = torch.where(adj_meo>0, e, zero_vec)   # (B, N, N)
#         attention = F.softmax(attention, dim=2)    # (N, N)
# #         attention = F.dropout(attention, self.dropout, training=self.training)   # dropout layer
#         homo_meo = torch.matmul(attention, homo_meo)  # (B, N, N).(B, N, out_features) => (B, N, out_features)
    
#         heter_aqi = torch.cat([heter_aqi, homo_aqi], axis=2)
#         heter_meo = torch.cat([heter_meo, homo_meo], axis=2)
    
        return heter_aqi, heter_meo

class Generator(nn.Module):
    def __init__(self, dropout=0.5, alpha=0.2, hid_dim=32, t_out=48):
        super(Generator, self).__init__()
        self.aqi_nfeat = 6
        self.meo_nfeat = 4
        self.hid_dim = hid_dim
        # FC layers
        self.output_fc_aqi = nn.Linear(hid_dim+54, 1, bias=True)
        self.output_fc_temp = nn.Linear(hid_dim+3, 1, bias=True)
        self.output_fc_humi = nn.Linear(hid_dim+3, 1, bias=True)
        self.output_fc_wind = nn.Linear(hid_dim+3, 1, bias=True)
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        #embedding layer
        self.aqi_idEmbed = nn.Embedding(35, 2)
        self.aqi_monthEmbed = nn.Embedding(13, 2)
        self.aqi_weekdayEmbed = nn.Embedding(7, 2)
        self.aqi_hourEmbed = nn.Embedding(24, 2)
        self.meo_windEmbed = nn.Embedding(9, 2)
        self.meo_idEmbed = nn.Embedding(18, 2)
        self.meo_monthEmbed = nn.Embedding(13, 2)
        self.meo_weekdayEmbed = nn.Embedding(7, 2)
        self.meo_hourEmbed = nn.Embedding(24, 2)
        
        # context-aware heterogeneous graph attention
        self.chgat1 = HeterGraphAttentionLayer(in_aqi_features=self.aqi_nfeat, in_meo_features=self.meo_nfeat, \
                                               out_features=self.hid_dim, dropout=dropout, alpha=alpha)
        self.chgat2 = HeterGraphAttentionLayer(in_aqi_features=self.hid_dim, in_meo_features=self.hid_dim, \
                                               out_features=self.hid_dim, dropout=dropout, alpha=alpha)
        
        # SGRU Cell
        self.GRU_aqi = nn.GRUCell(hid_dim, hid_dim, bias=True)
        self.GRU_meo = nn.GRUCell(hid_dim, hid_dim, bias=True)
        nn.init.xavier_uniform_(self.GRU_aqi.weight_ih, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.GRU_aqi.weight_hh, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.GRU_meo.weight_ih, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.GRU_meo.weight_hh, gain=math.sqrt(2.0))
        
        # SGRU Cell
        self.GRU_daqi = nn.GRUCell(6, hid_dim, bias=True)
        self.GRU_dmeo = nn.GRUCell(6, hid_dim, bias=True)
        nn.init.xavier_uniform_(self.GRU_daqi.weight_ih, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.GRU_daqi.weight_hh, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.GRU_dmeo.weight_ih, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.GRU_dmeo.weight_hh, gain=math.sqrt(2.0))
        
        # Parameter initialization
        for ele in self.modules():
            if isinstance(ele, nn.Linear):
                nn.init.xavier_uniform_(ele.weight,gain=math.sqrt(2.0))
    
    def forward(self, X_batch, y_batch, context_feat, adj, adj_norm):
        """
        :param adj: heterogeneous station graph adj.
        :param X_batch: Input data of shape (batch_size, num_nodes(N), T_in, num_features(F)).
        :param y_batch: To init GRU hidden state with shape (N, 2*hid_dim).
        :return: future prediction
        """
        
        X_aqi, X_meo, X_aqi_ex, X_meo_ex = X_batch
        y_aqi, y_meo, y_nwp, y_aqi_mask, y_meo_mask, y_aqi_ex, y_meo_ex = y_batch
        
        batch_size, T, N_aqi, F_aqi = X_aqi.size() #(Batch, T_in, N_aqi, F_aqi)
        batch_size, T, N_meo, F_meo = X_meo.size() #(Batch, T_in, N_meo, F_meo)
        
        # to init SGRU hidden state
        h_aqi = torch.zeros(batch_size, N_aqi, self.hid_dim).cuda()
        h_meo = torch.zeros(batch_size, N_meo, self.hid_dim).cuda()

        # encoder
        for i in range(T):
            # context-aware heterogeneous graph attention
            aqi_feat, meo_feat = self.chgat1(X_aqi[:, i, :, :], X_meo[:, i, :, :], \
                                             X_aqi_ex[:, i, :, :], X_meo_ex[:, i, :, :], context_feat, adj, adj_norm)
#             aqi_feat, meo_feat = self.chgat2(aqi_feat, meo_feat, adj)
            # station-aware GRU
            h_aqi = self.GRU_aqi(aqi_feat.contiguous().view(-1, self.hid_dim), h_aqi.view(-1, self.hid_dim)) # (B*N, tmp_hid)
            h_meo = self.GRU_meo(meo_feat.contiguous().view(-1, self.hid_dim), h_meo.view(-1, self.hid_dim)) # (B*N, tmp_hid)
            h_aqi = h_aqi.view(batch_size, N_aqi, -1)
            h_meo = h_meo.view(batch_size, N_meo, -1)

        # decoder
        aqi_id = self.aqi_idEmbed(y_aqi_ex[:, :, :, 0]) # (B, T_out, N_aqi)
        aqi_month = self.aqi_monthEmbed(y_aqi_ex[:, :, :, 1])
        aqi_weekday = self.aqi_weekdayEmbed(y_aqi_ex[:, :, :, 2])
        aqi_hour = self.aqi_hourEmbed(y_aqi_ex[:, :, :, 3])
        meo_id = self.meo_idEmbed(y_meo_ex[:, :, :, 1])
        meo_month = self.meo_monthEmbed(y_meo_ex[:, :, :, 2])
        meo_weekday = self.meo_weekdayEmbed(y_meo_ex[:, :, :, 3])
        meo_hour = self.meo_hourEmbed(y_meo_ex[:, :, :, 4])
        
        c_aqi = torch.cat([aqi_month, aqi_weekday, aqi_hour], dim=2) # (B, T_out, N_aqi, Embed)
        c_meo = torch.cat([meo_month, meo_weekday, meo_hour], dim=2)
        
        for i in range(48):
            # heterogeneous GRU
            h_aqi = self.GRU_daqi(c_aqi[:, i, :, :].contiguous().view(-1, 6), h_aqi.view(-1, self.hid_dim))
            h_meo = self.GRU_dmeo(c_meo[:, i, :, :].contiguous().view(-1, 6), h_meo.view(-1, self.hid_dim))
            h_aqi = h_aqi.view(batch_size, N_aqi, -1)
            h_meo = h_meo.view(batch_size, N_meo, -1)
            if i == 0:
                aqi_ex = y_nwp[:,i,:,:].contiguous().view(-1, 18*3).unsqueeze(1).repeat(1, 35, 1)
                out_aq = torch.cat((h_aqi,aqi_ex),axis=2)
                out_aqi = self.output_fc_aqi(out_aq) # (B, N_a, 1)
                out_meo = torch.cat((h_meo,y_nwp[:,i,:,:]), axis=2)
                out_temp = self.output_fc_temp(out_meo) # (B, N_w, 1)
                out_humi = self.output_fc_humi(out_meo) # (B, N_w, 1)
                out_wind = self.output_fc_wind(out_meo) # (B, N_w, 1)
            else:
                aqi_ex = y_nwp[:,i,:,:].contiguous().view(-1, 18*3).unsqueeze(1).repeat(1, 35, 1)
                out_aq = torch.cat((h_aqi,aqi_ex),axis=2)
                out_aqi = torch.cat((out_aqi, self.output_fc_aqi(out_aq)), axis=2) # (B, N_a, 1)
                out_meo = torch.cat((h_meo,y_nwp[:,i,:,:]), axis=2)
                out_temp = torch.cat((out_temp,self.output_fc_temp(out_meo)),axis=2) # (B, N_w, 1)
                out_humi = torch.cat((out_humi,self.output_fc_humi(out_meo)),axis=2) # (B, N_w, 1)
                out_wind = torch.cat((out_wind,self.output_fc_wind(out_meo)),axis=2) # (B, N_w, 1)

#         aqi_id = self.aqi_idEmbed(y_aqi_ex[:, :, :, 0]).transpose(1, 2)
#         aqi_weekday = self.aqi_weekdayEmbed(y_aqi_ex[:, :, :, 1]).transpose(1, 2)
#         aqi_hour = self.aqi_hourEmbed(y_aqi_ex[:, :, :, 2]).transpose(1, 2)
#         meo_id = self.meo_idEmbed(y_meo_ex[:, :, :, 1]).transpose(1, 2)
#         meo_weekday = self.meo_weekdayEmbed(y_meo_ex[:, :, :, 2]).transpose(1, 2)
#         meo_hour = self.meo_hourEmbed(y_meo_ex[:, :, :, 3]).transpose(1, 2)
        
#         h_aqi = h_aqi.unsqueeze(2).repeat(1, 1, 48, 1)
#         h_meo = h_meo.unsqueeze(2).repeat(1, 1, 48, 1)
        
#         print(h_aqi.shape, aqi_id.shape)
        
#         h_aqi = torch.cat([h_aqi, aqi_id, aqi_weekday, aqi_hour], dim=3)
#         h_meo = torch.cat([h_meo, meo_id, meo_weekday, meo_hour], dim=3)
        
#         print(h_aqi.shape, h_meo.shape)
        
#         out_aqi = self.output_fc_aqi(h_aqi) # (B, N_a, T_out)
#         out_temp = self.output_fc_temp(h_meo) # (B, N_w, T_out)
#         out_humi = self.output_fc_humi(h_meo) # (B, N_w, T_out)
#         out_wind = self.output_fc_wind(h_meo) # (B, N_w, T_out)
        
        return out_aqi, out_temp, out_humi, out_wind

class macroDiscriminator(nn.Module):
    def __init__(self):
        super(macroDiscriminator, self).__init__()
        self.GRU = nn.GRUCell(N_aqi+3*N_meo, N_aqi+3*N_meo, bias=True)
        nn.init.xavier_uniform_(self.GRU.weight_ih, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.GRU.weight_hh, gain=math.sqrt(2.0))
        self.metric_fc = nn.Linear(N_aqi+3*N_meo, 32, bias=True)
        self.output_fc = nn.Linear(32, 1, bias=True)
        self.output_ac = nn.Sigmoid()
        
    def forward(self, out):
        """
        :param out: (B, T, N_aqi+3*N_meo)
        :return: macro discriminator classification score
        """
        h_out = torch.zeros(out.shape[0], N_aqi+3*N_meo).cuda()
#         out_aqi = torch.transpose(out_aqi, 1, 2) # (B, T, N_aqi+3*N_meo)
#         out_flatten = torch.cat((out_aqi, out_temp, out_humi, out_wind), 2) # (B, T_out, N_a+3*N_w)
        
        for i in range(120):
            h_out = self.GRU(out[:, i, :], h_out) # (B*N, 2*tmp_hid)
            
        out_mmetric = self.metric_fc(h_out)
        out_mmetric = self.output_ac(out_mmetric)
        
        out_mscore = self.output_fc(out_mmetric)
        out_mscore = self.output_ac(out_mscore)
        
        return out_mscore, out_mmetric

class spatialDiscriminator(nn.Module):
    def __init__(self):
        super(spatialDiscriminator, self).__init__()
        self.aqi_trans_fc = nn.Linear(1, 6, bias=True)
        self.output_fc = nn.Linear(32, 1, bias=True)
        self.output_ac = nn.Sigmoid()
        
        self.in_aqi_features = 6   # input dim
        self.in_meo_features = 3   # input dim
        self.out_features = 16   # output dim
        self.alpha = 0.2     # leakyrelu alpha

        self.W_xa = nn.Parameter(torch.zeros(size=(self.in_aqi_features, self.out_features))) 
        nn.init.xavier_uniform_(self.W_xa.data, gain=1.414)  # initilization
        self.W_xm = nn.Parameter(torch.zeros(size=(self.in_meo_features, self.out_features))) 
        nn.init.xavier_uniform_(self.W_xm.data, gain=1.414)  # initilization
        
        self.W_ua = nn.Parameter(torch.zeros(size=(self.in_aqi_features, self.out_features))) 
        nn.init.xavier_uniform_(self.W_ua.data, gain=1.414)  # initilization
        self.W_um = nn.Parameter(torch.zeros(size=(self.in_meo_features, self.out_features))) 
        nn.init.xavier_uniform_(self.W_um.data, gain=1.414)  # initilization
        self.a_aa = nn.Parameter(torch.zeros(size=(2*self.out_features+1, 1)))
        nn.init.xavier_uniform_(self.a_aa.data, gain=1.414)   # initilization
        self.a_am = nn.Parameter(torch.zeros(size=(2*self.out_features+1, 1)))
        nn.init.xavier_uniform_(self.a_am.data, gain=1.414)   # initilization
        self.a_ma = nn.Parameter(torch.zeros(size=(2*self.out_features+1, 1)))
        nn.init.xavier_uniform_(self.a_ma.data, gain=1.414)   # initilization
        self.a_mm = nn.Parameter(torch.zeros(size=(2*self.out_features+1, 1)))
        nn.init.xavier_uniform_(self.a_mm.data, gain=1.414)   # initilization
        
        # leakyrelu layer
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, out_aqi, out_meo, adj, adj_norm):
        """
        :param out_aqi: # (B, T_out, N_aqi, 1)
        :param out_meo: # (B, T_out, N_meo, 3)
        :return: spatial discriminator classification score
        """
        out_aqi = out_aqi.contiguous().view(-1, out_aqi.shape[2], out_aqi.shape[3]) # (B*T_out, N_aqi, 1)
        out_meo = out_meo.contiguous().view(-1, out_meo.shape[2], out_meo.shape[3]) # (B*T_out, N_aqi, 1)
        out_aqi = self.aqi_trans_fc(out_aqi) # (B*T_out, N_aqi, F)
        
        attri_aqi = torch.matmul(out_aqi, self.W_xa) # (B, N, out_features)
        attri_meo = torch.matmul(out_meo, self.W_xm) # (B, N, out_features)
        heter_attri = torch.cat([attri_aqi, attri_meo], axis=1)

        batch_size = out_aqi.shape[0]
        
        # heterogeneous graph attention
        heter_aqi = torch.matmul(out_aqi, self.W_ua) # (B, N_aqi, out_features)
        heter_meo = torch.matmul(out_meo, self.W_um) # (B, N_meo, out_features)
        heter_feat = torch.cat([heter_aqi, heter_meo], axis=1)
        N_aqi = heter_aqi.size()[1]
        N_meo = heter_meo.size()[1]
        N = N_aqi+N_meo
        heter_input = torch.cat([heter_feat.repeat(1, 1, N).view(batch_size, N*N, -1), \
                               heter_feat.repeat(1, N, 1)], dim=2)
        heter_input = heter_input.view(batch_size, N, -1, 2*self.out_features) # (B, N, N, 2*out_features)
        adj_norm = adj_norm.unsqueeze(0)
        adj_norm = adj_norm.unsqueeze(3)
        adj_norm = adj_norm.repeat(batch_size, 1, 1, 1) # (B, N, N, 1)
        heter_input = torch.cat([heter_input, adj_norm], axis=3)
        heter_input_aa = heter_input[:, :N_aqi, :N_aqi, :]
        heter_input_am = heter_input[:, :N_aqi, -1*N_meo:, :]
        heter_input_ma = heter_input[:, -1*N_meo:, :N_aqi, :]
        heter_input_mm = heter_input[:, -1*N_meo:, -1*N_meo:, :]
        e_aa = self.leakyrelu(torch.matmul(heter_input_aa, self.a_aa).squeeze(3))
        e_am = self.leakyrelu(torch.matmul(heter_input_am, self.a_am).squeeze(3))
        e_ma = self.leakyrelu(torch.matmul(heter_input_ma, self.a_ma).squeeze(3))
        e_mm = self.leakyrelu(torch.matmul(heter_input_mm, self.a_mm).squeeze(3))
        e_1 = torch.cat([e_aa, e_am], axis=2)
        e_2 = torch.cat([e_ma, e_mm], axis=2)
        e = torch.cat([e_1, e_2], axis=1)
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj>0, e, zero_vec)   # (B, N, N)
        attention = F.softmax(attention, dim=2)    # (B, N, N)
        heter_out = torch.matmul(attention, heter_attri)  # (B, N, N).(B, N, out_features) => (B, N, out_features)
        heter_aqi = heter_out[:, :N_aqi, :] # (B, N_aqi, out_features)
        heter_meo = heter_out[:, N_aqi:, :] # (B, N_meo, out_features)
        
        heter_aqi = torch.mean(heter_aqi, dim=1)
        heter_meo = torch.mean(heter_meo, dim=1)
        out_smetric = torch.cat((heter_aqi, heter_meo), dim=1)
        out_smetric = self.output_ac(out_smetric)
        out_sscore = self.output_fc(out_smetric)
        out_sscore = self.output_ac(out_sscore)

        return out_sscore, out_smetric
        
class temporalDiscriminator(nn.Module):
    def __init__(self):
        super(temporalDiscriminator, self).__init__()
        self.metric_fc_aqi = nn.Linear(120, 32, bias=True)
        self.metric_fc_meo = nn.Linear(3*120, 32, bias=True)
        self.output_fc_aqi = nn.Linear(32, 1, bias=True)
        self.output_fc_meo = nn.Linear(32, 1, bias=True)
        self.output_ac = nn.Sigmoid()
        
    def forward(self, out_aqi, out_meo):
        """
        :param out_aqi: (B, N_aqi, T)
        :param out_meo: (B, N_meo, 3*T)
        :return: temporal discriminator classification score
        """
        out_aqi = torch.transpose(out_aqi, 1, 2)
        out_meo = out_meo.view(out_meo.shape[0], out_meo.shape[1], 3, -1)
        out_meo = torch.transpose(out_meo, 1, 3)
        out_meo = out_meo.contiguous().view(out_meo.shape[0], out_meo.shape[1], -1)
        # We can also use GRU to model temporal dependency
#         h_aqi = torch.zeros(out.shape[0], 16).cuda()
#         h_meo = torch.zeros(out.shape[0], 16).cuda()
#         for i in range(120):
#             h_out = self.GRU(out_aqi[:, i, :], h_out) # (B*N, 2*tmp_hid)
#         for i in range(120):
#             h_out = self.GRU(out_meo[:, i, :], h_out) # (B*N, 2*tmp_hid)
        aqi_tmetric = self.metric_fc_aqi(out_aqi) # (B, N_aqi, 64)
        meo_tmetric = self.metric_fc_meo(out_meo) # (B, N_meo, 64)
        aqi_tmetric = self.output_ac(aqi_tmetric)
        meo_tmetric = self.output_ac(meo_tmetric)
        aqi_tscore = self.output_fc_aqi(aqi_tmetric) # (B, N_aqi, 1)
        meo_tscore = self.output_fc_meo(meo_tmetric) # (B, N_meo, 1)
        out_tmetric = torch.cat((aqi_tmetric, meo_tmetric), 1) # (B, N_aqi+N_meo, 64)
        out_tscore = torch.cat((aqi_tscore, meo_tscore), 1) # (B, N_aqi+N_meo, 1)
        out_tscore = self.output_ac(out_tscore)
        out_tscore = out_tscore.view(-1, 1)
        return out_tscore, out_tmetric