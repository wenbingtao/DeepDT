import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def soft_cross_entropy(pred, soft_targets, reduction='mean'):

    entropy = torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1)
    if reduction == 'mean':
        return torch.mean(entropy)
    elif reduction == 'sum':
        return torch.sum(entropy)
    elif reduction == 'none':
        return entropy

def cal_loss3(pred, target):

    entropies = []
    for i in range(target.shape[1]):
        tmp_target = target[:, i, :]  
        entropies.append(soft_cross_entropy(pred, tmp_target, reduction='none'))
    loss = torch.stack(entropies, dim=1) 
    loss = torch.mean(loss, dim=1)  
    return loss  

def cal_loss4(c_pred, p_target):

    entropies = []
    for i in range(p_target.shape[1]):
        tmp_p_target = p_target[:, i]  
        entropies.append(F.cross_entropy(c_pred, tmp_p_target, reduction='none'))
    loss = torch.stack(entropies, dim=1)
    loss = torch.mean(loss, 1)  
    return loss  


def random_sample(feature, sample_idx):

    pool_features = feature[sample_idx]        
    pool_features = torch.max(pool_features, 1, False)[0]  
    return pool_features


def gather_neighbour(feature, neighbor_idx):


    neighbor_features = feature[neighbor_idx] 
    return neighbor_features


def nearest_interpolation(feature, interp_idx):

    interpolated_features = feature[interp_idx]        
    interpolated_features = torch.squeeze(interpolated_features, 1)    

    return interpolated_features


def feature_fetch(input_tensor, index_tensor):

    feature = input_tensor[index_tensor]  
    feature[index_tensor == -1] = 0

    return feature


class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode=0,
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):

        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

class att_pooling(nn.Module):

    def __init__(self, d_in, d_out):
        super(att_pooling, self).__init__()

        self.fc = nn.Linear(d_in, d_in, bias=False)       
        self.mlp = SharedMLP(d_in, d_out, bn=True, activation_fn=nn.ReLU())      


    def forward(self, feature_set):

        att_activation = self.fc(feature_set)                           
        att_scores = torch.softmax(att_activation, 1)                   
        f_agg = feature_set.mul(att_scores)                             
        f_agg = torch.sum(f_agg, dim=1, keepdim=True)                                
        f_agg = f_agg.permute(2,0,1).unsqueeze(0)                      
        f_agg = self.mlp(f_agg)                                         
        f_agg = f_agg.squeeze(3).squeeze(0).permute(1,0)                         

        return f_agg


class Building_block(nn.Module):

    def __init__(self, geo_in, d_out):                                                     
        super(Building_block, self).__init__()

        self.conv2d_1 = SharedMLP(geo_in, d_out // 2, bn=True, activation_fn=nn.ReLU())
        self.conv2d_2 = SharedMLP(d_out // 2, d_out // 2, bn=True, activation_fn=nn.ReLU())

        self.attentive_pooling_1 = att_pooling(d_out, d_out // 2)                           
        self.attentive_pooling_2 = att_pooling(d_out, d_out)


    def forward(self, f_geo, feature, neigh_idx):

        f_geo = self.conv2d_1(f_geo.permute(2, 0, 1).unsqueeze(0))                      

        f_neighbours = gather_neighbour(feature, neigh_idx)                             

        f_concat = torch.cat((f_neighbours, f_geo.squeeze(0).permute(1, 2, 0)), dim=2)  
                                 
        f_pc_agg = self.attentive_pooling_2(f_concat)  

                                 
        return f_pc_agg


class dilated_res_block(nn.Module):

    def __init__(self, geo_in, d_in, d_out):
        super(dilated_res_block, self).__init__()

        self.conv2d_1 = SharedMLP(d_in, d_out // 2, activation_fn=nn.LeakyReLU(0.2))
        self.conv2d_2 = SharedMLP(d_out, d_out)
        self.shortcut = SharedMLP(d_in, d_out, bn=True)

        self.building_block = Building_block(geo_in, d_out)                                 
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)



    def forward(self, f_geo, feature, neigh_idx):

        f_pc = self.conv2d_1(feature.permute(1,0).unsqueeze(0).unsqueeze(3))                
        f_pc = f_pc.permute(0,2,1,3).squeeze(3).squeeze(0)                                  
        f_pc = self.building_block(f_geo, f_pc, neigh_idx)                                  

        f_pc = self.conv2d_2(f_pc.permute(1,0).unsqueeze(0).unsqueeze(3))                  

        shortcut = self.shortcut(feature.unsqueeze(0).unsqueeze(3).permute(0,2,1,3))        
        lfa = self.lrelu(f_pc + shortcut)    
        lfa = lfa.squeeze(3).squeeze(0).permute(1, 0)

        return lfa


class GraphConvolution(Module):
    """
    Simple GCN, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, channels):
        super(GCN, self).__init__()

        self.gc = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.gc.append(
                GraphConvolution(channels[i], channels[i + 1])
            )
        self.dropout = 0.5

    def forward(self, x, adj):

        for i, gc in enumerate(self.gc):
            x = gc(x, adj)
            x = F.leaky_relu(x, negative_slope=0.1)
            if self.dropout != 0.0:
                x = F.dropout(x,  self.dropout, training=self.training)
        return x

class R_GCN(nn.Module):
    def __init__(self, geo_in):
        super(R_GCN, self).__init__()
        self.fc1 = nn.Linear(geo_in, 8)
        self.ap1 = att_pooling(8, 8)

        self.bn_start = nn.Sequential(
            nn.BatchNorm1d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            dilated_res_block(geo_in, 8, 32),
            dilated_res_block(geo_in, 32, 64),
            dilated_res_block(geo_in, 64, 128)
        ])

        self.mlp = SharedMLP(128, 128, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            SharedMLP(64 + 128, 64, **decoder_kwargs),
            SharedMLP(32 + 64, 32, **decoder_kwargs),
            SharedMLP(32 + 32, 8, **decoder_kwargs),
        ])

        self.fcs = nn.Sequential(
            SharedMLP(8, 64, bn=True, activation_fn=nn.ReLU()),
            
            nn.Dropout(),
 
            SharedMLP(64, 32, activation_fn=nn.LeakyReLU(0.2))
        )

        self.aa = att_pooling(32, 32)  
        self.gc1 = GCN([32, 64, 128])
        self.gc2 = GCN([128, 64, 32])
        self.gc3 = GraphConvolution(32, 2)

    def forward(self, deepdt_data):

        point_feature = self.fc1(deepdt_data.batch_feature[0])    
        point_feature = self.ap1(point_feature)
        point_feature = self.bn_start(point_feature)

        decimation_ratio = 1
        point_feature_stack = []



        # ENCODER
        i = 0
        for lfa in self.encoder:
            f_encoder_i = lfa(deepdt_data.batch_feature[i], point_feature, deepdt_data.neigh_idx[i])
            if i == 0:
                point_feature_stack.append(f_encoder_i.clone())
            f_sampled_i = random_sample(f_encoder_i, deepdt_data.sub_idx[i])
            point_feature = f_sampled_i
            point_feature_stack.append(f_sampled_i.clone())
            i = i + 1

        point_feature = self.mlp(point_feature.permute(1,0).unsqueeze(0).unsqueeze(3))
        point_feature = point_feature.squeeze(3).squeeze(0).permute(1, 0)

        # DECODER
        j = 0
        for mlp in self.decoder:
            f_interp_i = nearest_interpolation(point_feature, deepdt_data.interp_idx[-j - 1])

            point_feature = torch.cat((f_interp_i, point_feature_stack[-j - 2]), dim=1)  
            point_feature = mlp(point_feature.permute(1,0).unsqueeze(0).unsqueeze(3))
            point_feature = point_feature.squeeze(3).squeeze(0).permute(1, 0)
            j+=1

        point_feature = self.fcs(point_feature.permute(1,0).unsqueeze(0).unsqueeze(3))
        point_feature = point_feature.squeeze(3).squeeze(0).permute(1, 0)

        node_feature = feature_fetch(point_feature, deepdt_data.cell_vertex_idx)
        node_feature = self.aa(node_feature)


        node_feature = self.gc1(node_feature, deepdt_data.adj)
        node_feature = self.gc2(node_feature, deepdt_data.adj)
        cell_pred = self.gc3(node_feature, deepdt_data.adj) 
        cell_pred_soft = F.softmax(cell_pred, dim=1)

        loss1 = cal_loss4(cell_pred, deepdt_data.ref_label)

        c_c_target = cell_pred_soft[deepdt_data.adj_idx]  
        loss2 = cal_loss3(cell_pred, c_c_target)

        return cell_pred, loss1, loss2








