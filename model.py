import math
from functools import partial
import json
import os
import torch.nn.functional as F
from collections import namedtuple
import random
from torch import Tensor
from torch.nn import Parameter

import torch
import torch.nn as nn
from torch.autograd import Variable

import fairseq
from WavLM import WavLM, WavLMConfig

from DecoAr import DeCoAR2

from fairseq.models.hubert import HubertModel

class SSLModel(nn.Module):
    
    def __init__(self, small, n_layers):
        super(SSLModel, self).__init__()

        if small:
            cp_path = '/netscratch/yelkheir/ssl_models/wav2vec_small.pt.1'
        else:
            cp_path = '/netscratch/yelkheir/ssl_models/xlsr2_300m.pt'

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]   
        self.n_layers = n_layers

        return 
    
    def extract_feat(self, input_data):
        
        input_data = input_data.squeeze(1)
        dict_ = self.model(input_data, mask=False, features_only=True)
        x, layerresult = dict_['x'], dict_['layer_results']
        return torch.stack([t[0].permute(1,0,2) if isinstance(t, tuple) else t for t in layerresult[:self.n_layers]], dim=1)

    def freeze_feature_extraction(self):
        """Freezes the feature extraction layers of the base SSL model."""
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

class SSLModelHubert(nn.Module):
    
    def __init__(self, small, n_layers):
        super(SSLModelHubert, self).__init__()

        if small:
            cp_path = '/netscratch/yelkheir/ssl_models/hubert_small.pt'
        else:
            cp_path = '/netscratch/yelkheir/ssl_models/hubert_large_ll60k.pt'

        if not small:
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
            self.model = model[0]   
            self.n_layers = n_layers
        
        else:
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(["/netscratch/yelkheir/ssl_models/hubert_small_1.pt"])
            self.model = model[0]  
            state_dict = torch.load(cp_path, map_location='cpu')
            self.model.label_embs_concat = torch.nn.Parameter(torch.randn(1004, 256))

            self.model.load_state_dict(state_dict['model'], strict=False)
            self.n_layers = n_layers


        self.small = small

        return 
    
    def extract_feat(self, input_data):
        input_data = input_data.squeeze(1)
        dict_ = self.model(input_data, mask=False, features_only=True)
        x, layerresult = dict_['x'], dict_['features']
        return torch.stack([t[0].permute(1,0,2) if isinstance(t, tuple) else t for t in layerresult[:self.n_layers]], dim=1)

    def freeze_feature_extraction(self):
        """Freezes the feature extraction layers of the base SSL model."""
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

class SSLModelWavlm(nn.Module):
    def __init__(self, small, n_layers):
        super(SSLModelWavlm, self).__init__()
        
        if small:
            cp_path = '/netscratch/yelkheir/ssl_models/WavLM-Base+.pt'
        else:
            cp_path = '/netscratch/yelkheir/ssl_models/WavLM-Large.pt'

        checkpoint = torch.load(cp_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        self.n_layers = n_layers
        self.model = WavLM(cfg)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        
        return

    def extract_feat(self, input_data):
        input_data = input_data.squeeze(1)
        x, layers = self.model.extract_features(input_data, mask=False, ret_layer_results=True)[0]

        return torch.stack(layers[:self.n_layers], dim=1).permute(2,1,0,3).contiguous()
    
    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

class SSLDecoAR(nn.Module):
    def __init__(self, n_layers):
        super(SSLDecoAR, self).__init__()

        self.model = DeCoAR2()
        checkpoint = torch.load("/netscratch/yelkheir/ssl_models/checkpoint_decoar2.pt")
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.n_layers = n_layers

    def extract_feat(self, input_data):
        x, y = self.model(input_data)
        return torch.stack([t[0].permute(1,0,2) if isinstance(t, tuple) else t for t in y[:self.n_layers]], dim=1)
    
    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        #print('x1',x1.shape)
        #print('x2',x2.shape)
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        #print('num_type1',num_type1)
        #print('num_type2',num_type2)
        x1 = self.proj_type1(x1)
        #print('proj_type1',x1.shape)
        x2 = self.proj_type2(x2)
        #print('proj_type2',x2.shape)
        x = torch.cat([x1, x2], dim=1)
        #print('Concat x1 and x2',x.shape)
        
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
            #print('master',master.shape)
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)
        #print('master',master.shape)
        # directional edge for master node
        master = self._update_master(x, master)
        #print('master',master.shape)
        # projection
        x = self._project(x, att_map)
        #print('proj x',x.shape)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        #print('x1',x1.shape)
        x2 = x.narrow(1, num_type1, num_type2)
        #print('x2',x2.shape)
        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)
        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h

class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        #print('out',out.shape)
        out = self.conv1(x)

        #print('aft conv1 out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        #print('conv2 out',out.shape)
        
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        #out = self.mp(out)
        return out

class AASIST(nn.Module):
    def __init__(self, config, device="cuda"):
        super(AASIST, self).__init__()

        self.config = config 
        self.model = config['model']
        self.small = config['small']
        self.n_layers = config['n_layers']

        self.layer_norm = nn.BatchNorm2d(num_features=self.n_layers)

        self.weight_hidd = nn.Parameter(torch.ones(self.n_layers))
        self.n_layers = self.n_layers

        if self.model == "w2v":
            self.ssl = SSLModel(self.small, self.n_layers)
        elif self.model == "wavlm":
            self.ssl = SSLModelWavlm(self.small, self.n_layers)      
        elif self.model == "decor":
            self.ssl = SSLDecoAR(self.n_layers)    
        elif self.model == "hub":
            self.ssl = SSLModelHubert(self.small, self.n_layers)

        if self.small:
            self.out_dim = 768
        else:
            self.out_dim = 1024

        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        if self.small:
            self.out_dim = 768
        else:
            self.out_dim = 1024

        if self.model == "decor":
            self.out_dim = 768

        self.LL = nn.Linear(self.out_dim, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )

        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[1])

        # HS-GAL layer
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

        # self.layer_norm = nn.LayerNorm(self.out_dim)

    def forward(self, x):

        feature = self.ssl.extract_feat(x)
        feature = self.layer_norm(feature)
        norm_weights = F.softmax(self.weight_hidd, dim=-1)
        weighted_feature = (feature * norm_weights.view(-1, 1, 1)).sum(dim=1)
        x = self.LL(weighted_feature)  # (bs, frame_number, feat_out_dim)

        # Post-processing on front-end features
        x = x.transpose(1, 2)  # (bs, feat_out_dim, frame_number)
        x = x.unsqueeze(dim=1)  # Add channel
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        
        w = self.attention(x)

        # SA for spectral feature
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        # Graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # SA for temporal feature
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        # Graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # Learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # Inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # Inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return output

    def get_weights(self, ):
        return self.weight_hidd

class Classic_Attention(nn.Module):
    
    def __init__(self,input_dim, embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.lin_proj = nn.Linear(input_dim,embed_dim)
        self.v = torch.nn.Parameter(torch.randn(embed_dim))
    
    def forward(self,inputs):
        lin_out = self.lin_proj(inputs)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        attention_weights = F.tanh(lin_out.bmm(v_view).squeeze())
        attention_weights_normalized = F.softmax(attention_weights)
        return attention_weights_normalized

class ScaledDotProduct_attention(nn.Module):
    """
    Scaled dot product attention
    """

    def __init__(self, embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.scaling = self.embed_dim ** -0.5
    
    
        self.in_proj_weight = Parameter(torch.Tensor(2 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        
        self.reset_parameters()
        
        self.in_proj_weight = Parameter(torch.Tensor(2 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(2 * embed_dim))
        
    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
        
    def in_proj_qk(self, query):
        return self._in_proj(query).chunk(2, dim=-1)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
        
    def forward(self,query,key):
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        q, k = self.in_proj_qk(query)
        q *= self.scaling
        return q,k
    
class FFN(nn.Module):
    def __init__(self, config, device="cuda"):
        super(FFN, self).__init__()
        self.config = config 
        self.model = config['model']
        self.small = config['small']
        self.n_layers = config['n_layers']

        self.weight_hidd = nn.Parameter(torch.ones(self.n_layers))
        self.n_layers = self.n_layers

        if self.model == "w2v":
            self.ssl = SSLModel(self.small, self.n_layers)
        elif self.model == "wavlm":
            self.ssl = SSLModelWavlm(self.small, self.n_layers)      
        elif self.model == "decor":
            self.ssl = SSLDecoAR(self.n_layers)    
        elif self.model == "hub":
            self.ssl = SSLModelHubert(self.small, self.n_layers)
            
        if self.small:
            self.out_dim = 768
        else:
            self.out_dim = 1024

        if self.model == "decor":
            self.out_dim = 768

        self.layer_norm = nn.BatchNorm2d(num_features=self.n_layers)
        self.LL = nn.Linear(self.out_dim, 128)
        self.LL2 = nn.Linear(128, 128)
        self.attention_block = Classic_Attention(128,128)
        self.LL3 =  nn.Linear(256, 128)
        self.LL4 =  nn.Linear(128, 2)
        self.relu = nn.ReLU(inplace=True)

    def weighted_sd(self,inputs,attention_weights, mean):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        hadmard_prod = torch.mul(inputs,el_mat_prod)
        variance = torch.sum(hadmard_prod,1) - torch.mul(mean,mean)
        return variance

    def stat_attn_pool(self,inputs,attention_weights):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        mean = torch.mean(el_mat_prod,1)
        variance = self.weighted_sd(inputs,attention_weights,mean)
        stat_pooling = torch.cat((mean,variance),1)
        return stat_pooling
    
    def forward(self, x):
        feature = self.ssl.extract_feat(x)
        feature = self.layer_norm(feature)
        norm_weights = F.softmax(self.weight_hidd, dim=-1)
        weighted_feature = (feature * norm_weights.view(-1, 1, 1)).sum(dim=1)

        x = self.LL(weighted_feature)
        x = self.relu(x)
        x = self.LL2(x)
        x = self.relu(x)

        att_weights = self.attention_block(x)

        if len(att_weights.shape) == 1:
            att_weights = att_weights.unsqueeze(dim=0)

        x = self.stat_attn_pool(x, att_weights)
    
        x = self.LL3(x)
        x = self.relu(x)
        x = self.LL4(x)

        return x
    
    def get_weights(self, ):
        return self.weight_hidd


# rand = torch.randn(16,1,64600)
# model = AASIST({"model": "w2v", "n_layers": 2, "small": 0})
# print(model(rand).shape)