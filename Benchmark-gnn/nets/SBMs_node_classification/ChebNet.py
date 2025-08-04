import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    ChebNet - SBM
"""
from layers.Cheb_layer import ChebLayer
from layers.mlp_readout_layer import MLPReadout
from layers.Spec_layer import SpecLayer
from layers.Eigval_layer import EigvalLayer
from layers.Cheb_augmented_layer import ChebAugmentedLayer

# BW
from torch import norm as torch_norm

class ChebNet(nn.Module):
    def __init__(self, net_params, model='ChebNet'):
        super().__init__()
        in_dim_node = net_params['in_dim']  # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']
        self.k = net_params['k']

        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)  # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        # BW
        if model == 'ChebNet':
            layer = ChebLayer
        elif model == 'SpecFilters':
            layer = SpecLayer
        elif model == 'EigvalFilters':
            layer = EigvalLayer
        elif model == 'ChebAugmentedFilters':
            layer = ChebAugmentedLayer

        self.layers = nn.ModuleList([layer(hidden_dim, hidden_dim, self.k, F.relu, dropout,
                                               self.graph_norm, self.batch_norm, self.residual) for _ in
                                     range(n_layers - 1)])
        self.layers.append(
            layer(hidden_dim, out_dim, self.k, F.relu, dropout, self.graph_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, g, h, e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        #lambda_max = dgl.laplacian_lambda_max(g)
        lambda_max = [2] * g.batch_size
        # Cheb
        for conv in self.layers:
            h = conv(g, h, lambda_max)  # , lambda_max = [2]*128)

        # output
        h_out = self.MLP_layer(h)

        return h_out

    def loss(self, pred, label):
        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)
        
        verbose = False
        if verbose:
            print('target loss', loss)

        # BW: add regularization
        if self.l1_reg > 0.0:
            l1_loss = 0.0
            for param in self.parameters():
                l1_loss += torch_norm(param, p=1)
            if verbose:
                print('l1_loss', l1_loss)
            loss += self.l1_reg * l1_loss

        if self.l2_reg > 0.0:
            l2_loss = 0.0
            for param in self.parameters():
                l2_loss += torch_norm(param, p=2)
            if verbose:
                print('l2_loss', l2_loss)
            loss += self.l2_reg * l2_loss
        
        # BW: allow other regularization specific to the layer subtype
        if self.gen_reg > 0.0:
            reg_loss = 0.0
            for layer in self.layers:
                if hasattr(layer, 'general_regularization_loss'):
                    reg_loss += layer.general_regularization_loss()
            if verbose:
                print('reg_loss', reg_loss)
            loss += self.gen_reg * reg_loss

        if verbose:
            print()
        return loss
