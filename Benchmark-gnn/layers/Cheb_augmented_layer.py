import dgl
import dgl.function as fn
import torch
import torch.nn as nn

"""
    Cheb
"""

class ChebAugmentedLayer(nn.Module):

    eig_dict = dict() # mapping (num_nodes, sorted_edge_tuples) to (eigenvalues (e), eigenvectors (nxe))
    num_eigs = None
    eigval_hidden_dim = None
    eigval_num_hidden_layer = None

    @staticmethod
    def _graph_hash(g):
        # Create hash from edges and number of nodes
        edges = g.edges()
        num_nodes = g.num_nodes()
        # Sort edges to ensure consistent ordering
        edge_tuple = tuple(sorted(zip(edges[0].tolist(), edges[1].tolist())))
        return hash((num_nodes, edge_tuple))
    
    @staticmethod
    def _get_eigenvectors(g):
        """Get eigenvectors and eigenvalues for the graph, caching them if not already available"""
        graph_key = ChebAugmentedLayer._graph_hash(g)
        if graph_key is None or graph_key not in ChebAugmentedLayer.eig_dict:
            # Compute eigenvectors here using pytorch and store them
            # NOTE: this uses L, not the normalized laplacian (maybe should be normalized?)
            adj = g.adjacency_matrix().to_dense()
            #if ChebAugmentedLayer.normalized_laplacian:
            #    d_12 = torch.pow(adj.sum(dim=1), -0.5).view(1,-1)
            #    laplacian = torch.eye(adj.size(0)) - d_12 * adj * d_12.T
            #else:
            laplacian = torch.diag(adj.sum(dim=1)) - adj
            torch_eigenvalues, torch_eigenvectors = torch.eig(laplacian, eigenvectors=True)

            # Sort on real component of eigenvalues, and remove complex part (should be 0)
            sort_indices = torch_eigenvalues[:, 0].argsort()
            torch_eigenvectors = torch_eigenvectors[:, sort_indices]
            torch_eigenvalues = torch_eigenvalues[:,0][sort_indices]

            # Remove high-end spectrum
            torch_eigenvectors = torch_eigenvectors[:, :ChebAugmentedLayer.num_eigs].to(g.device)
            torch_eigenvalues = torch_eigenvalues[:ChebAugmentedLayer.num_eigs].to(g.device)
            
            # Pad with trailing zeros
            if torch_eigenvectors.size(1) < ChebAugmentedLayer.num_eigs:
                num_missing = ChebAugmentedLayer.num_eigs - torch_eigenvectors.size(1)
                vec_padding = torch.zeros((torch_eigenvectors.size(0), num_missing), device=g.device)
                val_padding = torch.zeros((num_missing,), device=g.device)
                torch_eigenvectors = torch.cat((torch_eigenvectors, vec_padding), dim=1)
                torch_eigenvalues = torch.cat((torch_eigenvalues, val_padding), dim=0)

            # Save the computation results
            ChebAugmentedLayer.eig_dict[graph_key] = (torch_eigenvalues.detach(), torch_eigenvectors.detach())

        return ChebAugmentedLayer.eig_dict[graph_key]


    """
        Param: [in_dim, out_dim, k, activation, dropout, graph_norm, batch_norm, residual connection]
    """

    def __init__(
            self,
            in_dim,
            out_dim,
            k,
            activation,
            dropout,
            graph_norm,
            batch_norm,
            residual=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self._k = k
        self.linear = nn.Linear((k + 1) * in_dim, out_dim)

        if in_dim != out_dim:
            self.residual = False

        # BW
        eigval_layers = [nn.Linear(1, self.eigval_hidden_dim), nn.ReLU()]
        for i in range(1, self.eigval_num_hidden_layer):
            eigval_layers.append(nn.Linear(self.eigval_hidden_dim, self.eigval_hidden_dim))
            eigval_layers.append(nn.ReLU())
        eigval_layers.append(nn.Linear(self.eigval_hidden_dim, 1))
        self.eigval_filter = nn.Sequential(*eigval_layers)

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    # BW
    def _spectral_forward(self, g, feature):
        graphs = dgl.unbatch(g)
        features = torch.split(feature, [gr.num_nodes() for gr in graphs], dim=0)
        
        h_list = []
        for graph, feat in zip(graphs, features):
            evals,evecs = self._get_eigenvectors(graph)

            evals_filtered = self.eigval_filter(evals.view(-1,1))
            h = torch.einsum('ne,nf->ef', evecs, feat)
            h = evals_filtered * h
            h = torch.einsum('ne,ef->nf', evecs, h)
            h_list.append(h)
        
        return torch.cat(h_list, dim=0)



    def forward(self, g, feature, lambda_max=None):
        h_in = feature  # to be used for residual connection

        def unnLaplacian(feature, D_sqrt, graph):
            """ Operation D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feature * D_sqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return graph.ndata.pop('h') * D_sqrt

        with g.local_scope():
            D_sqrt = torch.pow(g.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feature.device)

            if lambda_max is None:
                try:
                    lambda_max = dgl.laplacian_lambda_max(g)
                except BaseException:
                    # if the largest eigonvalue is not found
                    lambda_max = [2]

            if isinstance(lambda_max, list):
                lambda_max = torch.Tensor(lambda_max).to(feature.device)
            if lambda_max.dim() == 1:
                lambda_max = lambda_max.unsqueeze(-1)  # (B,) to (B, 1)

            # broadcast from (B, 1) to (N, 1)
            lambda_max = dgl.broadcast_nodes(g, lambda_max)

            # X_0(f)
            Xt = X_0 = feature

            # X_1(f)
            if self._k > 1:
                re_norm = (2. / lambda_max).to(feature.device)
                h = unnLaplacian(X_0, D_sqrt, g)
                # print('h',h,'norm',re_norm,'X0',X_0)
                X_1 = - re_norm * h + X_0 * (re_norm - 1)

                Xt = torch.cat((Xt, X_1), 1)

            # Xi(x), i = 2...k
            for _ in range(2, self._k):
                h = unnLaplacian(X_1, D_sqrt, g)
                X_i = - 2 * re_norm * h + X_1 * 2 * (re_norm - 1) - X_0

                Xt = torch.cat((Xt, X_i), 1)
                X_1, X_0 = X_i, X_1

            # BW: append the eigenvalue filter as features
            Xt = torch.cat((Xt, self._spectral_forward(g, feature)), 1)

            # Put the Chebyschev polynomes as featuremaps
            h = self.linear(Xt)

        # if self.graph_norm:
        #    h = h * snorm_n  # normalize activation w.r.t. graph size

        if self.batch_norm:
            h = self.batchnorm_h(h)  # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h  # residual connection

        h = self.dropout(h)
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, k={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self._k)
