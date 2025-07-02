import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecLayer(nn.Module):
    evec_dict = dict()
    num_eigs = None
    hidden_dim = None
    group_by = None # from ['eigen', 'features', 'none']
    biases = True
    
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
        """Get eigenvectors for the graph, caching them if not already available"""
        graph_key = SpecLayer._graph_hash(g)
        if graph_key is None or graph_key not in SpecLayer.evec_dict:
            # Compute eigenvectors here using pytorch and store them
            adj = g.adjacency_matrix().to_dense()
            laplacian = torch.diag(adj.sum(dim=1)) - adj
            torch_eigenvalues, torch_eigenvectors = torch.eig(laplacian, eigenvectors=True)
            torch_eigenvectors = torch_eigenvectors[:, torch_eigenvalues[:, 0].argsort()] # sort real components
            torch_eigenvectors = torch_eigenvectors[:, :SpecLayer.num_eigs].to(g.device)
            
            # Pad if needed, then store
            if torch_eigenvectors.size(1) < SpecLayer.num_eigs:
                padding = torch.zeros((torch_eigenvectors.size(0), SpecLayer.num_eigs - torch_eigenvectors.size(1)), device=g.device)
                torch_eigenvectors = torch.cat((torch_eigenvectors, padding), dim=1)
            SpecLayer.evec_dict[graph_key] = torch_eigenvectors
        return SpecLayer.evec_dict[graph_key]

    """
        Param: [in_dim, out_dim]
    """

    def __init__(self, input_dim, output_dim, dropout=0.0, graph_norm=True, batch_norm=True, residual=True, **kwargs):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        # self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual

        if self.in_channels != self.out_channels:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(self.out_channels)
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

        # choose whether biases can be trained away from zero
        biasBuilder = nn.Parameter if self.biases else (lambda x: x)

        assert(self.group_by in ['eigen', 'features', 'none'])
        if self.group_by == 'eigen':
            # linear layers mixing the features
            self.weights1 = nn.Parameter(torch.randn(
                (self.num_eigs, self.in_channels, self.hidden_dim)))
            self.biases1 = biasBuilder(torch.zeros(self.num_eigs, self.hidden_dim))
            self.hidden_activation = F.relu
            self.weights2 = nn.Parameter(torch.randn(
                (self.num_eigs, self.hidden_dim, self.out_channels)))
            self.biases2 = biasBuilder(torch.zeros(self.num_eigs, self.out_channels))

        elif self.group_by == 'features':
            assert(self.in_channels == self.out_channels)
            # linear layers mixing the eigen-information
            self.weights1 = nn.Parameter(torch.randn(
                (self.in_channels, self.num_eigs, self.hidden_dim)))
            self.biases1 = biasBuilder(torch.zeros(self.in_channels, self.hidden_dim))
            self.hidden_activation = F.relu
            self.weights2 = nn.Parameter(torch.randn(
                (self.in_channels, self.hidden_dim, self.num_eigs)))
            self.biases2 = biasBuilder(torch.zeros(self.in_channels, self.num_eigs))

        elif self.group_by == 'none':
            self.weights1 = nn.Parameter(torch.randn(
                (self.num_eigs, self.in_channels, self.hidden_dim, self.hidden_dim)))
            self.biases1 = biasBuilder(torch.zeros(self.hidden_dim, self.hidden_dim))
            self.hidden_activation = F.relu
            self.weights2 = nn.Parameter(torch.randn(
                (self.hidden_dim, self.hidden_dim, self.num_eigs, self.out_channels)))
            self.biases2 = biasBuilder(torch.zeros(self.num_eigs, self.out_channels))



    # def forward(self, g, feature, snorm_n, lambda_max=None):
    def forward(self, g, feature, e):
        #print(dgl.unbatch(g)[:5])
        #print(feature.shape)
        #print(0/0)

        h_in = feature  # to be used for residual connection
        
        graphs = dgl.unbatch(g)
        features = torch.split(feature, [gr.num_nodes() for gr in graphs], dim=0)
        
        h_list = []
        for graph, feat in zip(graphs, features):
            eigen = self._get_eigenvectors(graph)

            if self.group_by == 'eigen':
                spec_in = torch.einsum('ni,ne->ei', feat, eigen)
                spec_hidden = torch.einsum('eik,ei->ek', self.weights1, spec_in) + self.biases1
                spec_hidden = self.hidden_activation(spec_hidden)
                spec_out = torch.einsum('eko,ek->eo', self.weights2, spec_hidden) + self.biases2
            elif self.group_by == 'features':
                spec_in = torch.einsum('ni,ne->ie', feat, eigen)
                spec_hidden = torch.einsum('iek,ie->ik', self.weights1, spec_in) + self.biases1
                spec_hidden = self.hidden_activation(spec_hidden)
                spec_out = torch.einsum('ike,ik->ie', self.weights2, spec_hidden) + self.biases2
                spec_out = spec_out.T
            elif self.group_by == 'none':
                spec_in = torch.einsum('ni,ne->ei', feat, eigen)
                spec_hidden = torch.einsum('eijk,ei->jk', self.weights1, spec_in) + self.biases1
                spec_hidden = self.hidden_activation(spec_hidden)
                spec_out = torch.einsum('jkeo,jk->eo', self.weights2, spec_hidden) + self.biases2

            h_list.append(torch.einsum('eo,ne->no', spec_out, eigen))

        h = torch.cat(h_list, dim=0)

        if self.batch_norm:
            h = self.batchnorm_h(h)  # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h  # residual connection

        h = self.dropout(h)
        return h, e
