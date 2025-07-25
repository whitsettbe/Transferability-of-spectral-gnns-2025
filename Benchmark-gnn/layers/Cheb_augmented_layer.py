import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.parameters import param_weights_and_biases
from supp_data.molecules import metis_import

"""
    Cheb
"""

class ChebAugmentedLayer(nn.Module):

    eig_dict = dict() # mapping (num_nodes, sorted_edge_tuples) to (eigenvalues (e), eigenvectors (nxe))
    num_eigs = None
    subtype = "dense" # one of ["dense", "poly", "parallel", ETC. ETC.]
    eigval_norm = "" # can be "" or "scale(-1,1)_all" (scale to -1,1 before subsetting)
    bias_mode = "" # can be "", spatial, or spectral
    eigval_hidden_dim = None
    eigval_num_hidden_layer = None
    normalized_laplacian = None
    post_normalized = None
    eigmod = None
    eigInFiles = None
    fixMissingPhi1 = True # whether to add a constant 1 as the first eigenvector (phi_1)
    extraOrtho = False # whether to do Graham-Schmidt orthogonalization on the eigenvectors
    k_aug = None  # number of monomials for augmented filter

    @staticmethod
    def _graph_info_hash(num_nodes, edge_index):
        edge_tuple = tuple(sorted(zip(edge_index[0].tolist(), edge_index[1].tolist())))
        return hash((num_nodes, edge_tuple))

    @staticmethod
    def _graph_hash(g):
        # Create hash from edges and number of nodes
        edges = g.edges()
        num_nodes = g.num_nodes()
        return ChebAugmentedLayer._graph_info_hash(num_nodes, edges)
    
    @staticmethod
    def _get_eigenvectors(g):
        """Get eigenvectors and eigenvalues for the graph, caching them if not already available"""
        
        # Check if the graph is invalid or computation was already done
        graph_key = ChebAugmentedLayer._graph_hash(g)
        if graph_key is None:
            return None, None
        if graph_key in ChebAugmentedLayer.eig_dict:
            # Return cached eigenvalues and eigenvectors
            return ChebAugmentedLayer.eig_dict[graph_key]
        
        # Check if eigenvectors should be imported
        if ChebAugmentedLayer.eigmod == "import_csv" and graph_key not in ChebAugmentedLayer.eig_dict:
            metis_import(ChebAugmentedLayer.eig_dict, ChebAugmentedLayer._graph_info_hash, ChebAugmentedLayer.eigInFiles,
                         ChebAugmentedLayer.num_eigs, fixMissingPhi1=ChebAugmentedLayer.fixMissingPhi1,
                         device=g.device,
                         normalizedLaplacian=ChebAugmentedLayer.normalized_laplacian,
                         eigval_norm=ChebAugmentedLayer.eigval_norm,
                         extraOrtho=ChebAugmentedLayer.extraOrtho)
            if graph_key in ChebAugmentedLayer.eig_dict: # ideally this should always be true
                return ChebAugmentedLayer.eig_dict[graph_key]
        
        # Compute eigenvectors here using pytorch and store them
        # NOTE: this uses L, not the normalized laplacian (maybe should be normalized?)
        adj = g.adjacency_matrix().to_dense()
        if ChebAugmentedLayer.normalized_laplacian:
            d_12 = torch.pow(adj.sum(dim=1), -0.5).view(1,-1)
            laplacian = torch.eye(adj.size(0)) - d_12 * adj * d_12.T
        else:
            laplacian = torch.diag(adj.sum(dim=1)) - adj
        torch_eigenvalues, torch_eigenvectors = torch.eig(laplacian, eigenvectors=True)

        # Sort on real component of eigenvalues, and remove complex part (should be 0)
        sort_indices = torch_eigenvalues[:, 0].argsort()
        torch_eigenvectors = torch_eigenvectors[:, sort_indices]
        torch_eigenvalues = torch_eigenvalues[:,0][sort_indices]

        if ChebAugmentedLayer.eigval_norm == "scale(-1,1)_all":
            torch_eigenvalues = torch_eigenvalues / torch_eigenvalues.max()
            torch_eigenvalues = 2 * torch_eigenvalues - 1
        elif ChebAugmentedLayer.eigval_norm == "scale(0,2)_all":
            torch_eigenvalues = torch_eigenvalues / torch_eigenvalues.max()
            torch_eigenvalues = 2 * torch_eigenvalues

        # Remove high-end spectrum
        torch_eigenvectors = torch_eigenvectors[:, :ChebAugmentedLayer.num_eigs].to(g.device)
        torch_eigenvalues = torch_eigenvalues[:ChebAugmentedLayer.num_eigs].to(g.device)
        
        if ChebAugmentedLayer.eigval_norm == "scale(0,2)_sub":
            # Scale the eigenvalues to [0, 2] range
            torch_eigenvalues = torch_eigenvalues / torch_eigenvalues.max()
            torch_eigenvalues = 2 * torch_eigenvalues
        
        # Replace the eigenvectors with a random orthonormal basis for the eigenspace, if requested
        if ChebAugmentedLayer.eigmod == "rand_basis":
            # Generate random vectors
            r = torch.randn((torch_eigenvectors.size(0), ChebAugmentedLayer.num_eigs), device=g.device)
            # Project into the eigenspace
            r = torch_eigenvectors @ (torch_eigenvectors.T @ r)
            # Orthonormalize
            for i in range(r.size(1)):
                for j in range(i):
                    r[:, i] -= torch.dot(r[:, i], r[:, j]) * r[:, j]
                norm = torch.norm(r[:, i])
                if norm > 0:
                    r[:, i] = r[:, i] / norm
            torch_eigenvectors = r
                    
            # Generate eigenvalues as Rayleigh quotients
            quots = torch.einsum('ze,nz,ne->e', r, laplacian, r)
            if ChebAugmentedLayer.eigval_norm == "scale(-1,1)_all":
                quots = quots / quots.max()
                quots = 2 * quots - 1
            elif ChebAugmentedLayer.eigval_norm in ["scale(0,2)_all", "scale(0,2)_sub"]:
                quots = quots / quots.max()
                quots = 2 * quots
            torch_eigenvalues = quots[:ChebAugmentedLayer.num_eigs]
        
        # Pad with trailing zeros
        if torch_eigenvectors.size(1) < ChebAugmentedLayer.num_eigs:
            num_missing = ChebAugmentedLayer.num_eigs - torch_eigenvectors.size(1)
            vec_padding = torch.zeros((torch_eigenvectors.size(0), num_missing), device=g.device)
            val_padding = torch.zeros((num_missing,), device=g.device)
            torch_eigenvectors = torch.cat((torch_eigenvectors, vec_padding), dim=1)
            torch_eigenvalues = torch.cat((torch_eigenvalues, val_padding), dim=0)

        # Save the computation results and return
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

        # Build layers for constructing the spectral-domain filter
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        if self.subtype == 'dense':
            for n in range(self._k):
                self.weights.append(nn.Parameter(torch.randn(
                    (self.num_eigs, self.num_eigs, self.num_eigs, self.num_eigs))))
                self.biases.append(nn.Parameter(torch.zeros(
                    (self.num_eigs, self.num_eigs))))
        elif self.subtype == 'parallel':
            # 1 layer to expand
            self.weights.append(nn.Parameter(torch.randn(
                (self.num_eigs, self.num_eigs, self.num_eigs)
            )))
            self.biases.append(nn.Parameter(torch.zeros(
                (self.num_eigs, self.num_eigs, self.num_eigs)
            )))
            # k-2 layers for nonlinearity
            for n in range(1, self._k - 1):
                self.weights.append(nn.Parameter(torch.randn(
                    (self.num_eigs, self.num_eigs, self.num_eigs, self.num_eigs)
                )))
                self.biases.append(nn.Parameter(torch.zeros(
                    (self.num_eigs, self.num_eigs, self.num_eigs)
                )))
            # 1 to summarize
            self.weights.append(nn.Parameter(torch.randn(
                (self.num_eigs, self.num_eigs, self.num_eigs)
            )))
            self.biases.append(nn.Parameter(torch.zeros(
                (self.num_eigs, self.num_eigs)
            )))
        elif self.subtype == 'poly':
            self.weights.append(nn.Parameter(torch.randn(
                (self._k * self.num_eigs, self.num_eigs)
            )))
        elif self.subtype == 'poly_vec':
            # each output feature is a linear combination of input features which have passed through a polynomial of lambda
            self.weights.append(nn.Parameter(torch.randn(
                (self._k, self.in_channels, self.out_channels)
            )))
        elif self.subtype == 'cheb_vec':
            self.weights.append(nn.Parameter(torch.randn(
                (self._k, self.in_channels, self.out_channels)
            )))

        elif self.subtype == 'cheb02_vec':
            # Use PyTorch Linear-style initialization
            weight = nn.Parameter(torch.empty((self._k, self.in_channels, self.out_channels)))
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            self.weights.append(weight)
            
            # Initialize bias the same way as nn.Linear
            fan_in = self._k * self.in_channels  # fan_in for our weight tensor
            bound = 1 / math.sqrt(fan_in)
            bias = nn.Parameter(torch.empty((self.out_channels,)))
            nn.init.uniform_(bias, -bound, bound)
            self.biases.append(bias)

        elif self.subtype == 'sparse_vec':
            weight, bias = param_weights_and_biases(
                (1,), (self.eigval_hidden_dim,))
            self.weights.append(weight)
            self.biases.append(bias)
            for n in range(1, self.eigval_num_hidden_layer):
                weight, bias = param_weights_and_biases(
                    (self.eigval_hidden_dim,), (self.eigval_hidden_dim,))
                self.weights.append(weight)
                self.biases.append(bias)
            weight, bias = param_weights_and_biases(
                (self.eigval_hidden_dim,),
                (self.in_channels, self.out_channels))
            self.weights.append(weight)
            self.biases.append(bias)

            # extra bias entry for the signals
            self.biases.append(param_weights_and_biases(
                (self.in_channels,), (self.out_channels,))[1])
            
        elif self.subtype == 'rational_vec':
            # Rational function approximation of spectral filters
            # Each filter is a ratio of polynomials in eigenvalues
            numer_weights, bias = param_weights_and_biases(
                (self.k_aug, self.in_channels), (self.out_channels,))
            denom_weights, _ = param_weights_and_biases(
                (self.k_aug, self.in_channels), (self.out_channels,))
            self.numerator_weights = numer_weights
            self.denominator_weights = denom_weights
            self.biases.append(bias)
            
            # Add small constant to avoid division by zero
            self.epsilon = 1e-6
            
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    # Process a single graph (allows parallelism)
    def process_single_graph(self, graph, feat):
        
        evals,evecs = self._get_eigenvectors(graph)
        if self.post_normalized:
            adj = graph.adjacency_matrix().to_dense()
            d_12 = torch.pow(adj.sum(dim=1), -0.5).view(-1,1)

        # Build and apply a filter from the eigenvalues
        if self.subtype == 'dense':
            s = torch.diag(evals)
            for n in range(self._k):
                s = torch.einsum('ijkl,ij->kl',self.weights[n],s)+self.biases[n]
                if n < self._k - 1: # don't do this at the end
                    s = self.spectral_activation(s)

            # evecs shaped (num_nodes,num_eigs)
            # feat shaped (num_nodes,num_features)
            # e,i are size num_eigs; n,z are size num_nodes
            return torch.einsum('zi,ei,ne,nf->zf', evecs, s, evecs, feat)
                    
        elif self.subtype == 'parallel':
            # 1 to expand, k-2 for nonlinearity, 1 to summarize
            s = self.spectral_activation(torch.einsum('xyi,x->xyi',self.weights[0],evals)+self.biases[0])
            for n in range(1, self._k - 1):
                s = self.spectral_activation(torch.einsum('xyij,xyi->xyj',self.weights[n],s)+self.biases[n])
            s = torch.einsum('xyj,xyj->xy',self.weights[-1],s)+self.biases[-1]
            # NOTE: x is the dimension varying in which eigenvalue is used, which should be 
            #  the *summed* dimension when used in multiplication (as it is here)

            # evecs shaped (num_nodes,num_eigs)
            # feat shaped (num_nodes,num_features)
            # e,i are size num_eigs; n,z are size num_nodes
            return torch.einsum('zi,ei,ne,nf->zf', evecs, s, evecs, feat)
        
        elif self.subtype == 'poly':
            s = [torch.eye(evals.size(0)), torch.diag(evals)]
            for n in range(2, self._k):
                s.append(s[1] * s[-1])
            s = torch.cat(s, dim=0)
            s = torch.einsum('pe,pi->ei',self.weights[0],s)

            # evecs shaped (num_nodes,num_eigs)
            # feat shaped (num_nodes,num_features)
            # e,i are size num_eigs; n,z are size num_nodes
            return torch.einsum('zi,ei,ne,nf->zf', evecs, s, evecs, feat)

        elif self.subtype == 'poly_vec':
            s = [torch.ones(evals.shape), evals]
            for n in range(2, self._k):
                s.append(s[1] * s[-1])
            s = torch.stack(s, dim=1)
            s = torch.einsum('kio,ek->eio', self.weights[0], s)

            h = torch.einsum('ne,ni->ei', evecs, feat)
            h = torch.einsum('eio,ei->eo', s, h) # this step is big?
            h = torch.einsum('ne,eo->no', evecs, h)
            return h
        
        elif self.subtype == 'cheb_vec':
            s = [torch.ones(evals.shape), evals]
            for n in range(2, self._k):
                s.append(2 * s[1] * s[-1] - s[-2])
            s = torch.stack(s, dim=1)
            s = torch.einsum('kio,ek->eio', self.weights[0], s)

            h = torch.einsum('ne,ni->ei', evecs, feat)
            h = torch.einsum('eio,ei->eo', s, h) # this step is big?
            h = torch.einsum('ne,eo->no', evecs, h)
            return h

        elif self.subtype == 'cheb02_vec': # chebyshev recurrence shifted for 0 to 2
            s = [torch.ones(evals.shape), evals - 1]
            for n in range(2, self._k):
                s.append(2 * (evals - 1) * s[-1] - s[-2])
            s = torch.stack(s, dim=1)
            #s = torch.einsum('kio,ek->eio', self.weights[0], s)

            h = feat * d_12 if self.post_normalized else feat
            h = torch.einsum('ne,ni->ei', evecs, h)
            h = torch.einsum('ek,ei->eki', s, h) # broken up (should be faster now)
            h = torch.einsum('kio,eki->eo', self.weights[0], h) # ^
            h = torch.einsum('ne,eo->no', evecs, h)
            h = h * d_12 if self.post_normalized else h

            # include bias and save
            if self.bias_mode == 'spatial':
                h = h + self.biases[-1].view(1,-1)
            return h

        elif self.subtype == 'sparse_vec':
            s = self.spectral_activation(self.weights[0].view(-1,1) * evals.view(1,-1) \
                    + self.biases[0].view(-1,1))
            for n in range(1, self.eigval_num_hidden_layer - 1):
                s = self.spectral_activation(
                    torch.einsum('xy,xe->ye',self.weights[n],s) + self.biases[n].view(-1,1))
            #print(self.weights[self.eigval_num_hidden_layer].shape, s.shape)
            s = torch.einsum('xio,xe->ioe',self.weights[self.eigval_num_hidden_layer], s) \
                    + self.biases[self.eigval_num_hidden_layer].unsqueeze(2)
            
            h = feat * d_12 if self.post_normalized else feat
            h = torch.einsum('ne,ni->ei', evecs, feat)
            h = torch.einsum('ioe,ei->eo', s, h)
            h = torch.einsum('ne,eo->no', evecs, h)
            h = h * d_12 if self.post_normalized else h
            
            # include bias and save
            if self.bias_mode == 'spatial':
                h = h + self.biases[-1].view(1,-1)
            return h
            
        elif self.subtype == 'rational_vec':
            # Compute rational function filters
            eval_powers = [torch.ones_like(evals)]
            for k in range(1, self.k_aug):
                eval_powers.append(eval_powers[-1] * evals)
            eval_stack = torch.stack(eval_powers, dim=0)  # (k, num_eigs)
            
            # Compute numerator and denominator
            numerator = torch.einsum('kio,ke->ioe', self.numerator_weights, eval_stack)
            denominator = torch.einsum('kio,ke->ioe', self.denominator_weights, eval_stack)
            
            # Add epsilon to avoid division by zero and compute ratio
            s = numerator / (torch.abs(denominator) + self.epsilon)
            
            h = feat * d_12 if self.post_normalized else feat
            h = torch.einsum('ne,ni->ei', evecs, h)
            h = torch.einsum('ioe,ei->eo', s, h)
            h = torch.einsum('ne,eo->no', evecs, h)
            h = h * d_12 if self.post_normalized else h
            
            if self.bias_mode == 'spatial':
                h = h + self.biases[-1].view(1,-1)
            return h

    # BW
    def _spectral_forward(self, g, feature):
        graphs = dgl.unbatch(g)
        features = torch.split(feature, [gr.num_nodes() for gr in graphs], dim=0)
        
        h_list = []
        for graph, feat in zip(graphs, features):
            h_list.append(self.process_single_graph(graph, feat))

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
    
    def general_regularization_loss(self):
        """Compute regularization loss for spectral filters"""
        reg_loss = 0.0
        
        if self.subtype == 'rational_vec':
            # Stability regularization for rational functions
            reg_loss += torch.mean(torch.abs(self.denominator_weights))
            
        return reg_loss

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, k={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self._k)
