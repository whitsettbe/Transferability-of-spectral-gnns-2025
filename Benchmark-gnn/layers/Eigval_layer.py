import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.parameters import param_weights_and_biases
from supp_data.molecules import metis_import

"""
    Eigenvalue layer - learn a linear map on the signal by nonlinear computation from the Laplacian eigenvalues
"""

class EigvalLayer(nn.Module):
    eig_dict = dict() # mapping (num_nodes, sorted_edge_tuples) to (eigenvalues (e), eigenvectors (nxe))
    num_eigs = None
    subtype = "dense" # one of ["dense", "poly", "parallel", ETC. ETC.]
    eigval_norm = "" # can be "" or "scale(-1,1)_all" (scale to -1,1 before subsetting)
    bias_mode = "" # can be "", spatial, or spectral
    normalized_laplacian = None
    post_normalized = None
    eigmod = None
    eigInFiles = None
    fixMissingPhi1 = True # whether to add a constant 1 as the first eigenvector (phi_1)
    extraOrtho = False # whether to do Graham-Schmidt orthogonalization on the eigenvectors
    
    @staticmethod
    def _graph_info_hash(num_nodes, edge_index):
        edge_tuple = tuple(sorted(zip(edge_index[0].tolist(), edge_index[1].tolist())))
        return hash((num_nodes, edge_tuple))

    @staticmethod
    def _graph_hash(g):
        # Create hash from edges and number of nodes
        edges = g.edges()
        num_nodes = g.num_nodes()
        return EigvalLayer._graph_info_hash(num_nodes, edges)
    
    @staticmethod
    def _get_eigenvectors(g):
        """Get eigenvectors and eigenvalues for the graph, caching them if not already available"""
        
        # Check if the graph is invalid or computation was already done
        graph_key = EigvalLayer._graph_hash(g)
        if graph_key is None:
            return None, None
        if graph_key in EigvalLayer.eig_dict:
            # Return cached eigenvalues and eigenvectors
            return EigvalLayer.eig_dict[graph_key]
        
        # Check if eigenvectors should be imported
        if EigvalLayer.eigmod == "import_csv" and graph_key not in EigvalLayer.eig_dict:
            metis_import(EigvalLayer.eig_dict, EigvalLayer._graph_info_hash, EigvalLayer.eigInFiles,
                         EigvalLayer.num_eigs, fixMissingPhi1=EigvalLayer.fixMissingPhi1,
                         device=g.device,
                         normalizedLaplacian=EigvalLayer.normalized_laplacian,
                         eigval_norm=EigvalLayer.eigval_norm,
                         extraOrtho=EigvalLayer.extraOrtho)
            if graph_key in EigvalLayer.eig_dict: # ideally this should always be true
                return EigvalLayer.eig_dict[graph_key]
        
        # Compute eigenvectors here using pytorch and store them
        # NOTE: this uses L, not the normalized laplacian (maybe should be normalized?)
        adj = g.adjacency_matrix().to_dense()
        if EigvalLayer.normalized_laplacian:
            d_12 = torch.pow(adj.sum(dim=1), -0.5).view(1,-1)
            laplacian = torch.eye(adj.size(0)) - d_12 * adj * d_12.T
        else:
            laplacian = torch.diag(adj.sum(dim=1)) - adj
        torch_eigenvalues, torch_eigenvectors = torch.eig(laplacian, eigenvectors=True)

        # Sort on real component of eigenvalues, and remove complex part (should be 0)
        sort_indices = torch_eigenvalues[:, 0].argsort()
        torch_eigenvectors = torch_eigenvectors[:, sort_indices]
        torch_eigenvalues = torch_eigenvalues[:,0][sort_indices]

        if EigvalLayer.eigval_norm == "scale(-1,1)_all":
            torch_eigenvalues = torch_eigenvalues / torch_eigenvalues.max()
            torch_eigenvalues = 2 * torch_eigenvalues - 1
        elif EigvalLayer.eigval_norm == "scale(0,2)_all":
            torch_eigenvalues = torch_eigenvalues / torch_eigenvalues.max()
            torch_eigenvalues = 2 * torch_eigenvalues

        # Remove high-end spectrum
        torch_eigenvectors = torch_eigenvectors[:, :EigvalLayer.num_eigs].to(g.device)
        torch_eigenvalues = torch_eigenvalues[:EigvalLayer.num_eigs].to(g.device)
        
        if EigvalLayer.eigval_norm == "scale(0,2)_sub":
            # Scale the eigenvalues to [0, 2] range
            torch_eigenvalues = torch_eigenvalues / torch_eigenvalues.max()
            torch_eigenvalues = 2 * torch_eigenvalues
        
        # Replace the eigenvectors with a random orthonormal basis for the eigenspace, if requested
        if EigvalLayer.eigmod == "rand_basis":
            # Generate random vectors
            r = torch.randn((torch_eigenvectors.size(0), EigvalLayer.num_eigs), device=g.device)
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
            if EigvalLayer.eigval_norm == "scale(-1,1)_all":
                quots = quots / quots.max()
                quots = 2 * quots - 1
            elif EigvalLayer.eigval_norm in ["scale(0,2)_all", "scale(0,2)_sub"]:
                quots = quots / quots.max()
                quots = 2 * quots
            torch_eigenvalues = quots[:EigvalLayer.num_eigs]
        
        # Pad with trailing zeros
        if torch_eigenvectors.size(1) < EigvalLayer.num_eigs:
            num_missing = EigvalLayer.num_eigs - torch_eigenvectors.size(1)
            vec_padding = torch.zeros((torch_eigenvectors.size(0), num_missing), device=g.device)
            val_padding = torch.zeros((num_missing,), device=g.device)
            torch_eigenvectors = torch.cat((torch_eigenvectors, vec_padding), dim=1)
            torch_eigenvalues = torch.cat((torch_eigenvalues, val_padding), dim=0)

        # Save the computation results and return
        EigvalLayer.eig_dict[graph_key] = (torch_eigenvalues.detach(), torch_eigenvectors.detach())
        return EigvalLayer.eig_dict[graph_key]

    """
        Param: [in_dim, out_dim, k, activation, dropout, graph_norm, batch_norm, residual connection]
    """
    # k reinterpreted as DEPTH of transformation predictor
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

        # Build layers for constructing the spectral-domain filter
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        if self.subtype == 'cheb02_vec':
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
    
        elif self.subtype == 'rational_vec':
            # Rational function approximation of spectral filters
            # Each filter is a ratio of polynomials in eigenvalues
            numer_weights, bias = param_weights_and_biases(
                (self._k, self.in_channels), (self.out_channels,))
            denom_weights, _ = param_weights_and_biases(
                (self._k, self.in_channels), (self.out_channels,))
            self.numerator_weights = numer_weights
            self.denominator_weights = denom_weights
            self.biases.append(bias)
            
            # Add small constant to avoid division by zero
            self.epsilon = 1e-6
        
        elif self.subtype == 'parallel_dense_simp':
            self.weights.append(param_weights_and_biases(
                (1,), (self.num_eigs, self.in_channels))[0])
            self.linear = nn.Linear(self.in_channels, self.out_channels)


        # Pick an activation for the spectral construction
        self.spectral_activation = F.relu

        if in_dim != out_dim:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        
    # Process a single graph (allows parallelism)
    def process_single_graph(self, graph, feat):
        
        evals,evecs = self._get_eigenvectors(graph)
        if self.post_normalized:
            adj = graph.adjacency_matrix().to_dense()
            d_12 = torch.pow(adj.sum(dim=1), -0.5).view(-1,1)

        if self.subtype == 'cheb02_vec': # chebyshev recurrence shifted for 0 to 2
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
    
        elif self.subtype == 'rational_vec':
            # Compute rational function filters
            eval_powers = [torch.ones_like(evals)]
            for k in range(1, self._k):
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
        
        elif self.subtype == 'parallel_dense_simp':
            
            h = feat * d_12 if self.post_normalized else feat
            h = torch.einsum('ne,ni->ei', evecs, h)
            h = self.weights[0].squeeze(0) * h
            #h = torch.einsum('e,ei->ei', self.weights[0].view(-1), h)
            h = torch.einsum('ne,ei->ni', evecs, h)
            h = h * d_12 if self.post_normalized else h
            h = self.linear(h) # mix features
            return h
        

    def forward(self, g, feature, lambda_max=None):
        # NOTE: what was lambda_max intended for in the older architecture??

        h_in = feature  # to be used for residual connection

        graphs = dgl.unbatch(g)
        features = torch.split(feature, [gr.num_nodes() for gr in graphs], dim=0)
        
        h_list = []
        for graph, feat in zip(graphs, features):
            h = self.process_single_graph(graph, feat)
            h_list.append(h)
            

        # Recompile the filtered signals into a batch
        h = torch.cat(h_list, dim=0)

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
            
        elif self.subtype == 'parallel_dense_simp':
            # Regularize the filtering layer, not the fully-connected layer
            reg_loss += torch.mean(torch.abs(self.weights[0]))
            
        return reg_loss

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, k={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self._k)
