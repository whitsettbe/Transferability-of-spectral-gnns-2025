import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Eigenvalue layer - learn a linear map on the signal by nonlinear computation from the Laplacian eigenvalues
"""

class EigvalLayer(nn.Module):
    eig_dict = dict() # mapping (num_nodes, sorted_edge_tuples) to (eigenvalues (e), eigenvectors (nxe))
    num_eigs = None
    subtype = "dense" # one of ["dense", "poly", "parallel", ETC. ETC.]
    eigval_norm = "" # can be "" or "scale(-1,1)_all" (scale to -1,1 before subsetting)

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
        graph_key = EigvalLayer._graph_hash(g)
        if graph_key is None or graph_key not in EigvalLayer.eig_dict:
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

            # Remove high-end spectrum
            torch_eigenvectors = torch_eigenvectors[:, :EigvalLayer.num_eigs].to(g.device)
            torch_eigenvalues = torch_eigenvalues[:EigvalLayer.num_eigs].to(g.device)
            
            # Pad with trailing zeros
            if torch_eigenvectors.size(1) < EigvalLayer.num_eigs:
                num_missing = EigvalLayer.num_eigs - torch_eigenvectors.size(1)
                vec_padding = torch.zeros((torch_eigenvectors.size(0), num_missing), device=g.device)
                val_padding = torch.zeros((num_missing,), device=g.device)
                torch_eigenvectors = torch.cat((torch_eigenvectors, vec_padding), dim=1)
                torch_eigenvalues = torch.cat((torch_eigenvalues, val_padding), dim=0)

            # Save the computation results
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
        self.weights = []
        self.biases = []
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
            self.weights.append(nn.Parameter(torch.randn(
                (self._k, self.in_channels, self.out_channels)
            )))


        # Pick an activation for the spectral construction
        self.spectral_activation = F.relu

        if in_dim != out_dim:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, feature, lambda_max=None):
        # NOTE: what was lambda_max intended for in the older architecture??

        h_in = feature  # to be used for residual connection

        graphs = dgl.unbatch(g)
        features = torch.split(feature, [gr.num_nodes() for gr in graphs], dim=0)
        
        h_list = []
        for graph, feat in zip(graphs, features):
            evals,evecs = self._get_eigenvectors(graph)

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
                h_list.append(torch.einsum('zi,ei,ne,nf->zf', evecs, s, evecs, feat))
                        
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
                h_list.append(torch.einsum('zi,ei,ne,nf->zf', evecs, s, evecs, feat))
            
            elif self.subtype == 'poly':
                s = [torch.eye(evals.size(0)), torch.diag(evals)]
                for n in range(2, self._k):
                    s.append(s[1] * s[-1])
                s = torch.cat(s, dim=0)
                s = torch.einsum('pe,pi->ei',self.weights[0],s)

                # evecs shaped (num_nodes,num_eigs)
                # feat shaped (num_nodes,num_features)
                # e,i are size num_eigs; n,z are size num_nodes
                h_list.append(torch.einsum('zi,ei,ne,nf->zf', evecs, s, evecs, feat))

            elif self.subtype == 'poly_vec':
                s = [torch.ones(evals.shape), evals]
                for n in range(2, self._k):
                    s.append(s[1] * s[-1])
                s = torch.stack(s, dim=1)
                s = torch.einsum('kio,ek->eio', self.weights[0], s)

                h = torch.einsum('ne,ni->ei', evecs, feat)
                h = torch.einsum('eio,ei->eo', s, h) # this step is big?
                h = torch.einsum('ne,eo->no', evecs, h)
                h_list.append(h)
            
            elif self.subtype == 'cheb_vec':
                s = [torch.ones(evals.shape), evals]
                for n in range(2, self._k):
                    s.append(2 * s[1] * s[-1] - s[-2])
                s = torch.stack(s, dim=1)
                s = torch.einsum('kio,ek->eio', self.weights[0], s)

                h = torch.einsum('ne,ni->ei', evecs, feat)
                h = torch.einsum('eio,ei->eo', s, h) # this step is big?
                h = torch.einsum('ne,eo->no', evecs, h)
                h_list.append(h)

            elif self.subtype == 'cheb02_vec': # chebyshev recurrence shifted for 0 to 1
                s = [torch.ones(evals.shape), evals - 1]
                for n in range(2, self._k):
                    s.append(2 * (evals - 1) * s[-1] - s[-2])
                s = torch.stack(s, dim=1)
                #s = torch.einsum('kio,ek->eio', self.weights[0], s)

                h = torch.einsum('ne,ni->ei', evecs, feat)
                h = torch.einsum('ek,ei->eki', s, h) # broken up (should be faster now)
                h = torch.einsum('kio,eki->eo', self.weights[0], h) # ^
                h = torch.einsum('ne,eo->no', evecs, h)
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

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, k={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self._k)
