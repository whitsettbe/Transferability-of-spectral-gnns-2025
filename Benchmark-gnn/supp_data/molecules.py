import pandas as pd
import torch

def metis_import(out_dict, info_hash, eigInFiles,
        num_eigs, symmetrizeImportedEdges = True,
        fixMissingPhi1 = True, device='cpu', normalizedLaplacian = False,
        eigval_norm = '',
        extraOrtho = False):
    
    # load all stored eigenvectors!
    tables = [pd.read_csv(eigInFiles[split]) for split in ['train', 'test', 'val']
            if split in eigInFiles and eigInFiles[split] is not None]

    for table in tables:
        for row in table.itertuples():
            # fetch graph identifiers
            num_nodes = row.num_nodes
            edge_index = torch.tensor(eval(row.edge_index)).to(device) # eval isn't secure...
            if symmetrizeImportedEdges:
                edge_index = torch.cat((edge_index, torch.flip(edge_index, dims=(0,))), dim=1)
            
            # fetch the harmonics
            signals = torch.tensor(eval(row.signals), dtype=torch.float32).to(device)
            if fixMissingPhi1:
                signals = torch.cat((torch.ones(signals[0:1].shape), signals))
                
            # if requested, do Gram-Schmidt orthogonalization
            if extraOrtho:
                for i in range(signals.size(0)):
                    for j in range(i):
                        signals[i] -= torch.dot(signals[i], signals[j]) * signals[j]
                    norm = torch.norm(signals[i])
                    if norm > 0:
                        signals[i] = signals[i] / norm

            # normalize the signals
            signals = signals / signals.pow(2).sum(1).pow(0.5).view((-1,1))
            
            # Construct the graph laplacian from the edge_index and num_nodes.
            adj = torch.zeros((num_nodes, num_nodes), device=device)
            row, col = edge_index # should be symmetrized already
            adj[row, col] = 1
            D = torch.abs(torch.sum(adj, dim=1))
            D_12 = torch.pow(D, -0.5)
            laplacian = torch.diag(D) - adj
            if normalizedLaplacian:
                laplacian = D_12.view(-1,1) * laplacian * D_12.view(1,-1)
                
            # Compute the Rayleigh quotients of the signals, with normalization if specified
            quots = torch.einsum('ez,nz,en->e', signals, laplacian, signals)
            if eigval_norm == "scale(-1,1)_all":
                quots = quots / quots.max()
                quots = 2 * quots - 1
            elif eigval_norm == "scale(0,2)_all":
                quots = quots / quots.max()
                quots = 2 * quots
            quots = quots[:num_eigs]
            
            if eigval_norm == "scale(0,2)_sub":
                # Scale the Rayleigh quotients to [0, 2] range
                quots = quots / quots.max()
                quots = 2 * quots

            # first dim of signals is eigencomponents, second is nodes
            # want to store: nodes-by-eigencomponents
            key = info_hash(num_nodes, edge_index)
            signals = signals.T[:,:num_eigs].to(device)
            
            # pad the signals and Rayleigh quotients if necessary
            if signals.size(1) < num_eigs:
                signal_padding = torch.zeros((signals.size(0),
                        num_eigs - signals.size(1)), device=device)
                quot_padding = torch.zeros((num_eigs - quots.size(0),), device=device)
                signals = torch.cat((signals, signal_padding), dim=1)
                quots = torch.cat((quots, quot_padding), dim=0)
            out_dict[key] = (quots.clone().detach(), signals.clone().detach())