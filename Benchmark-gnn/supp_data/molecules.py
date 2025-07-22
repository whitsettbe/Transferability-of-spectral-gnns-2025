import pandas as pd
import torch

def metis_import(out_dict, info_hash, eigInFiles,
        num_eigs, symmetrizeImportedEdges = True,
        fixMissingPhi1 = True, device='cpu'):
    
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
            
            # fetch the harmonics and normalize them
            signals = torch.tensor(eval(row.signals)).to(device)
            if fixMissingPhi1:
                signals = torch.cat((torch.ones(signals[0:1].shape), signals))
            signals = signals / signals.pow(2).sum(1).pow(0.5).view((-1,1))
            
            # Construct the graph laplacian from the edge_index and num_nodes.
            laplacian = torch.zeros((num_nodes, num_nodes), device=device)
            if edge_index.size(1) > 0:
                row, col = edge_index # should be symmetrized already
                laplacian[row, col] = -1
                laplacian += torch.diag(torch.sum(laplacian, dim=1))
                
            # Compute the Rayleigh quotients of the signals
            quots = torch.einsum('ez,nz,en->e', signals, laplacian, signals)[:num_eigs]

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