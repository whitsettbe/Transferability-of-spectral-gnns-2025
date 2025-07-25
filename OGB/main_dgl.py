import argparse
import os
import dgl
import numpy as np
from copy import deepcopy
import torch
import torch.optim as optim
from gnn_dgl import GNN
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator, collate_dgl
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from spec_layer import SpecLayer

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer, task_type, l1_reg_coeff, l2_reg_coeff):
    model.train()

    for step, (batch_graphs, batch_labels) in enumerate(tqdm(loader, desc="Iteration")):
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)
        batch_h = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device) 
        pred = model(batch_graphs, batch_h, batch_e)

        optimizer.zero_grad()

        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch_labels == batch_labels
        if "classification" in task_type:
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch_labels.to(torch.float32)[is_labeled])
        else:
            loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch_labels.to(torch.float32)[is_labeled])

        # include regularization terms (weight is loss per datum per total f(parameter))
        if l1_reg_coeff > 0:
            l1_mean = sum(p.abs().sum() for p in model.parameters())
            l1_mean /= sum(torch.tensor(p.shape).prod() for p in model.parameters())
            loss += l1_reg_coeff * l1_mean * pred.size(0)
        if l2_reg_coeff > 0:
            l2_mean = sum(p.pow(2).sum() for p in model.parameters())
            l2_mean /= sum(torch.tensor(p.shape).prod() for p in model.parameters())
            loss += l2_reg_coeff * l2_mean * pred.size(0)

        loss.backward()
        optimizer.step()


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, (batch_graphs, batch_labels) in enumerate(tqdm(loader, desc="Iteration")):
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)
        batch_h = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device)

        with torch.no_grad():
            pred = model(batch_graphs, batch_h, batch_e)

        y_true.append(batch_labels.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with DGL')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='Cheb_net',
                        help='GNN (default: Cheb_net)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--num_eigs', type=int, default=15,
                        help='number of eigenvectors to compute (default: 15)') # BW
    parser.add_argument('--hidden_dim', type=int, default=75,
                        help='size of the spectral filter hidden dimension (default: 75)') # BW
    parser.add_argument('--filter_grouping', type=str, default='eigen',
                        help='filters parallel-wise process "eigen" (default), "features", or "none" (fully-connected mode)') # BW
    parser.add_argument('--with_residual', dest='residual', action='store_true') # BW
    parser.add_argument('--no_residual', dest='residual', action='store_false') # BW
    parser.set_defaults(residual=True) # BW
    parser.add_argument('--with_biases', dest='biases', action='store_true') # BW
    parser.add_argument('--no_biases', dest='biases', action='store_false') # BW
    parser.set_defaults(biases=True) # BW
    parser.add_argument('--l1_reg', type=float, default=0,
                        help='weight of L1 regularization in the loss (default 0)') # BW
    parser.add_argument('--l2_reg', type=float, default=0,
                        help='weight of L2 regularization in the loss (default 0)') # BW

    
    args = parser.parse_args()

    # BW: print all arguments
    print(vars(args))

    # BW: establish seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    #BW: cuda is not available for this machine and this version of cuda
    #device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")  # Force CPU right now due to compatibility issues

    ### automatic dataloading and splitting
    dataset = DglGraphPropPredDataset(name=args.dataset)

    if not os.path.exists('results'):
        os.makedirs('results')
    writer = SummaryWriter(log_dir='results/' + args.filename + 'logs/' + args.dataset + '/' + args.gnn)

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_dgl, pin_memory = False)#BW: True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_dgl, pin_memory = True)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_dgl, pin_memory = True)

    if args.gnn == 'Spec_filters':
        SpecLayer.num_eigs = args.num_eigs  # Set the number of eigenvectors for SpecLayer
        SpecLayer.hidden_dim = args.hidden_dim
        SpecLayer.group_by = args.filter_grouping
        SpecLayer.biases = args.biases
    
    if args.gnn in ['gated-gcn', 'mlp', 'Cheb_net', 'Spec_filters']: # BW: Spec_filters
        model = GNN(gnn_type=args.gnn, num_tasks=dataset.num_tasks, num_layer=args.num_layer,
                    emb_dim=args.emb_dim, dropout=args.dropout, batch_norm=True,
                    residual=args.residual, graph_pooling="mean")
        model.to(device)
    else:
        raise ValueError('Invalid GNN type')

    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    valid_curve = []
    test_curve = []
    train_curve = []
    best_valid = None # BW
    best_model_state = None # BW

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type, args.l1_reg, args.l2_reg)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        writer.add_scalar('Val', valid_perf[dataset.eval_metric], epoch)
        writer.add_scalar('Test', test_perf[dataset.eval_metric], epoch)
        writer.add_scalar('Train', train_perf[dataset.eval_metric], epoch)

        if 'classification' in dataset.task_type: # BW
            if best_valid is None or valid_curve[-1] > best_valid:
                best_valid = valid_curve[-1]
                best_model_state = deepcopy(model.state_dict())
        else:
            if best_valid is None or valid_curve[-1] < best_valid:
                best_valid = valid_curve[-1]
                best_model_state = deepcopy(model.state_dict())

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.filename == '':
        torch.save({
            'Val': valid_curve[best_val_epoch],
            'Test': test_curve[best_val_epoch],
            'Train': train_curve[best_val_epoch],
            'BestTrain': best_train,
            'Model': best_model_state # BW
        }, args.filename)

    writer.add_scalar('Best Val', valid_curve[best_val_epoch], best_val_epoch)
    writer.add_scalar('Best Test', test_curve[best_val_epoch], best_val_epoch)
    writer.add_scalar('Best Train', train_curve[best_val_epoch], best_val_epoch)
    writer.add_scalar('BestTrain', best_train)
    writer.close()


if __name__ == "__main__":
    main()
