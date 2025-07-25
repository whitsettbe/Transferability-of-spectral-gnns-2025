"""
    IMPORTING LIBS
"""

import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.molecules_graph_regression.ChebNet import ChebNet  # import the ChebNet GNN
from data.data import LoadData  # import dataset
from layers.Spec_layer import SpecLayer # BW
from layers.Eigval_layer import EigvalLayer # BW
from layers.Cheb_augmented_layer import ChebAugmentedLayer # BW

"""
    GPU Setup
"""


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""


def view_model_param(MODEL_NAME, net_params):
    model = ChebNet(net_params, MODEL_NAME) # BW
    total_param = 0
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""


def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    t0 = time.time()
    per_epoch_time = []

    DATASET_NAME = dataset.name

    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # Write the network and optimization hyper-parameters in folder config/
    modelInfo = """Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format(
            DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param'])
    with open(write_config_file + '.txt', 'w') as f:
        f.write(modelInfo)
    print(modelInfo)

    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    # BW: Save global parameters required by SpecLayer
    SpecLayer.num_eigs = net_params.get('num_eigs', 15)  # Set the number of eigenvectors for SpecLayer
    SpecLayer.hidden_dim = net_params.get('hidden_dim', 150)
    SpecLayer.group_by = net_params.get('filter_grouping', "features")
    SpecLayer.biases = net_params.get('biases',True)

    # BW: Save global parameters required by EigvalLayer
    EigvalLayer.num_eigs = net_params.get('num_eigs',15)
    EigvalLayer.subtype = net_params.get('subtype', 'dense')
    EigvalLayer.normalized_laplacian = net_params.get('normalized_laplacian', False)
    EigvalLayer.post_normalized = net_params.get('post_normalized', False)
    EigvalLayer.eigval_norm = net_params.get('eigval_norm', '')
    EigvalLayer.bias_mode = net_params.get('bias_mode', '')
    EigvalLayer.eigval_hidden_dim = net_params.get('eigval_hidden_dim', 10)
    EigvalLayer.eigval_num_hidden_layer = net_params.get('eigval_num_hidden_layer', 3)
    EigvalLayer.eigmod = net_params.get('eigmod', '')
    EigvalLayer.eigInFiles = net_params.get('eigInFiles', dict())
    EigvalLayer.fixMissingPhi1 = net_params.get('fixMissingPhi1', True)
    EigvalLayer.extraOrtho = net_params.get('extraOrtho', False)

    # BW: Save global parameters required by ChebAugmentedLayer
    ChebAugmentedLayer.num_eigs = net_params.get('num_eigs', 15)
    ChebAugmentedLayer.subtype = net_params.get('subtype', 'dense')
    ChebAugmentedLayer.normalized_laplacian = net_params.get('normalized_laplacian', False)
    ChebAugmentedLayer.post_normalized = net_params.get('post_normalized', False)
    ChebAugmentedLayer.eigval_norm = net_params.get('eigval_norm', '')
    ChebAugmentedLayer.bias_mode = net_params.get('bias_mode', '')
    ChebAugmentedLayer.eigval_hidden_dim = net_params.get('eigval_hidden_dim', 10)
    ChebAugmentedLayer.eigval_num_hidden_layer = net_params.get('eigval_num_hidden_layer', 3)
    ChebAugmentedLayer.eigmod = net_params.get('eigmod', '')
    ChebAugmentedLayer.eigInFiles = net_params.get('eigInFiles', dict())
    ChebAugmentedLayer.fixMissingPhi1 = net_params.get('fixMissingPhi1', True)
    ChebAugmentedLayer.extraOrtho = net_params.get('extraOrtho', False)
    ChebAugmentedLayer.k_aug = net_params.get('k_aug', 4)  # number of monomials for augmented filter

    # BW: Inform ChebNet of the regularization weights
    ChebNet.l1_reg = net_params.get('l1_reg', 0.0)
    ChebNet.l2_reg = net_params.get('l2_reg', 0.0)
    ChebNet.gen_reg = net_params.get('gen_reg', 0.0)

    # BW
    model = ChebNet(net_params, model=MODEL_NAME)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_MAEs, epoch_val_MAEs = [], []

    # batching exception for Diffpool
    drop_last = True if MODEL_NAME == 'DiffPool' else False

    # import train functions for all other GNNs
    from train.train_molecules_graph_regression import train_epoch_sparse as train_epoch, \
        evaluate_network_sparse as evaluate_network

    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last,
                              collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last,
                            collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last,
                             collate_fn=dataset.collate)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, epoch_train_mae, optimizer = train_epoch(model, optimizer, device, train_loader,
                                                                           epoch)

                epoch_val_loss, epoch_val_mae = evaluate_network(model, device, val_loader, epoch)
                _, epoch_test_mae = evaluate_network(model, device, test_loader, epoch)

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_MAEs.append(epoch_train_mae)
                epoch_val_MAEs.append(epoch_val_mae)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_mae', epoch_train_mae, epoch)
                writer.add_scalar('val/_mae', epoch_val_mae, epoch)
                writer.add_scalar('test/_mae', epoch_test_mae, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_MAE=epoch_train_mae, val_MAE=epoch_val_mae,
                              test_MAE=epoch_test_mae)

                per_epoch_time.append(time.time() - start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch - 1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time() - t0 > params['max_time'] * 3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    _, test_mae = evaluate_network(model, device, test_loader, epoch)
    _, train_mae = evaluate_network(model, device, train_loader, epoch)
    print("Test MAE: {:.4f}".format(test_mae))
    print("Train MAE: {:.4f}".format(train_mae))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST MAE: {:.4f}\nTRAIN MAE: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n""" \
                .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                        test_mae, train_mae, epoch, (time.time() - t0) / 3600, np.mean(per_epoch_time)))


def main():
    """
        USER CONTROLS
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")
    parser.add_argument('--pos_enc', help="Please give a value for pos_enc")
    parser.add_argument('--num_eigs', type=int,
                        help='number of eigenvectors to compute') # BW
    parser.add_argument('--filter_grouping', type=str,
                        help='filters parallel-wise process "features" (default), "eigen", or "none" (fully-connected mode)') # BW
    parser.add_argument('--with_biases', dest='biases', action='store_true') # BW
    parser.add_argument('--no_biases', dest='biases', action='store_false') # BW
    #parser.set_defaults(biases=True) # BW
    parser.add_argument('--l1_reg', type=float,
                        help='weight of L1 regularization in the loss') # BW
    parser.add_argument('--l2_reg', type=float,
                        help='weight of L2 regularization in the loss') # BW
    parser.add_argument('--subtype', type=str,
                        help='subtype for the eigenvalue-based filter') # BW
    parser.add_argument('--normalized_laplacian', type=bool,
                        help='whether to use normalized laplacian') # BW
    parser.add_argument('--post_normalized', type=bool,
                        help='whether to normalize features separately from laplacian calculation') # BW
    parser.add_argument('--eigval_norm', type=str,
                        help='normalization of the eigenvalues') # BW
    parser.add_argument('--bias_mode', type=str,
                        help='type of bias to include in kernel construction modes') # BW
    parser.add_argument('--eigval_hidden_dim', type=int,
                        help='hidden dimension of eigenvalue filter') # BW
    parser.add_argument('--eigval_num_hidden_layer', type=int,
                        help='number of hidden layers of eigenvalue filter') # BW
    parser.add_argument('--gen_reg', type=float,
                        help='weight of general regularization in the loss') # BW
    parser.add_argument('--eigmod', type=str,
                        help='modifications to eigenvectors (including imports)') # BW
    parser.add_argument('--eigTrainFile', type=str,
                        help='file with eigenvectors for training') # BW
    parser.add_argument('--eigTestFile', type=str,
                        help='file with eigenvectors for testing') # BW
    parser.add_argument('--eigValFile', type=str,
                        help='file with eigenvectors for validation') # BW
    parser.add_argument('--fixMissingPhi1', type=bool,
                        help='whether to add a constant 1 as the first eigenvector (phi_1)') # BW
    parser.add_argument('--extraOrtho', type=bool,
                        help='whether to do extra orthogonalization for imported eigenvectors') # BW
    parser.add_argument('--k_aug', type=int,
                        help='number of monomials for augmented filter') # BW
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)

    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated == 'True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm == 'True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop == 'True' else False

    # BW
    if args.num_eigs is not None:
        net_params['num_eigs'] = args.num_eigs
    if args.filter_grouping is not None:
        net_params['filter_grouping'] = args.filter_grouping
    if args.biases is not None:
        net_params['biases'] = args.biases
    if args.l1_reg is not None:
        net_params['l1_reg'] = args.l1_reg
    if args.l2_reg is not None:
        net_params['l2_reg'] = args.l2_reg
    if args.subtype is not None:
        net_params['subtype'] = args.subtype
    if args.normalized_laplacian is not None:
        net_params['normalized_laplacian'] = args.normalized_laplacian
    if args.post_normalized is not None:
        net_params['post_normalized'] = args.post_normalized
    if args.eigval_norm is not None:
        net_params['eigval_norm'] = args.eigval_norm
    if args.bias_mode is not None:
        net_params['bias_mode'] = args.biases
    if args.eigval_hidden_dim is not None:
        net_params['eigval_hidden_dim'] = args.eigval_hidden_dim
    if args.eigval_num_hidden_layer is not None:
        net_params['eigval_num_hidden_layer'] = args.eigval_num_hidden_layer
    if args.gen_reg is not None:
        net_params['gen_reg'] = args.gen_reg
    if args.eigmod is not None:
        net_params['eigmod'] = args.eigmod
    if args.eigTrainFile is not None:
        net_params['eigInFiles'] = {
            'train': args.eigTrainFile,
            'test': net_params.get('eigTestFile', None),
            'val': net_params.get('eigValFile', None)
        }
    if args.eigTestFile is not None:
        net_params['eigInFiles'] = {
            'train': net_params.get('eigTrainFile', None),
            'test': args.eigTestFile,
            'val': net_params.get('eigValFile', None)
        }
    if args.eigValFile is not None:
        net_params['eigInFiles'] = {
            'train': net_params.get('eigTrainFile', None),
            'test': net_params.get('eigTestFile', None),
            'val': args.eigValFile
        }
    if args.fixMissingPhi1 is not None:
        net_params['fixMissingPhi1'] = args.fixMissingPhi1
    if args.extraOrtho is not None:
        net_params['extraOrtho'] = args.extraOrtho
    if args.k_aug is not None:
        net_params['k_aug'] = args.k_aug

    # ZINC
    net_params['num_atom_type'] = dataset.num_atom_type
    net_params['num_bond_type'] = dataset.num_bond_type

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + str(params['seed']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + str(params['seed']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + str(params['seed']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + str(params['seed']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = -1 #view_model_param(MODEL_NAME, net_params) # too complicated to initialize just for this calculation
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)


main()
