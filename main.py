import argparse
import torch
import numpy as np
from sram_dataset import performat_SramDataset, adaption_for_sgrl
from downstream_train import downstream_train
import os
import random
from sgrl_train import sgrl_train

if __name__ == "__main__":
    # STEP 0: Parse Arguments ======================================================================= #
    parser = argparse.ArgumentParser(description="CircuitGPS_simple")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--dataset", type=str, default="ssram+digtime+timing_ctrl+array_128_32_8t", help="Names of datasets.")
    parser.add_argument("--add_tar_edge", type=int, default=0, help="0 or 1. Inject target edges into the graph.")
    # SGRL arguments
    parser.add_argument('--sgrl', type=int, default=0, help='Enable contrastive learning, i.e., SGRL.')
    parser.add_argument('--e1_lr', type=float, default=1e-6, help='Learning rate for online encoder in SGRL.')
    parser.add_argument('--e2_lr', type=float, default=2e-7, help='Learning rate for target encoder in SGRL.')
    parser.add_argument('--momentum', type=float, default=0.99, help='EMA')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight_decay')
    parser.add_argument('--cl_epochs', type=int, default=800, help='cl_epochs')
    parser.add_argument('--cl_hid_dim', type=int, default=144, help='hidden_dim for contrastive learning')
    # parser.add_argument('--trials', type=int, default=20, help='trials')
    # CirGPS arguments
    parser.add_argument("--task", type=str, default="classification", help="Task type. 'classification' or 'regression'.")
    parser.add_argument("--loss", type=str, default='mse', choices=['mse', 'gai', 'bmc', 'bni', 'lds'], help="The loss function. Could be 'mse', 'bmc', or 'gai'.")
    # Balanced MSE setting for GAI implementation
    parser.add_argument("--noise_sigma", type=float, default=0.0001, help="The simga_noise of Balanced MSE (EQ 3.6).")
    # LDS setting
    parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
    parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size: should be odd number. 5 0r 9.')
    parser.add_argument('--lds_sigma', type=float, default=0.02, help='LDS gaussian/laplace kernel sigma. 1 or 2.')

    parser.add_argument("--use_pe", type=int, default=0, help="Positional encoding. Defualt: True.")
    parser.add_argument("--num_hops", type=int, default=4, help="Number of hops in subgraph sampling.")
    parser.add_argument("--max_dist", type=int, default=350, help="The max values in DSPD.")
    parser.add_argument("--num_workers", type=int, default=8, help="The number of workers in data loaders.")
    parser.add_argument("--gpu", type=int, default=1, help="GPU index. Default: -1, using cpu.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--model", type=str, default='sage', help="The gnn model. Could be 'clustergcn', 'resgatedgcn', 'gat', 'gcn', 'sage', 'gine'.")
    parser.add_argument("--num_gnn_layers", type=int, default=5, help="Number of GNN layers.")
    parser.add_argument("--num_head_layers", type=int, default=2, help="Number of head layers.")
    parser.add_argument("--hid_dim", type=int, default=144, help="Hidden layer dim.")
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout for neural networks.')
    parser.add_argument('--use_bn', type=int, default=0, help='0 or 1. Batch norm for neural networks.')
    parser.add_argument('--act_fn', default='tanh', help='Activation function')
    parser.add_argument('--src_dst_agg', type=str, default='concat', help='The way to aggregate nodes. Can be `concat` or `add` or `pooladd` or `poolmean`.')
    parser.add_argument('--use_stats', type=int, default=1, help='0 or 1. Circuit statistics features.')
    args = parser.parse_args()

    # Syncronize all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
        print('Using GPU: {}'.format(args.gpu))
    else:
        device = torch.device("cpu")

    print(f"============= PID = {os.getpid()} ============= ")
    print(args)

    # STEP 1: Load Dataset =================================================================== #
    dataset = performat_SramDataset(
        name=args.dataset, 
        dataset_dir='./datasets/', 
        add_target_edges=args.add_tar_edge,
        neg_edge_ratio=0.5,
        to_undirected=True,
        sample_rates=1.0,
        task_type=args.task,
    )

    # STEP 2-3: If you do graph contrastive learning, you should add the code here =========== #
    if args.sgrl == 1:
        cl_embeds = sgrl_train(args, dataset, device)
    else:
        cl_embeds = None

    # STEP 4: Training Epochs ================================================================ #
    # No graph contrastive learning, no initail embeddings
    # embeds = torch.zeros(( train_graph.num_nodes, hid_dim))

    downstream_train(args, dataset, device, cl_embeds)