import torch
import networkx as nx
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import pickle

def get_double_spd(data, anchor_indices, max_dist):
    """
    Compute shortest path distances from multiple anchor nodes to all other nodes
    in an undirected graph using NetworkX.
    
    Args:
        data (torch_geometric.data.Data): Input graph data
        anchor_indices (torch.Tensor or list): Indices of anchor nodes (shape: [M])
    
    Returns:
        torch.Tensor: Tensor of shortest path distances (shape: [num_nodes, M])
    """
    # Convert PyG data to NetworkX undirected graph once
    G = to_networkx(data, to_undirected=True)
    num_nodes = data.num_nodes
    M = len(anchor_indices)
    
    # Initialize distance matrix with -1 (unreachable)
    distances = torch.full((num_nodes, M), max_dist, dtype=torch.long)

    # Convert to list if given as tensor
    if isinstance(anchor_indices, torch.Tensor):
        anchor_indices = anchor_indices.tolist()

    for i, anchor in enumerate(anchor_indices):
        if anchor not in G:
            raise ValueError(f"Anchor node {anchor} not found in graph")
            
        # Get shortest paths using BFS
        shortest_lengths = nx.single_source_shortest_path_length(G, anchor)
        
        # Fill distances for this anchor column
        for node, dist in shortest_lengths.items():
            distances[node, i] = dist if dist < max_dist else max_dist
    # print(distances)
    return distances

def pe_encoding_for_graph(
        args, graph, edge_label_index, edge_label, processed_pe_path=None,
    ):
    """
    With a given graph in dataset, do subgraph sampling and 
    then calculate the DSPD for the sampled subgraph.
    Args:
        args (argparse.Namespace): The arguments
        graph (torch_geometric.data.Data): The graph
        graph_name (str): The name of the graph
        edge_label_index (torch.Tensor): The edge label index
        edge_label (torch.Tensor): The edge label
        processed_pe_path (str): The path to save the DSPD per batch
    Return:
        loader: The loader with 'batch_size' for mini-batch training
        batch_dspd_list: The DSPDs of batches coming from the loader.
    """
    num_neighbors = -1
    path_exist = os.path.exists(processed_pe_path)
    

    ## If we do not use PE, just return the loader and an empty list
    if (not args.use_pe) or path_exist:
        ## The actual loader used in mini-batch training
        loader = LinkNeighborLoader(
            graph,
            num_neighbors=args.num_hops * [num_neighbors],
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            subgraph_type='bidirectional',
            disjoint=True,
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=args.num_workers,
        )
        if path_exist and args.use_pe:
            print("Found existing file of dspd_per_batch!")
            print(f"Loading from {processed_pe_path}")

            with open (processed_pe_path, 'rb') as fp:
                dspd_per_batch = pickle.load(fp)
        
        else:
            dspd_per_batch = [None] * edge_label.size(0)
        return loader, dspd_per_batch
    
    ## Create a LinkNeighborLoader for subgraph sampling.
    ## For each edge_label_index, we sample a 'num_hops' subgraph.
    ## NOTE: This loader is only used for PE calculation.
    loader = LinkNeighborLoader(
        graph,
        num_neighbors=args.num_hops * [num_neighbors],
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        subgraph_type='bidirectional',
        disjoint=True,
        batch_size=1, ## batch_size is always 1
        shuffle=False, 
        num_workers=args.num_workers,
    )

    dspd_per_subg = []
    gid_per_subg = []

    ## Calculate the SPD for each batch
    for subgraph in tqdm(
        loader, 
        desc=f"{graph.name}: Subgraph sampling and DSPD calculation"
    ):
        dspd_per_subg.append(
            get_double_spd(
                subgraph,
                ## src and dst nodes in edge_label_index are always 
                ## the first 2 nodes in the subgraph.
                anchor_indices=[0, 1], max_dist=args.max_dist,
            )
        )
        assert dspd_per_subg[-1].size(0) == subgraph.num_nodes
        gid_per_subg.append(subgraph.n_id)
        
    ## The actual loader used in mini-batch training
    loader = LinkNeighborLoader(
        graph,
        num_neighbors=args.num_hops * [num_neighbors],
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        subgraph_type='bidirectional',
        disjoint=True,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
    )

    dspd_per_batch = [None] * edge_label.size(0)
    
    ## match the DSPDs of subgraphs back to the data batches
    for b, batch in enumerate(
        tqdm(loader, desc='Matching back to batches', leave=False)
    ):
        batched_dspd = torch.empty(
            (batch.num_nodes, 2), dtype=torch.long).fill_(args.max_dist)
        ## For each batrch, we have:
        ## batch.edge_label.size(0) == batch.edge_label_index.size(1)
        ## batch.batch.max()+1 == batch.input_id.size(0) == \
        num_subgraphs = batch.input_id.size(0)

        for i in range(num_subgraphs):
            subg_node_mask = batch.batch == i
            ## global subgraph id is the id of the sampled 'edge_label_index'
            global_subg_id = batch.input_id[i]
            batched_dspd[subg_node_mask] = dspd_per_subg[global_subg_id]
        
        ## store the dspd for each batch
        dspd_per_batch[b] = batched_dspd

    ## save dspd_per_batch to file
    print(f"Saving dspd_per_batch to {processed_pe_path}")
    with open(processed_pe_path, 'wb') as fp:
        pickle.dump(dspd_per_batch, fp)

    return loader, dspd_per_batch

def dataset_sampling_and_pe_calculation(args, dataset):
    """ 
    Sampling subgraphs for each graph in dataset and 
    calculate the PE for each sampled subgraph.
    Args:
        args (argparse.Namespace): The arguments
        dataset (torch_geometric.data.InMemoryDataset): The dataset
    Return:
        train_loader, val_loader, test_loaders, 
        train_subgraph_dspd_list, valid_subgraph_dspd_list, test_subgraph
    """

    ## default training data come from the first dataset
    graph_idx = 0
    train_graph = dataset[graph_idx]
    ## get split for validation
    train_ind, val_ind = train_test_split(
        np.arange(train_graph.edge_label.size(0)), 
        test_size=0.2, shuffle=True, #stratify=stratify,
    )
    train_ind = torch.tensor(train_ind, dtype=torch.long)
    val_ind = torch.tensor(val_ind, dtype=torch.long)

    train_edge_label_index = train_graph.edge_label_index[:, train_ind]
    train_edge_label = train_graph.edge_label[train_ind]
    dspd_name = f'_h{args.num_hops}_seed{args.seed}_train.dspd'

    ## Create the dataloaders and cached DSPD for training dataset
    train_loader, train_dspd_list = pe_encoding_for_graph(
        args, train_graph,
        train_edge_label_index, train_edge_label, 
        dataset.processed_paths[graph_idx]+dspd_name,
    )

    val_edge_label_index = train_graph.edge_label_index[:, val_ind]
    val_edge_label = train_graph.edge_label[val_ind]
    dspd_name = f'_h{args.num_hops}_seed{args.seed}_val.dspd'

    ## Create the dataloaders and cached DSPD for validation dataset
    val_loader, val_dspd_list = pe_encoding_for_graph(
        args, train_graph,
        val_edge_label_index, val_edge_label, 
        dataset.processed_paths[graph_idx]+dspd_name,
    )

    test_dspd_dict = {}
    test_loaders = {}

    ## The remaining datasets are all used for testing
    for i in range(graph_idx+1, len(dataset.names)):
        test_graph = dataset[i]
        graph_name = test_graph.name
        dspd_name = f'_h{args.num_hops}_seed{args.seed}_test.dspd'

        ## Create the dataloaders and cached DSPD for each test dataset
        test_loaders[graph_name], test_dspd_dict[graph_name] = \
            pe_encoding_for_graph(
                args, test_graph,
                test_graph.edge_label_index, test_graph.edge_label,
                dataset.processed_paths[i]+dspd_name,
            )

    return (
        train_loader, val_loader, test_loaders,
        train_dspd_list, 
        val_dspd_list, 
        test_dspd_dict,
    )