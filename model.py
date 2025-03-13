import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ResGatedGraphConv, GINConv, ChebConv, GINEConv, ClusterGCNConv, SSGConv
from torch_geometric.nn.models.mlp import MLP

NET = 0
DEV = 1
PIN = 2

class GraphHead(nn.Module):
    """ GNN head for graph-level prediction.

    Implementation adapted from the transductive GraphGPS.

    Args:
        hidden_dim (int): Hidden features' dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        num_layers (int): Number of layers of GNN model
        layers_post_mp (int): number of layers of head MLP
        use_bn (bool): whether to use batch normalization
        drop_out (float): dropout rate
        activation (str): activation function
        src_dst_agg (str): the way to aggregate src and dst nodes, which can be 'concat' or 'add' or 'pool'
    """
    def __init__(self, hidden_dim, dim_out, num_layers=2, num_head_layers=2, 
                 use_bn=False, drop_out=0.0, activation='relu', 
                 src_dst_agg='concat', use_pe=False, max_dist=400, 
                 task='classification', use_stats=False):
        super().__init__()
        self.use_pe = use_pe
        self.hidden_dim = hidden_dim
        self.task = task
        self.use_stats = use_stats
        if self.task == 'regression' and use_stats:
            ## circuit statistics encoder + PE encoder + node/edge type encoders
            if use_pe:
                node_embed_dim = int(hidden_dim / 3)
            ## circuit statistics encoder + node/edge type encoders
            else:
                node_embed_dim = int(hidden_dim / 2)
        else:
            ## PE encoder + node/edge type encoders
            if use_pe:
                node_embed_dim = int(hidden_dim / 2)
            ## only use node/edge type encoders
            else:
                node_embed_dim = hidden_dim

        ## Circuit Statistics encoder
        if self.task == 'regression' and use_stats:
            c_embed_dim = hidden_dim % node_embed_dim + node_embed_dim
            self.c_embed_dim = c_embed_dim
            # add node_attr transform layer for net/device/pin nodes, by shan
            self.net_attr_layers = nn.Linear(17, c_embed_dim, bias=True)
            self.dev_attr_layers = nn.Linear(17, c_embed_dim, bias=True)
            self.pin_attr_layers = nn.Embedding(17, c_embed_dim)

        ## PE encoder for DSPD
        if use_pe:
            ## DSPD has 2 dimensions, distances to src and dst nodes
            self.pe_encoder = nn.Embedding(num_embeddings=max_dist+1,
                                           embedding_dim=int(node_embed_dim/2))

        ## Node / Edge type encoders
        self.node_encoder = nn.Embedding(num_embeddings=4,
                                         embedding_dim=node_embed_dim)
        self.edge_encoder = nn.Embedding(num_embeddings=10,
                                         embedding_dim=hidden_dim)
        
        # GNN layers
        self.layers = nn.ModuleList()
        self.drop_out = drop_out
        self.use_bn = use_bn
        for _ in range(num_layers):
            self.layers.append(ClusterGCNConv(hidden_dim,hidden_dim))

            ## the following are examples of using different GNN layers
            # self.layers.append(SSGConv(hidden_dim,hidden_dim, alpha=0.1, K=10))
            # self.layers.append(ResGatedGraphConv(hidden_dim,hidden_dim, edge_dim=hidden_dim))
            # self.layers.append(GATConv(hidden_dim,hidden_dim, heads=1))
            # mlp = MLP(
            #     in_channels=hidden_dim, 
            #     hidden_channels=hidden_dim, 
            #     out_channels=hidden_dim, 
            #     num_layers=2, 
            #     norm=None,
            # )
            # self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
        
        ## The head configuration
        head_input_dim = hidden_dim
        self.src_dst_agg = src_dst_agg
        head_input_dim = hidden_dim * 2 if src_dst_agg == 'concat' else hidden_dim
        if src_dst_agg == 'pool':
            self.pooling_fun = pygnn.pool.global_mean_pool

        self.num_head_layers = num_head_layers

        # head MLP layers
        self.head_layers = MLP(
            in_channels=head_input_dim, 
            hidden_channels=hidden_dim, 
            out_channels=dim_out, 
            num_layers=num_head_layers, 
            use_bn=False, dropout=0.0, activation=activation
        )
        self.bn_node_x = nn.BatchNorm1d(hidden_dim)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation')

    def forward(self, batch):
        ## Node type / Edge type encoding
        z = self.node_encoder(batch.x[:, 0])
        ze = self.edge_encoder(batch.edge_type)

        ## DSPD encoding
        if self.use_pe:
            dspd_emb = self.pe_encoder(batch.dspd)
            if dspd_emb.ndim == 3 and dspd_emb.size(1) == 2:
                dspd_emb = torch.cat((dspd_emb[:, 0, :], dspd_emb[:, 1, :]), dim=1)
            else:
                raise ValueError(
                    f"Dimension number of DSPD embedding is" + 
                    f" {dspd_emb.ndim}, size {dspd_emb.size()}")
            ## concatenate node embeddings and DSPD embeddings
            z = torch.cat((z, dspd_emb), dim=1)

        ## If we use circuit statistics encoder
        if self.task == 'regression' and self.use_stats:
            net_node_mask = batch.x.squeeze() == NET
            dev_node_mask = batch.x.squeeze() == DEV
            pin_node_mask = batch.x.squeeze() == PIN

            node_attr_emb = torch.zeros((batch.num_nodes, self.c_embed_dim), device=batch.x.device)
            node_attr_emb[net_node_mask] = \
                self.net_attr_layers(batch.node_attr[net_node_mask])
            node_attr_emb[dev_node_mask] = \
                self.dev_attr_layers(batch.node_attr[dev_node_mask])
            node_attr_emb[pin_node_mask] = \
                self.pin_attr_layers(batch.node_attr[pin_node_mask, 0].long())
            ## concatenate node embeddings and circuit statistics embeddings
            z = torch.cat((z, node_attr_emb), dim=1)

        for conv in self.layers:
            z = conv(z, batch.edge_index)
            ## for models that also take edge_attr as input
            # z = conv(z, batch.edge_index, edge_attr=ze)

            if self.use_bn:
                z = self.bn_node_x(z)
            z = self.activation(z)
            if self.drop_out > 0.0:
                z = F.dropout(z, p=self.drop_out, training=self.training)

        ## In head layers. If we use graph pooling, we need to call the pooling function here
        if self.src_dst_agg == 'pool':
            graph_emb = self.pooling_fun(z, batch.batch)
        ## Otherwise, only 2 anchor nodes are used to final prediction.
        else:
            batch_size = batch.edge_label.size(0)
            ## In the LinkNeighbor loader, the first batch_size nodes in z are source nodes and,
            ## the second 'batch_size' nodes in z are destination nodes. 
            ## Remaining nodes are the neighbors.
            src_emb = z[:batch_size, :]
            dst_emb = z[batch_size:batch_size*2, :]
            if self.src_dst_agg == 'concat':
                graph_emb = torch.cat((src_emb, dst_emb), dim=1)
            else:
                graph_emb = src_emb + dst_emb

        pred = self.head_layers(graph_emb)

        return pred, batch.edge_label