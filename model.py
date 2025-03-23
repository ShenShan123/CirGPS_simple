import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, ResGatedGraphConv, 
    GINConv, ChebConv, GINEConv, ClusterGCNConv, SSGConv
)
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
    def __init__(self, args):
        super().__init__()
        self.use_pe = args.use_pe
        self.use_stats = args.use_stats
        hidden_dim = args.hid_dim
        ## only use node&edge type encoders
        node_embed_dim = hidden_dim
        
        ## circuit statistics encoder + PE encoder + node&edge type encoders
        if args.use_stats + self.use_pe == 2:
            node_embed_dim = hidden_dim // 3
            ## deal with hidden_dim not divisible by 3
            c_embed_dim = hidden_dim - 2 * node_embed_dim
        ## circuit statistics/pe encoder + node&edge type encoders
        elif self.use_stats + self.use_pe == 1:
            node_embed_dim = hidden_dim // 2
            c_embed_dim = hidden_dim - node_embed_dim

        ## Circuit Statistics encoder, producing matrix C
        if self.use_stats:
            ## add node_attr transform layer for net/device/pin nodes, by shan
            self.net_attr_layers = nn.Linear(17, c_embed_dim, bias=True)
            self.dev_attr_layers = nn.Linear(17, c_embed_dim, bias=True)
            ## pin attributes are {0, 1, 2} for gate pin, source/drain pin, and base pin
            self.pin_attr_layers = nn.Embedding(17, c_embed_dim)
            self.c_embed_dim = c_embed_dim


        ## PE encoder, producing D_0 and D_1
        if self.use_pe:
            assert node_embed_dim % 2 == 0, "node_embed_dim of self.pe_encoder should be even"
            ## DSPD has 2 dimensions, distances to src and dst nodes
            self.pe_encoder = nn.Embedding(num_embeddings=args.max_dist+1,
                                           embedding_dim=node_embed_dim//2)

        ## Node / Edge type encoders.
        ## Node attributes are {0, 1, 2} for net, device, and pin
        self.node_encoder = nn.Embedding(num_embeddings=4,
                                         embedding_dim=node_embed_dim)
        ## Edge attributes are {0, 1} for 'device-pin' and 'pin-net' edges
        self.edge_encoder = nn.Embedding(num_embeddings=4,
                                         embedding_dim=hidden_dim)
        
        # GNN layers
        self.layers = nn.ModuleList()
        self.model = args.model

        for _ in range(args.num_gnn_layers):
            ## the following are examples of using different GNN layers
            if args.model == 'clustergcn':
                self.layers.append(ClusterGCNConv(hidden_dim, hidden_dim))
            elif args.model == 'gcn':
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            elif args.model == 'sage':
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            elif args.model == 'gat':
                self.layers.append(GATConv(hidden_dim, hidden_dim, heads=1))
            elif args.model == 'resgatedgcn':
                self.layers.append(ResGatedGraphConv(hidden_dim, hidden_dim, edge_dim=hidden_dim))
            elif args.model == 'gine':
                mlp = MLP(
                    in_channels=hidden_dim, 
                    hidden_channels=hidden_dim, 
                    out_channels=hidden_dim, 
                    num_layers=2, 
                    norm=None,
                )
                self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
            else:
                raise ValueError(f'Unsupported GNN model: {args.model}')
        
        self.src_dst_agg = args.src_dst_agg

        ## Add graph pooling layer
        if args.src_dst_agg == 'pool':
            # self.pooling_fun = pygnn.pool.global_mean_pool
            self.pooling_fun = pygnn.pool.global_add_pool
        
        ## The head configuration
        head_input_dim = hidden_dim * 2 if self.src_dst_agg == 'concat' else hidden_dim
        dim_out = 1
        # head MLP layers
        self.head_layers = MLP(
            in_channels=head_input_dim, 
            hidden_channels=hidden_dim, 
            out_channels=dim_out, 
            num_layers=args.num_head_layers, 
            use_bn=False, dropout=0.0, 
            activation=args.act_fn,
        )

        ## Batch normalization
        self.use_bn = args.use_bn
        self.bn_node_x = nn.BatchNorm1d(hidden_dim)

        ## activation setting
        if args.act_fn == 'relu':
            self.activation = nn.ReLU()
        elif args.act_fn == 'elu':
            self.activation = nn.ELU()
        elif args.act_fn == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation')
        
        ## Dropout setting
        self.drop_out = args.dropout

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
        if self.use_stats:
            net_node_mask = batch.x.squeeze() == NET
            dev_node_mask = batch.x.squeeze() == DEV
            pin_node_mask = batch.x.squeeze() == PIN

            node_attr_emb = torch.zeros(
                (batch.num_nodes, self.c_embed_dim), device=batch.x.device
            )
            node_attr_emb[net_node_mask] = \
                self.net_attr_layers(batch.node_attr[net_node_mask])
            node_attr_emb[dev_node_mask] = \
                self.dev_attr_layers(batch.node_attr[dev_node_mask])
            node_attr_emb[pin_node_mask] = \
                self.pin_attr_layers(batch.node_attr[pin_node_mask, 0].long())
            ## concatenate node embeddings and circuit statistics embeddings
            z = torch.cat((z, node_attr_emb), dim=1)

        for conv in self.layers:
            ## for models that also take edge_attr as input
            if self.model == 'gine' or self.model == 'resgatedgcn':
                z = conv(z, batch.edge_index, edge_attr=ze)
            else:
                z = conv(z, batch.edge_index)

            if self.use_bn:
                z = self.bn_node_x(z)
            
            z = self.activation(z)

            if self.drop_out > 0.0:
                z = F.dropout(z, p=self.drop_out, training=self.training)

        ## In head layers. If we use graph pooling, we need to call the pooling function here
        if self.src_dst_agg == 'pool':
            graph_emb = self.pooling_fun(z, batch.batch)
        ## Otherwise, only 2 embeddings from the anchor nodes are used to final prediction.
        else:
            batch_size = batch.edge_label.size(0)
            ## In the LinkNeighbor loader, the first batch_size nodes in z are source nodes and,
            ## the second 'batch_size' nodes in z are destination nodes. 
            ## Remaining nodes are their '1-hop', '2-hop', 'n-hop' neighbors.
            src_emb = z[:batch_size, :]
            dst_emb = z[batch_size:batch_size*2, :]
            if self.src_dst_agg == 'concat':
                graph_emb = torch.cat((src_emb, dst_emb), dim=1)
            else:
                graph_emb = src_emb + dst_emb

        pred = self.head_layers(graph_emb)

        return pred, batch.edge_label