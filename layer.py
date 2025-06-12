import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_scatter import scatter
from torch_geometric.graphgym import cfg


class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    """
        GatedGCN layer
        Residual Gated Graph ConvNets
        https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, in_dim, out_dim, dropout, residual, act='relu',
                 equivstable_pe=False, **kwargs):
        super().__init__(**kwargs)
        self.activation = register.act_dict[act]
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.C = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim),
                self.activation(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid())

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.act_fn_x = self.activation()
        self.act_fn_e = self.activation()
        self.dropout = dropout
        self.residual = residual
        self.e = None

    def forward(self, batch):
        batch.x = batch.x.float()
        batch.edge_attr = batch.edge_attr.float()
        # print("x", batch.x.size()) #([30672, 144])
        # print("edge_attr",batch.edge_attr.size()) #([61880, 144])
        # assert 0
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index

        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        x, e = self.propagate(edge_index,
                              Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce,
                              e=e, Ax=Ax,
                              PE=pe_LapPE)

        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = self.act_fn_x(x)
        e = self.act_fn_e(e)

        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            e = e_in + e

        batch.x = x
        batch.edge_attr = e

        return batch

    def message(self, Dx_i, Ex_j, PE_i, PE_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim
            sigma_ij = sigma_ij * r_ij

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size,
                                   reduce='sum')

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size,
                                     reduce='sum')

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out


# @register_layer('gatedgcnconv')
# class GatedGCNGraphGymLayer(nn.Module):
#     """GatedGCN layer.
#     Residual Gated Graph ConvNets
#     https://arxiv.org/pdf/1711.07553.pdf
#     """
#     def __init__(self, layer_config: LayerConfig, **kwargs):
#         super().__init__()
#         self.model = GatedGCNLayer(in_dim=layer_config.dim_in,
#                                    out_dim=layer_config.dim_out,
#                                    dropout=0.,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
#                                    residual=False,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
#                                    act=layer_config.act,
#                                    **kwargs)

#     def forward(self, batch):
#         return self.model(batch)
    
    
class GCNConvLayer(nn.Module):
    
    def __init__(self, dim_in, dim_out, dropout, residual, ffn):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = True
        self.ffn = ffn
        
        if self.batch_norm:
            self.bn_node_x = nn.BatchNorm1d(dim_out)
        
        self.act = nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            nn.Dropout(self.dropout),
        )
        
        self.model = pyg_nn.GCNConv(dim_in, dim_out, bias=True)
        
        if self.ffn:
            # Feed Forward block.
            if self.batch_norm:
                self.norm1_local = nn.BatchNorm1d(dim_out)
            self.ff_linear1 = nn.Linear(dim_out, dim_out*2)
            self.ff_linear2 = nn.Linear(dim_out*2, dim_out)
            self.act_fn_ff = register.act_dict[cfg.gnn.act]()
            if self.batch_norm:
                self.norm2 = nn.BatchNorm1d(dim_out)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)
        
    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, batch):
        x_in = batch.x
        
        batch.x = self.model(batch.x, batch.edge_index)
        if self.batch_norm:
            batch.x = self.bn_node_x(batch.x)
        batch.x = self.act(batch.x)
        
        if self.residual:
            batch.x = x_in + batch.x  # Residual connection.
        
        if self.ffn:
            if self.batch_norm:
                batch.x = self.norm1_local(batch.x)
            
            batch.x = batch.x + self._ff_block(batch.x)

            if self.batch_norm:
                batch.x = self.norm2(batch.x)

        return batch
   
    
class GINEConvLayer(nn.Module):
    """Graph Isomorphism Network with Edge features (GINE) layer.
    """
    def __init__(self, dim_in, dim_out, dropout, residual, ffn):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual

        gin_nn = nn.Sequential(
            pyg_nn.Linear(dim_in, dim_out), nn.ReLU(),
            pyg_nn.Linear(dim_out, dim_out))
        self.model = pyg_nn.GINEConv(gin_nn)
        
        self.batch_norm = True
        self.ffn = ffn
        
        if self.ffn:
            # Feed Forward block.
            if self.batch_norm:
                self.norm1_local = nn.BatchNorm1d(dim_out)
            self.ff_linear1 = nn.Linear(dim_out, dim_out*2)
            self.ff_linear2 = nn.Linear(dim_out*2, dim_out)
            self.act_fn_ff = register.act_dict[cfg.gnn.act]()
            if self.batch_norm:
                self.norm2 = nn.BatchNorm1d(dim_out)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))
    
    def forward(self, batch):
        x_in = batch.x
        # y_in = batch.edge_attr
        # print(f"x_in.size():{x_in.size()}") #([30649, 144])
        # print(f"y_in.size():{y_in.size()}")      
        # assert 0
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)

        batch.x = F.relu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        if self.ffn:
            if self.batch_norm:
                batch.x = self.norm1_local(batch.x)
            
            batch.x = batch.x + self._ff_block(batch.x)

            if self.batch_norm:
                batch.x = self.norm2(batch.x)
                
        return batch

    