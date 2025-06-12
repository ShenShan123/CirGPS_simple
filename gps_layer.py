import numpy as np
import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch

# from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from layer import GatedGCNLayer, GCNConvLayer, GINEConvLayer
# from graphgps.layer.gine_conv_layer import GINEConvESLapPE


class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, hid_dim,
                 local_gnn_type, global_model_type, num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 log_attn_weights=False,
                 res_linear=False):
        super().__init__()

        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in ['Transformer',
                                                          'BiasedTransformer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )
        # added by shan
        self.res_linear = res_linear
        self.residual_linear = nn.Linear(hid_dim, hid_dim)

        # Local message-passing model.
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == 'None':
            self.local_model = None

        # MPNNs without edge attributes support.
        elif local_gnn_type == 'GCN':
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GCNConv(hid_dim, hid_dim)
        elif local_gnn_type == 'GIN':
            self.local_gnn_with_edge_attr = False
            gin_nn = nn.Sequential(Linear_pyg(hid_dim, hid_dim),
                                   self.activation(),
                                   Linear_pyg(hid_dim, hid_dim))
            self.local_model = pygnn.GINConv(gin_nn)

        # MPNNs supporting also edge attributes.
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(hid_dim, hid_dim)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(hid_dim, hid_dim),
                                   self.activation(),
                                   Linear_pyg(hid_dim, hid_dim))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=hid_dim,
                                             out_channels=hid_dim // num_heads,
                                             heads=num_heads,
                                             edge_dim=hid_dim)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(hid_dim, hid_dim,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=min(128, hid_dim),
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(hid_dim, hid_dim,
                                             dropout=dropout,
                                             residual=True,
                                             act=act,
                                             equivstable_pe=equivstable_pe)
        elif local_gnn_type == 'CustomGCNConv':
            self.local_model = GCNConvLayer(hid_dim, hid_dim,
                                            dropout=dropout,
                                            residual=True,
                                            ffn=True)
        elif local_gnn_type == 'CustomGINEConv':    
            self.local_model = GINEConvLayer(hid_dim, hid_dim,
                                            dropout=dropout,
                                            residual=True,
                                            ffn=True)   
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type in ['Transformer', 'BiasedTransformer']:
            self.self_attn = torch.nn.MultiheadAttention(
                hid_dim, num_heads, dropout=self.attn_dropout, batch_first=True)
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=hid_dim, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        # elif global_model_type == 'Performer':
        #     self.self_attn = SelfAttention(
        #         dim=hid_dim, heads=num_heads,
        #         dropout=self.attn_dropout, causal=False)
        # elif global_model_type == "BigBird":
        #     bigbird_cfg.dim_hidden = hid_dim
        #     bigbird_cfg.n_heads = num_heads
        #     bigbird_cfg.dropout = dropout
        #     self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(hid_dim)
            self.norm1_attn = pygnn.norm.LayerNorm(hid_dim)
            # self.norm1_local = pygnn.norm.GraphNorm(hid_dim)
            # self.norm1_attn = pygnn.norm.GraphNorm(hid_dim)
            # self.norm1_local = pygnn.norm.InstanceNorm(hid_dim)
            # self.norm1_attn = pygnn.norm.InstanceNorm(hid_dim)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(hid_dim)
            self.norm1_attn = nn.BatchNorm1d(hid_dim)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(hid_dim, hid_dim * 2)
        self.ff_linear2 = nn.Linear(hid_dim * 2, hid_dim)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(hid_dim)
            # self.norm2 = pygnn.norm.GraphNorm(hid_dim)
            # self.norm2 = pygnn.norm.InstanceNorm(hid_dim)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(hid_dim)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        # 保存原始节点顺序以便后续恢复
        if self.res_linear:
            # linear residual connection
            h_in1 = self.residual_linear(h) 
        else:
            h_in1 = h  # for first residual connection
        
        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == 'CustomGatedGCN' or 'CustomGINEConv':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(Batch(batch=batch,
                                                x=h,
                                                edge_index=batch.edge_index,
                                                edge_attr=batch.edge_attr,
                                                pe_EquivStableLapPE=es_data))
                # GatedGCN does residual connection and dropout internally.
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr 
            elif self.local_gnn_type == 'CustomGCNConv' :
                local_out = self.local_model(Batch(batch=batch,
                                                x=h,
                                                edge_index=batch.edge_index))
                h_local = local_out.x          
            else:
                if self.local_gnn_with_edge_attr:
                    if self.equivstable_pe:
                        h_local = self.local_model(h,
                                                batch.edge_index,
                                                batch.edge_attr,
                                                batch.pe_EquivStableLapPE)
                    else:
                        h_local = self.local_model(h,
                                                batch.edge_index,
                                                batch.edge_attr)
                else:
                    h_local = self.local_model(h, batch.edge_index)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            # 保存原始顺序用于后续恢复
            sorted_idx = batch.batch.sort()[1]
            
            # 按批次索引对节点排序
            h_sorted = h[sorted_idx]
            batch_sorted = batch.batch[sorted_idx]
            
            # 转换为密集批次格式
            h_dense, mask = to_dense_batch(h_sorted, batch_sorted)

            if self.global_model_type == 'Transformer':
                h_attn_dense = self._sa_block(h_dense, None, ~mask)
            elif self.global_model_type == 'BiasedTransformer':
                h_attn_dense = self._sa_block(h_dense, batch.attn_bias, ~mask)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            # 从密集格式中提取有效节点
            h_attn_sorted = h_attn_dense[mask]  # [num_valid_nodes, hid_dim]
            h_attn_sorted = self.dropout_attn(h_attn_sorted)

            # 直接恢复原始节点顺序
            # h_attn_sorted 已经包含了所有有效节点的注意力输出，顺序与 h_sorted 中有效节点一致
            h_attn = torch.zeros_like(h_in1)  # 原始大小
            h_attn[sorted_idx] = h_attn_sorted  # 将排序后的结果放回原始位置
            
            # 残差连接
            h_attn = h_in1 + h_attn
            
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # 合并局部和全局输出
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: hid_dim={self.hid_dim}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s