import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ResGatedGraphConv, GINConv, ChebConv, GINEConv, ClusterGCNConv
from torch_geometric.utils import dropout_edge,mask_feature
    
class CustomConv(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,proj_dim,activation,num_layers,use_bn=False,drop_out=0.0):
        super(CustomConv,self).__init__()
        self.activation = activation
        self.node_type_embed = nn.Embedding(6, hidden_dim)
        self.edge_type_embed = nn.Embedding(8, hidden_dim)
        self.layers = nn.ModuleList()
        self.drop_out = drop_out

        for _ in range(num_layers):
            self.layers.append(ClusterGCNConv(hidden_dim,hidden_dim))
            # mlp = nn.Sequential(
            #     nn.Linear(hidden_dim, hidden_dim),
            #     nn.BatchNorm1d(hidden_dim),
            #     nn.ReLU(),
            #     nn.Linear(hidden_dim, hidden_dim),
            # )
            # self.layers.append(GINEConv(mlp, train_eps=True))
        # self.layers = Model(hidden_dim, hidden_dim, K=10, dprate=0.4, dropout=0.4, is_bns=False, act_fn='relu')

        self.use_bn = use_bn
        self.bn_node_x = nn.BatchNorm1d(hidden_dim)
        
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim,proj_dim),
            torch.nn.PReLU(),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(proj_dim,proj_dim)
        )  
        
    def forward(self,batch):
        z = self.node_type_embed(batch.x).squeeze()
        ze = self.node_type_embed(batch.edge_type).squeeze()
        for conv in self.layers:
            z = conv(z, batch.edge_index)#, edge_attr=ze)

            if self.use_bn:
                z = self.bn_node_x(z)

            z = self.activation(z)
            z = F.dropout(z, p=self.drop_out, training=self.training)
        # z = self.layers(z, batch.edge_index)
        return z, self.projection_head(z)

class CustomOnline(torch.nn.Module):
    def __init__(self,online_encoder,target_encoder,hidden_dim,num_hop,momentum):
        super(CustomOnline,self).__init__()
        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.num_hop = num_hop
        self.momentum = momentum
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
    def update_target_encoder(self):
        for p, new_p in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            next_p = self.momentum * p.data + (1 - self.momentum) * new_p.data
            p.data = next_p
            
    def forward(self, batch):
        h = self.embed(batch, self.num_hop)
        h_pred = self.predictor(h)
        with torch.no_grad():
               h_target,_ = self.target_encoder(batch)
              
        return h,h_pred,h_target
       
    def get_loss(self,z1,z2):
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)
        loss = (z1 * z2).sum(dim=-1)
        loss = -loss.mean()
        return loss
    
    def embed(self, batch, Globalhop=10):
        h_1,_ = self.online_encoder(batch)
        return h_1
        #NOTE: The following code is not used in the current implementation
        #TODO: Compare the performance with/without slsp_adj
        h_2 = h_1.clone()
        for _ in range(Globalhop):
            h_2 = batch.slsp_adj @ h_2
        return h_1 + h_2
    
    
class Target(torch.nn.Module):
    def __init__(self,target_encoder):
        super(Target,self).__init__()
        self.target_encoder = target_encoder
        
    def forward(self,batch):
        h_target,_ = self.target_encoder(batch)
        return h_target
    
    def get_loss(self,z):
        z = F.normalize(z,dim=-1, p=2)
        return -(z - z.mean(dim=0)).pow(2).sum(1).mean()


