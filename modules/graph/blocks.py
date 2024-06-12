import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

class SpatialBasicBlock(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, aggr='add', batch_norm=False, bias=True, **kwargs):
        super(SpatialBasicBlock, self).__init__(aggr=aggr, **kwargs)
        self.batch_norm = batch_norm
        # network architecture
        self.lin_f = nn.Linear(2*in_channels + edge_channels, out_channels, bias=bias)
        self.lin_s = nn.Linear(2*in_channels + edge_channels, out_channels, bias=bias)
        self.upsample = nn.Linear(in_channels, out_channels, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.bn(out) if self.batch_norm else out
        out += self.upsample(x[1])
        return out

    def message(self, x_i, x_j, edge_attr):
        if edge_attr is None:
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))
    
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels, dim):
        super(Encoder, self).__init__()
        self.out_channels = out_channels
        self.layers = nn.ModuleList([
            SpatialBasicBlock(in_channels=in_channels, out_channels=hidden_dims[0], edge_channels=dim)
        ])
        for i in range(len(hidden_dims)-1):
            self.layers.append(SpatialBasicBlock(in_channels=hidden_dims[i], out_channels=hidden_dims[i+1], edge_channels=dim))
        self.layers.append(SpatialBasicBlock(in_channels=hidden_dims[-1], out_channels=out_channels, edge_channels=dim))

    def forward(self, data):
        """
        Keyword arguments:
        x -- joint angles [num_nodes, num_node_features]
        edge_index -- edge index [2, num_edges]
        edge_attr -- edge features [num_edges, num_edge_features]
        """
        out = data.x
        for layer in self.layers:
            out = layer(out, data.edge_index, data.edge_attr)
        out = out.reshape(data.num_graphs, -1, self.out_channels)
        return out
    
class Decoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels, dim):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.layers = nn.ModuleList([
            SpatialBasicBlock(in_channels=in_channels, out_channels=hidden_dims[0], edge_channels=dim)
        ])
        for i in range(len(hidden_dims)-1):
            self.layers.append(SpatialBasicBlock(in_channels=hidden_dims[i], out_channels=hidden_dims[i+1], edge_channels=dim))
        self.layers.append(SpatialBasicBlock(in_channels=hidden_dims[-1], out_channels=out_channels, edge_channels=dim))


    def forward(self, z, target):
        """
        Keyword arguments:
        x -- joint angles [num_nodes, num_node_features]
        edge_index -- edge index [2, num_edges]
        edge_attr -- edge features [num_edges, num_edge_features]
        """
        out = z.reshape(-1, z.shape[-1])
        for layer in self.layers:
            out = layer(out, target.edge_index, target.edge_attr)
        out = out.reshape(target.num_graphs, -1, self.out_channels)
        return out