# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv

class EdgeWiseGNNLayer(MessagePassing):
    """
    A GNN layer that resolves the SCM modularity vs. GNN parameter sharing trade-off.
    
    In 'per_edge' mode, it instantiates a unique MLP for each edge, allowing it
    to learn distinct causal mechanisms for each parent-child relationship.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_edges, mode='per_edge'):
        super().__init__(aggr='add', flow='source_to_target')
        
        self.mode = mode
        self.num_edges = num_edges
        self.out_dim = out_dim

        if self.mode == 'per_edge':
            # Create a list of MLPs, one for each edge. This is our f_ij.
            self.edge_mlps = nn.ModuleList([
                nn.Sequential(nn.Linear(in_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, out_dim)) for i in range(num_edges)
            ])
        elif self.mode == 'shared': 
            # In shared mode, we only have one MLP for all edges.
            self.edge_mlps = nn.Sequential(nn.Linear(in_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, out_dim))
        else:
            raise ValueError("Mode must be 'shared' or 'per_edge'")
        
        
        # The update function φ remains shared across all nodes.
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, original_edge_ids=None):
        """
        Executes one full message-passing step.

        Args:
            x (Tensor): The input node features.
            edge_index (Tensor): The connectivity for this pass.
            original_edge_ids (Tensor, optional): The original IDs of the edges in edge_index.
                                                   If None, assumes a full graph pass.
        """

        if original_edge_ids is None:
            original_edge_ids = torch.arange(self.num_edges, device=x.device)

        aggr_out = self.propagate(edge_index, x=x, original_edge_ids=original_edge_ids)
        
        return self.update(aggr_out, x)


    def message(self, x_j, original_edge_ids):
        """
        Computes the message from parent (x_j) to child.
        
        Args:
            x_j (Tensor): The feature tensor of the source nodes (parents) for each edge.
            edge_ids (Tensor): A tensor containing the index of each edge, which we use
                               to select the appropriate MLP.
        """
        output_messages = torch.zeros(x_j.size(0), self.out_dim, device=x_j.device)
        if self.mode == 'per_edge':
            for i in range(self.num_edges):
                mask = (original_edge_ids == i)
                if mask.any(): output_messages[mask] = self.edge_mlps[i](x_j[mask])
            return output_messages
        else: 
            return self.edge_mlps(x_j)

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))

class GNN_NCM(nn.Module):
    """
    The GNN-NCM model: a high-fidelity SCM analogue capable of interventions.
      - Adds exogenous noise U
      - Supports explicit do-interventions via graph surgery
      - Uses EdgeWiseGNNLayer for per-edge mechanisms
    """
    def __init__(self, num_features, num_edges, hidden_dim=16, out_dim=8, noise_dim=4, gnn_mode='per_edge'):
        super().__init__()
        self.noise_dim, self.num_edges = noise_dim, num_edges
        input_dim = num_features + noise_dim
        self.conv1 = EdgeWiseGNNLayer(input_dim, hidden_dim, hidden_dim, num_edges, mode=gnn_mode)
        self.conv2 = EdgeWiseGNNLayer(hidden_dim, out_dim, out_dim, num_edges, mode=gnn_mode)
        self.out = nn.Linear(out_dim, 1)

    def forward(self, x, edge_index):
        noise = torch.randn(x.size(0), self.noise_dim, device=x.device)
        x_with_noise = torch.cat([x, noise], dim=1)
        h = F.relu(self.conv1(x_with_noise, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.out(h)


    def do_intervention(self, x, edge_index, intervened_nodes, new_feature_values):
        x_intervened = x.clone()
        x_intervened[intervened_nodes] = new_feature_values
        edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=x.device)
        for node_idx in intervened_nodes:
            edge_mask &= (edge_index[1] != node_idx)
        intervened_edge_index = edge_index[:, edge_mask]
        all_edge_ids = torch.arange(self.num_edges, device=x.device)
        intervened_edge_ids = all_edge_ids[edge_mask]
        noise = torch.randn(x.size(0), self.noise_dim, device=x.device)
        x_with_noise = torch.cat([x_intervened, noise], dim=1)
        h1 = self.conv1(x_with_noise, intervened_edge_index, original_edge_ids=intervened_edge_ids)
        h1 = F.relu(h1)
        h2 = self.conv2(h1, intervened_edge_index, original_edge_ids=intervened_edge_ids)
        h2 = F.relu(h2)
        return self.out(h2)

class BaselineGCN(nn.Module):
    """A simple, shared-parameter GCN for comparisons."""
    def __init__(self, num_features: int, hidden_dim: int = 32, out_dim: int = 16):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.out = nn.Linear(out_dim, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.out(h)
