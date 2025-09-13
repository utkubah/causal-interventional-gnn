# src/data_loader.py
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class CausalFactorDataset(Dataset):
    """
    A simple, streamlined PyTorch Geometric Dataset for loading the causal factor graph.
    
    This loader reads the processed CSV files and creates a time series of daily
    graph snapshots. The task is to predict the next day's VOL and LIQ values
    based on the current day's values for all nodes.
    """
    def __init__(self, root_dir: str, train: bool = True, train_split: float = 0.8):
        self.root_dir = Path(root_dir)
        self.train = train
        self.train_split = train_split
        
        # --- 1. Load all raw data ---
        nodes_df = pd.read_csv(self.root_dir / "nodes.csv")
        edges_df = pd.read_csv(self.root_dir / "edges.csv")
        features_df = pd.read_csv(self.root_dir / "features.csv", parse_dates=['date'])

        # --- 2. Define the static graph structure ---
        node_mapping = {node_id: i for i, node_id in enumerate(nodes_df['node_id'])}
        self.edge_index = torch.tensor([
            [node_mapping[src], node_mapping[dst]]
            for src, dst in edges_df[['source', 'target']].values
        ], dtype=torch.long).t().contiguous()

        # --- 3. Process the time-series features ---
        # Pivot from long to wide format (dates x nodes)
        features_wide = features_df.pivot(index='date', columns='node_id', values='value')
        # Ensure column order is consistent
        node_order = nodes_df['node_id'].tolist()
        features_wide = features_wide[node_order]
        
        # --- 4. Create inputs (x) and targets (y) ---
        # The goal is to predict the *next day's* VOL and LIQ.
        # So, x is today's data, and y is tomorrow's VOL/LIQ.
        all_x = torch.tensor(features_wide.values, dtype=torch.float32)
        all_y_targets = torch.tensor(features_wide[['VOL', 'LIQ']].values, dtype=torch.float32)
        
        # x becomes all days except the last one
        self.x = all_x[:-1]
        # y becomes all days except the first one
        self.y = all_y_targets[1:]
        self.dates = features_wide.index[:-1]

        # --- 5. Normalize the input features ---
        # Note: For simplicity, we scale across the whole dataset. For rigorous academic
        # work, the scaler should be fit ONLY on the training data.
        scaler = StandardScaler()
        self.x = torch.from_numpy(scaler.fit_transform(self.x))

        # --- 6. Split into training and test sets ---
        split_idx = int(len(self.x) * self.train_split)
        if self.train:
            self.x = self.x[:split_idx]
            self.y = self.y[:split_idx]
            self.dates = self.dates[:split_idx]
        else:
            self.x = self.x_data[split_idx:]
            self.y = self.y_data[split_idx:]
            self.dates = self.dates[split_idx:]
            
        # Store metadata
        self.num_nodes = len(nodes_df)
        self.node_mapping = node_mapping
        
        super().__init__(str(root_dir), None, None)

    def len(self):
        """Returns the number of days in the dataset."""
        return len(self.x_data)

    def get(self, idx: int):
        """
        Returns a single graph snapshot for a specific day.
        
        The features for each node (x) is its single value for that day.
        The target (y) is the [VOL, LIQ] tuple for the *next* day.
        """
        # Node features for the day `idx`. Shape must be [num_nodes, num_features].
        # Since each node has 1 feature (its value), we add a dimension.
        x_snapshot = self.x_data[idx].unsqueeze(1)
        
        # Target values for the day `idx` (which corresponds to day `idx+1`'s values)
        y_snapshot = self.y_data[idx]
        
        return Data(x=x_snapshot, edge_index=self.edge_index, y=y_snapshot, date=self.dates[idx])