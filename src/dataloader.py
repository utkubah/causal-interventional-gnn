# src/data_loader.py

import os
import torch
import pandas as pd
from torch_geometric.data import Dataset, Data

class CausalFactorDataset(Dataset):
    """
    We train on all nodes’ Y as same-day reconstruction; evaluation focuses on VOL.
    - X: node features per graph/day
    - Y: same-day reconstruction target = X.clone()
    - target_idx: index of VOL (used by eval in trainer/run_experiment)
    """
    def __init__(
        self,
        root_dir="data/processed",
        target_node="VOL",
        feature_col=None,      # if None -> auto-detect the single numeric col
        drop_self_for_target=True,
        fillna=0.0,
        dtype=torch.float
    ):
        super().__init__()
        self.root_dir = root_dir
        self.target_node = target_node
        self.drop_self_for_target = drop_self_for_target
        self.fillna = fillna
        self.dtype = dtype

        nodes = pd.read_csv(os.path.join(root_dir, "nodes.csv"))
        edges = pd.read_csv(os.path.join(root_dir, "edges.csv"))
        feats = pd.read_csv(os.path.join(root_dir, "features.csv"))

        node_ids = nodes["node_id"].tolist()
        self.node_map = {nid: i for i, nid in enumerate(node_ids)}
        if self.target_node not in self.node_map:
            raise ValueError(f"target_node '{self.target_node}' not in nodes.csv")
        self.target_idx = self.node_map[self.target_node]

        # edges
        es = edges["source"].map(self.node_map).to_numpy()
        ed = edges["target"].map(self.node_map).to_numpy()
        self.edge_index = torch.tensor([es, ed], dtype=torch.long)
        self.num_nodes = len(node_ids)

        # sort features by date/node
        feats["date"] = pd.to_datetime(feats["date"])
        feats = feats.sort_values(["date", "node_id"])

        # detect the single numeric feature column if not provided
        if feature_col is None:
            cand = [c for c in feats.columns if c not in {"date","node_id"} and pd.api.types.is_numeric_dtype(feats[c])]
            if len(cand) != 1:
                raise ValueError(f"expected exactly 1 numeric feature col; found {cand}")
            feature_col = cand[0]
        self.feature_col = feature_col

        # X: [T, N, 1]
        p = feats.pivot(index="date", columns="node_id", values=self.feature_col)
        p = p.reindex(columns=node_ids).sort_index().fillna(self.fillna)
        X = torch.tensor(p.values, dtype=self.dtype).unsqueeze(-1)  # [T,N,1]

        # Y = same-day full vector (trainer expects [N,1])
        Y = X.clone()

        self.dates = p.index
        self.X = X
        self.Y = Y
        self.node_ids = node_ids

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        x = self.X[idx].clone()   # [N,1]
        if self.drop_self_for_target:
            x[self.target_idx] = 0.0
        y = self.Y[idx]           # [N,1]
        return Data(x=x, edge_index=self.edge_index, y=y, num_nodes=self.num_nodes)
