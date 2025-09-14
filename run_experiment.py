# run_experiment.py
import argparse, yaml, torch, numpy as np, random
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from src.dataloader import CausalFactorDataset
from src.models import GNN_NCM
from src.trainer import CausalTwoPartTrainer  # your trainer name

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if (cfg.get("device","cuda")=="cuda" and torch.cuda.is_available()) else "cpu")

    # dataset (VOL-only, trainer already evaluates on VOL)
    data_cfg = cfg.get("data", {})
    data_dir = data_cfg.get("data_dir", "data/processed")
    batch_size = int(data_cfg.get("batch_size", 1))
    ds = CausalFactorDataset(root_dir=data_dir, target_node=data_cfg.get("target_node","VOL"), drop_self_for_target=True)

    split = int(0.8 * len(ds))
    train_loader = DataLoader(Subset(ds, range(split)), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(ds, range(split, len(ds))), batch_size=batch_size, shuffle=False)

    # model
    g0 = ds[0]
    num_features = g0.num_node_features
    num_edges    = g0.edge_index.size(1)

    mcfg = cfg.get("model", {})
    model = GNN_NCM(
        num_features=num_features,
        num_edges=num_edges,
        gnn_mode=mcfg.get("gnn_mode","per_edge"),
        hidden_dim=int(mcfg.get("hidden_dim",16)),
        out_dim=int(mcfg.get("out_dim",8)),
        noise_dim=int(mcfg.get("noise_dim",4))
    ).to(device)

    # trainer (two-part)
    tcfg = cfg.get("training", {})
    trainer = CausalTwoPartTrainer(
        epochs_obs=int(tcfg.get("epochs_obs",40)),
        epochs_do=int(tcfg.get("epochs_do",20)),
        lr=float(tcfg.get("lr",1e-2)),
        w_obs=float(tcfg.get("w_obs",0.2)),
        w_do=float(tcfg.get("w_do",1.0)),
        weight_decay=float(tcfg.get("weight_decay",1e-4)),
        clip=float(tcfg.get("clip",1.0)),
        neutral=tcfg.get("neutral","zeros"),
        delta=float(tcfg.get("delta",0.1))
    )

    # train
    trainer.train(model, train_loader, val_loader=val_loader)

    # simple print at the end
    print("done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/best_config.yaml")
    args = p.parse_args()
    main(args.config)
