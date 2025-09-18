import argparse, yaml, torch, numpy as np, random
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from src.dataloader import CausalFactorDataset
from src.models import GNN_NCM
from src.trainer import CausalTwoPartTrainer 

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

@torch.no_grad()
def stress_degradation(model, loader, ops):
    mses_f = []
    for g in loader:
        yhat = model(g.x, g.edge_index)
        mses_f.append(float(((yhat - g.y)**2).mean().item()))
    mse_f = float(np.mean(mses_f))

    out = []
    for op in ops:
        mses_s = []
        for g in loader:
            x = g.x.clone()
            node = op["node"]
            new_val = op["value_fn"](float(x[node, 0].item())) if "value_fn" in op else float(op["value_const"])
            try:
                yhat_s = model.do_intervention(
                    x, g.edge_index,
                    intervened_nodes=[node],
                    new_feature_values=torch.tensor([new_val]).float()
                )
            except AttributeError:
                x[node, 0] = new_val
                yhat_s = model(x, g.edge_index)
            mses_s.append(float(((yhat_s - g.y)**2).mean().item()))
        mse_s = float(np.mean(mses_s))
        out.append({
            "stress_name": op["name"],
            "mse_factual": mse_f,
            "mse_stress": mse_s,
            "degradation_ratio": float(mse_s / (mse_f + 1e-12)),
        })
    return out

def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if (cfg.get("device","cuda")=="cuda" and torch.cuda.is_available()) else "cpu")

    # dataset
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
    )

    # trainer (two-part)
    tcfg = cfg.get("training", {})
    trainer = CausalTwoPartTrainer(
        epochs_obs=int(tcfg.get("epochs_obs",30)),  
        epochs_do=int(tcfg.get("epochs_do",30)),
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

    # save minimal artifacts
    outdir = Path(cfg.get("output", {}).get("output_dir", "outputs")) / cfg.get("experiment_name", "exp")
    outdir.mkdir(parents=True, exist_ok=True)

    import pandas as pd, json
    pd.DataFrame(trainer.history).to_csv(outdir / "history.csv", index=False)
    torch.save(model.state_dict(), outdir / "model.pt")

    final_val = [h["val_obs"] for h in trainer.history if h.get("val_obs") is not None][-1]
    with open(outdir / "final_metrics.json", "w") as f:
        json.dump({"val_obs_mse_VOL": float(final_val)}, f)
    print("Saved:", outdir)

    import copy, json, numpy as np

    model = model.eval().cpu()

    bas_idx = ds.node_map["BAS"]
    mom_idx = ds.node_map["Mom"]


    BAS_IDX, MOM_IDX = ds.node_map["BAS"], ds.node_map["Mom"]
    DELTA_BAS = 0.5  

    ops = [
        {"name": "do_BAS_plus_delta", "node": BAS_IDX, "value_fn": lambda v: v + DELTA_BAS},
        {"name": "do_MOM_plus_delta", "node": MOM_IDX, "value_fn": lambda v: v + DELTA_BAS},
    ]

    stress =stress_degradation(model, val_loader, ops)


    outdir = Path(cfg.get("output", {}).get("output_dir", "outputs")) / cfg.get("experiment_name", "exp")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "stress_test.json", "w") as f:
        json.dump(stress, f, indent=2)

    print(stress)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/best_config.yaml")
    args = p.parse_args()
    main(args.config)