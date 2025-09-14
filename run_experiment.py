# run_experiment.py
import os, json, argparse, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset


# --- Project imports (aligned to repo layout) ---
# utils has set_seed() and load_config()
from src.utils import set_seed, load_config  # :contentReference[oaicite:3]{index=3}
from src.dataloader import CausalFactorDataset  # loader name in the repo  :contentReference[oaicite:4]{index=4}
from src.models import GNN_NCM
from src.trainer import HybridCausalTrainer  # loader-compatible trainer you’ve been using

# ----------------------------
# Helpers
# ----------------------------
def _to_device(batch, device):
    # batch is a single-day graph (batch_size=1). Keep it simple.
    batch.x = batch.x.to(device)
    batch.edge_index = batch.edge_index.to(device)
    if hasattr(batch, 'y') and batch.y is not None:
        batch.y = batch.y.to(device)
    return batch

def _safe_node_map(dataset):
    # Repo’s dataloader uses `node_mapping`; some of our newer snippets used `node_map`.
    if hasattr(dataset, "node_mapping"): return dataset.node_mapping
    if hasattr(dataset, "node_map"):     return dataset.node_map
    raise RuntimeError("Dataset missing node mapping attribute (expected node_mapping or node_map).")

def _pick_target_and_shock(node_map, prefer_target="VOL", prefer_shock="BAS"):
    # Fallbacks if nodes are named differently or missing
    tgt = prefer_target if prefer_target in node_map else list(node_map.keys())[-1]
    shock = prefer_shock if prefer_shock in node_map else list(node_map.keys())[0]
    return tgt, shock

def _infer_data_dir(cfg):
    # Accept both the new nested config and the old flat one
    if isinstance(cfg, dict):
        if "data" in cfg and isinstance(cfg["data"], dict) and "data_dir" in cfg["data"]:
            return cfg["data"]["data_dir"]
        if "data_dir" in cfg:
            return cfg["data_dir"]
    return "data/processed"

def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def _maybe_history_dict(trainer_obj):
    # New trainer (logged) exposes .history (list of dicts)
    # Old trainer doesn’t; return empty list in that case.
    return getattr(trainer_obj, "history", [])

def _as_float(x):
    return float(x) if hasattr(x, "item") else float(x)

# ----------------------------
# Core experiment
# ----------------------------
def run_experiment(config_path: str):
    # --- 1) Setup
    cfg = load_config(config_path)  # :contentReference[oaicite:5]{index=5}
    set_seed(cfg.get("seed", 42))   # :contentReference[oaicite:6]{index=6}

    dev_str = cfg.get("device", "cuda")
    device = torch.device(dev_str if torch.cuda.is_available() and dev_str.startswith("cuda") else "cpu")

    exp_name   = cfg.get("experiment_name", f"exp_{int(time.time())}")
    out_root   = cfg.get("output", {}).get("output_dir", cfg.get("output_dir", "outputs"))
    exp_dir    = _ensure_dir(os.path.join(out_root, exp_name))
    plots_dir  = _ensure_dir(os.path.join(exp_dir, "plots"))

    print(f"\n--- Running Experiment: {exp_name} ---")
    print(f"Device: {device}")

    # --- 2) Data
    data_dir = _infer_data_dir(cfg)
    print(f"Using data at: {data_dir}")

    full_dataset = CausalFactorDataset(root_dir=data_dir, target_node="VOL", feature_col=None, drop_self_for_target=True)

    # batch_size=1 → one graph per day (no sequence batching)
    split = int(0.8*len(full_dataset))
    train_loader = DataLoader(Subset(full_dataset, list(range(split))), batch_size=1, shuffle=True)
    val_loader   = DataLoader(Subset(full_dataset, list(range(split, len(full_dataset)))), batch_size=1, shuffle=False)
    
    # Pull graph shape from the first training item
    g0 = full_dataset[0]
    num_features = g0.num_node_features if hasattr(g0, "num_node_features") else g0.x.shape[-1]
    num_edges    = g0.edge_index.size(1)
    node_map     = _safe_node_map(full_dataset)
    target_node, shock_node = _pick_target_and_shock(node_map, 
                                                     prefer_target=cfg.get("data", {}).get("target_node", "VOL"),
                                                     prefer_shock =cfg.get("analysis", {}).get("shock_node", "BAS"))
    tgt_idx   = node_map[target_node]
    shock_idx = node_map[shock_node]

    print(f"Target node: {target_node} (idx={tgt_idx})")
    print(f"Shock node:  {shock_node} (idx={shock_idx})")

    # --- 3) Model
    model_cfg = cfg.get("model", {})
    gnn_mode  = model_cfg.get("gnn_mode", "per_edge")
    hidden    = model_cfg.get("hidden_dim", 32)
    out_dim   = model_cfg.get("out_dim", 16)
    noise_dim = model_cfg.get("noise_dim", 4)

    model = GNN_NCM(
        num_features=num_features,
        hidden_dim=hidden,
        out_dim=out_dim,
        num_edges=num_edges,
        noise_dim=noise_dim,
        gnn_mode=gnn_mode
    ).to(device)

    # --- 4) Trainer
    tr_cfg = cfg.get("training", {})
    trainer = HybridCausalTrainer(
        epochs=tr_cfg.get("epochs", 200),
        lr=tr_cfg.get("lr", 1e-2),
        gamma=tr_cfg.get("gamma", 0.5),
    )

    # Train with DataLoader (your updated trainer supports this; if not, it will raise and we’ll fall back)
    print("\n--- Training ---")
    try:
        trainer.train(model, train_loader)
    except TypeError:
        # Fallback: older trainer expects a single Data graph; average a few days into one for a quick run
        print("[warn] Trainer didn’t accept a DataLoader. Falling back to single-graph training.")
        # Simple average graph to mimic a day: take first item
        trainer.train(model, g0.to(device))

    # Save model
    ckpt_path = os.path.join(exp_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint → {ckpt_path}")

    # Save training history if available
    hist = _maybe_history_dict(trainer)
    if hist:
        import pandas as pd
        df_hist = pd.DataFrame(hist)
        df_hist.to_csv(os.path.join(exp_dir, "training_curve.csv"), index=False)
        print(f"Saved training curve → {os.path.join(exp_dir, 'training_curve.csv')}")

        # quick loss plot (optional)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(9,4))
            if "loss_obs" in df_hist:   plt.plot(df_hist["epoch"], df_hist["loss_obs"],   label="train obs")
            if "loss_causal" in df_hist:plt.plot(df_hist["epoch"], df_hist["loss_causal"],label="train causal")
            if "loss_total" in df_hist: plt.plot(df_hist["epoch"], df_hist["loss_total"], label="train total", linewidth=2)
            if "val_mse" in df_hist:    plt.plot(df_hist["epoch"], df_hist["val_mse"],    label="val mse", linestyle="--")
            plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Training/Validation Loss")
            plt.grid(True, linestyle="--"); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "loss_curve.png"), dpi=160)
            plt.close()
            print(f"Saved plot → {os.path.join(plots_dir, 'loss_curve.png')}")
        except Exception as e:
            print(f"[warn] Could not plot training curves: {e}")

    # --- 5) Evaluation (MSE on target node)
    print("\n--- Validation (MSE on target) ---")
    model.eval()
    val_sqerr = []
    with torch.no_grad():
        for d in val_loader:
            # d.y format: depending on loader version.
            # In older repo loader, y = [VOL, LIQ] for next day; we compare target dim.
            # If your updated loader produces scalar y for chosen target, handle both.
            if hasattr(d, "to"): d = d.to(device)
            d = _to_device(d, device)
            pred = model(d.x, d.edge_index)  # [num_nodes, 1] or [num_nodes, out_dim?] → assume 1 per node
            # Target extraction
            if d.y.ndim == 1 or (d.y.ndim == 2 and d.y.shape[-1] == 1):
                # scalar target (newer loader)
                y_t = d.y.squeeze()
            else:
                # older loader: y is [VOL, LIQ] (order: likely [VOL, LIQ])
                # Map by indices if columns order is known; else default VOL at pos 0.
                # We created tgt_idx above, but y doesn’t carry per-node alignment—so fallback to position.
                # If your updated loader has y aligned to target_node, it’s just y[..., 0].
                pos = 0  # VOL at position 0 in repo loader
                y_t = d.y[..., pos].squeeze()

            y_hat_t = pred[tgt_idx].squeeze()
            val_sqerr.append((y_hat_t - y_t).float().pow(2).mean().item())

    val_mse = sum(val_sqerr) / max(1, len(val_sqerr))
    print(f"val_mse[{target_node}] = {val_mse:.6f}")

    # --- 6) Robustness under a shock (observational OOD test)
    # multiply shock node’s feature by factor on validation set and see error lift.
    print("\n--- Robustness under synthetic shock (observational) ---")
    shock_factor = cfg.get("analysis", {}).get("shock_factor", 5.0)
    mse_normal, mse_shock = [], []

    with torch.no_grad():
        for d in val_loader:
            d = _to_device(d, device)
            # normal
            p_norm = model(d.x, d.edge_index)[tgt_idx].squeeze()
            if d.y.ndim > 1: y_t = d.y[..., 0].squeeze()
            else:            y_t = d.y.squeeze()
            mse_normal.append(F.mse_loss(p_norm, y_t).item())

            # shock
            x_shock = d.x.clone()
            # supports 1 feature per node or >1: scale entire row
            x_shock[shock_idx] = x_shock[shock_idx] * shock_factor
            p_shock = model(x_shock, d.edge_index)[tgt_idx].squeeze()
            mse_shock.append(F.mse_loss(p_shock, y_t).item())

    robust_lift = (_as_float(sum(mse_shock)/len(mse_shock)) -
                   _as_float(sum(mse_normal)/len(mse_normal)))
    print(f"MSE normal={sum(mse_normal)/len(mse_normal):.6f} | shock={sum(mse_shock)/len(mse_shock):.6f} | Δ={robust_lift:.6f}")

    # --- 7) ATE via do-operation (if model supports do_intervention)
    print("\n--- ATE (do-operation) sanity check ---")
    ate_est, ate_true = None, None
    try:
        has_do = hasattr(model, "do_intervention")
        if has_do:
            # Single held-out graph for clarity (first of val)
            d = next(iter(val_loader))
            d = _to_device(d, device)
            # Base preds
            p_before = model(d.x, d.edge_index).detach()

            # do-intervention: add +1 to shock node’s feature vector (or set to zeros+1)
            x_do = d.x.clone()
            if x_do.ndim == 2:
                x_do[shock_idx] = x_do[shock_idx] + 1.0
            else:
                x_do[shock_idx, 0] = x_do[shock_idx, 0] + 1.0

            p_after = model.do_intervention(
                d.x, d.edge_index,
                intervened_nodes=torch.tensor([shock_idx], device=device),
                new_feature_values=x_do[shock_idx].unsqueeze(0)
            ).detach()

            ate_est = (p_after - p_before)[tgt_idx].mean().item()

            # "true" ATE is unknown on real data; we just report estimated ATE.
            # (On synthetic we derive true_ate from the known SEM.)
            print(f"Estimated ATE do({shock_node}:+1) → {target_node}: {ate_est:.6f}")
        else:
            print("Model has no do_intervention(); skipping ATE.")
    except Exception as e:
        print(f"[warn] ATE estimation failed: {e}")

    # --- 8) Save results
    results = {
        "experiment": exp_name,
        "device": str(device),
        "target_node": target_node,
        "shock_node": shock_node,
        "val_mse_target": val_mse,
        "robustness_mse_normal": sum(mse_normal)/len(mse_normal),
        "robustness_mse_shock":  sum(mse_shock)/len(mse_shock),
        "robustness_mse_delta":  robust_lift,
    }
    if ate_est is not None:
        results["estimated_ate"] = ate_est

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results → {os.path.join(exp_dir, 'results.json')}\n")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base_config.yaml",
                   help="Path to YAML config.")
    args = p.parse_args()
    run_experiment(args.config)
