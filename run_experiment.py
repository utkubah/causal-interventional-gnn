# run_experiment.py
import os
import json
import argparse
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from src.utils import set_seed, load_config
from src.dataloader import CausalFactorDataset
from src.models import GNN_NCM
from src.trainer import HybridCausalTrainer

def run_experiment(config_path: str):
    """
    Main function to run a single experiment from a config file.
    """
    # --- 1. Setup ---
    config = load_config(config_path)
    set_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    
    exp_dir = os.path.join(config['output_dir'], config['experiment_name'])
    os.makedirs(exp_dir, exist_ok=True)
    print(f"--- Running Experiment: {config['experiment_name']} ---")
    print(f"Using device: {device}")

    # --- 2. Data Loading ---
    # We assume the `data/processed` directory is relative to the project root
    train_dataset = CausalFactorDataset(root_dir="data/processed", train=True)
    test_dataset = CausalFactorDataset(root_dir="data/processed", train=False)
    
    # We use a batch size of 1 because each graph is a single day's snapshot
    # In a real application, you might batch days together, but this is clearer.
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    num_features = train_dataset.num_features
    num_nodes = train_dataset.num_nodes
    num_edges = train_dataset.edge_index.size(1)

    # --- 3. Model Initialization ---
    model = GNN_NCM(
        num_features=num_features,
        hidden_dim=config['model']['hidden_dim'],
        out_dim=config['model']['out_dim'],
        num_edges=num_edges,
        noise_dim=config['model']['noise_dim'],
        gnn_mode=config['model']['gnn_mode']
    ).to(device)

    # --- 4. Training ---
    trainer = HybridCausalTrainer(
        model=model,
        epochs=config['training']['epochs'],
        lr=config['training']['lr'],
        gamma=config['training']['gamma']
    )
    # Note: For simplicity, the trainer will train on the entire train_dataset
    # A more complex setup would use the loader.
    trained_model = trainer.train(train_dataset)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), os.path.join(exp_dir, "model.pt"))
    print(f"\nModel saved to {os.path.join(exp_dir, 'model.pt')}")

    # --- 5. Evaluation ---
    print("\n--- Evaluating Model ---")
    trained_model.eval()
    all_preds = []
    all_trues = []
    all_dates = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            # The model predicts one value per node. We only care about VOL and LIQ.
            # Assuming VOL is the last node and LIQ is the second to last.
            # A more robust way would be to use the node_mapping.
            vol_idx = test_dataset.node_mapping['VOL']
            liq_idx = test_dataset.node_mapping['LIQ']
            
            pred_full = trained_model(data.x, data.edge_index)
            pred_targets = torch.cat([pred_full[vol_idx], pred_full[liq_idx]])

            all_preds.append(pred_targets.cpu())
            all_trues.append(data.y.cpu().squeeze())
            all_dates.append(data.date[0])

    preds_tensor = torch.stack(all_preds)
    trues_tensor = torch.stack(all_trues)
    
    # Calculate overall MSE
    overall_mse_vol = F.mse_loss(preds_tensor[:, 0], trues_tensor[:, 1]).item()
    overall_mse_liq = F.mse_loss(preds_tensor[:, 1], trues_tensor[:, 0]).item()
    
    # --- Robustness under Shocks ---
    # Define hardcoded shock periods for evaluation
    shock_periods = [
        ("COVID19_Crash", "2020-02-20", "2020-04-01"),
        ("Global_Financial_Crisis_Peak", "2008-09-01", "2008-12-31")
    ]
    results = {'overall_mse_vol': overall_mse_vol, 'overall_mse_liq': overall_mse_liq}
    df_results = pd.DataFrame({'date': all_dates, 'pred_vol': preds_tensor[:, 0], 'true_vol': trues_tensor[:, 1]})
    df_results['date'] = pd.to_datetime(df_results['date'])

    for name, start, end in shock_periods:
        shock_df = df_results[(df_results['date'] >= start) & (df_results['date'] <= end)]
        if not shock_df.empty:
            shock_mse = F.mse_loss(torch.tensor(shock_df['pred_vol'].values), torch.tensor(shock_df['true_vol'].values)).item()
            results[f'shock_mse_vol_{name}'] = shock_mse
    
    print("Evaluation Results:", results)
    
    # Save results to a file
    with open(os.path.join(exp_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {os.path.join(exp_dir, 'results.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Path to the config file for the experiment.")
    args = parser.parse_args()
    run_experiment(args.config)






