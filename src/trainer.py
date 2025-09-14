import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from src.models import TeacherGNN

class HybridCausalTrainer:
    def __init__(self, epochs=50, lr=0.01, gamma=0.5):
        self.epochs = epochs; self.lr = lr; self.gamma = gamma; self.loss_fn = nn.MSELoss()

    def _is_loader(self, obj):
        # treat as loader if it's iterable and not a single Data object
        return hasattr(obj, "__iter__") and not hasattr(obj, "x")

    def train(self, model_to_train, data_or_loader):
        if self._is_loader(data_or_loader):
            return self._train_over_loader(model_to_train, data_or_loader)
        else:
            return self._train_on_graph(model_to_train, data_or_loader)

    def _train_on_graph(self, model_to_train, g):
        device = g.x.device
        model_to_train = model_to_train.to(device)
        opt = optim.Adam(model_to_train.parameters(), lr=self.lr)

        in_dim = g.x.size(-1)
        teacher = TeacherGNN(in_dim, 16, 8).to(device)
        opt_t = optim.Adam(teacher.parameters(), lr=0.01)

        print("--- Pre-training Teacher Model ---")
        for _ in range(100):
            opt_t.zero_grad()
            out = teacher(g.x, g.edge_index)
            loss_t = F.mse_loss(out, g.y)
            loss_t.backward(); opt_t.step()
        teacher.eval(); print("Teacher model trained.")

        print("\n--- Starting Hybrid Causal Training ---")
        for epoch in range(self.epochs):
            model_to_train.train(); opt.zero_grad()

            preds = model_to_train(g.x, g.edge_index)
            loss_obs = self.loss_fn(preds, g.y)

            with torch.no_grad():
                tpreds = teacher(g.x, g.edge_index)

            loss_causal = 0.0; count = 0
            src = g.edge_index[0]; dst = g.edge_index[1]
            for v in range(g.num_nodes):
                parents = src[dst == v]
                if parents.numel() == 0: continue
                count += 1
                ivals = []
                for p in parents:
                    p = p.item()
                    after = model_to_train.do_intervention(
                        g.x, g.edge_index,
                        intervened_nodes=torch.tensor([p], device=device),
                        new_feature_values=g.x.new_zeros(1, g.x.size(-1))
                    )
                    ivals.append(after[v])
                ivals = torch.stack(ivals, dim=0).mean(dim=0)
                loss_causal = loss_causal + self.loss_fn(ivals, tpreds[v])

            if count > 0: loss_causal = loss_causal / count
            total = loss_obs + self.gamma * loss_causal
            total.backward(); opt.step()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:03d} | Total {total.item():.4f} (Obs {loss_obs.item():.4f}, Causal {loss_causal.item():.4f})")
        print("Training finished.")

    def _train_over_loader(self, model_to_train, loader):
        # get a fresh iterator every time
        it = iter(loader)
        first = next(it)
        device = first.x.device
        model_to_train = model_to_train.to(device)
        opt = optim.Adam(model_to_train.parameters(), lr=self.lr)

        in_dim = first.x.size(-1)
        teacher = TeacherGNN(in_dim, 32, 16).to(device)
        opt_t = optim.Adam(teacher.parameters(), lr=0.01)

        print("--- Pre-training Teacher Model (on first batch) ---")
        for _ in range(200):
            opt_t.zero_grad()
            out = teacher(first.x, first.edge_index)
            loss_t = F.mse_loss(out, first.y)
            loss_t.backward(); opt_t.step()
        teacher.eval(); print("Teacher model trained.")

        print("\n--- Starting Hybrid Causal Training over dataset ---")
        for epoch in range(self.epochs):
            for g in loader:
                g = g.to(device)
                model_to_train.train(); opt.zero_grad()

                preds = model_to_train(g.x, g.edge_index)
                loss_obs = self.loss_fn(preds, g.y)

                with torch.no_grad():
                    tpreds = teacher(g.x, g.edge_index)

                loss_causal = 0.0; count = 0
                src = g.edge_index[0]; dst = g.edge_index[1]
                for v in range(g.num_nodes):
                    parents = src[dst == v]
                    if parents.numel() == 0: continue
                    count += 1
                    ivals = []
                    for p in parents:
                        p = p.item()
                        after = model_to_train.do_intervention(
                            g.x, g.edge_index,
                            intervened_nodes=torch.tensor([p], device=device),
                            new_feature_values=g.x.new_zeros(1, g.x.size(-1))
                        )
                        ivals.append(after[v])
                    ivals = torch.stack(ivals, dim=0).mean(dim=0)
                    loss_causal = loss_causal + self.loss_fn(ivals, tpreds[v])

                if count > 0: loss_causal = loss_causal / count
                total = loss_obs + self.gamma * loss_causal
                total.backward(); opt.step()

            if (epoch + 1) % 20 == 0:
                print(f"[ep {epoch+1:03d}] total={total.item():.4f} (obs={loss_obs.item():.4f}, causal={loss_causal.item():.4f})")
        print("Training finished over dataset.")

    @torch.no_grad()
    def evaluate(self, model, data_or_loader, target_idx=None):
        model.eval()
        device = next(model.parameters()).device

        # DataLoader path
        if hasattr(data_or_loader, "__iter__") and not hasattr(data_or_loader, "x"):
            total, n = 0.0, 0
            for g in data_or_loader:
                g = g.to(device)  # <<< move batch to model device
                preds = model(g.x, g.edge_index)  # [num_nodes, 1] usually

                # choose target
                if target_idx is None:
                    y = g.y
                    p = preds
                else:
                    # if g.y is scalar for the chosen target
                    if g.y.ndim == 1 or (g.y.ndim == 2 and g.y.shape[-1] == 1):
                        y = g.y.squeeze()
                        p = preds[target_idx].squeeze()
                    else:
                        # if g.y has per-node targets
                        y = g.y[target_idx]
                        p = preds[target_idx]

                total += self.loss_fn(p, y).item()
                n += 1
            return total / max(n, 1)

        # Single-graph path
        g = data_or_loader.to(device)
        preds = model(g.x, g.edge_index)
        if target_idx is None:
            return self.loss_fn(preds, g.y).item()
        else:
            if g.y.ndim == 1 or (g.y.ndim == 2 and g.y.shape[-1] == 1):
                return self.loss_fn(preds[target_idx].squeeze(), g.y.squeeze()).item()
            return self.loss_fn(preds[target_idx], g.y[target_idx]).item()
