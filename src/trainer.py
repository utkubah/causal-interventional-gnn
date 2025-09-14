import torch, torch.nn as nn, torch.optim as optim

class CausalTwoPartTrainer:
    def __init__(self,
                 epochs_obs=30,
                 epochs_do=150,
                 lr=5e-3,
                 w_obs=0.2,
                 w_do=1.0,
                 weight_decay=1e-4,
                 clip=1.0,
                 # knobs for your causal do(parent) construction
                 neutral='zeros',     # 'zeros' or 'self+delta'
                 delta=0.0):          # scalar or 1D tensor of size F (used when neutral == 'self+delta')
        self.epochs_obs = int(epochs_obs)
        self.epochs_do  = int(epochs_do)
        self.lr  = float(lr)
        self.w_obs = float(w_obs)
        self.w_do  = float(w_do)
        self.wd  = float(weight_decay)
        self.clip = float(clip)

        self.neutral = str(neutral)
        # delta can be float or tensor; we’ll broadcast to feature size later
        self.delta = delta

        self.loss = nn.MSELoss()
        self.history = []

    
    # expects ONE DataLoader of observational snapshots (each g has: x, edge_index, y)
    def train(self, model, loader, val_loader=None, val_node_idx=None):
        dev = next(model.parameters()).device
        model = model.to(dev)

        # -------- Phase 1: observational warm-up (obs only) --------
        opt = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        for ep in range(1, self.epochs_obs + 1):
            model.train()
            obs_sum, n_obs = 0.0, 0
            for g in loader:
                g = g.to(dev)
                pred = model(g.x, g.edge_index)
                l_obs = self.loss(pred, g.y)

                opt.zero_grad()
                l_obs.backward()
                if self.clip: torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
                opt.step()

                obs_sum += float(l_obs.detach()); n_obs += 1

            m_obs = obs_sum / n_obs
            m_val= self.evaluate_obs_mse(model, val_loader, node_idx=val_node_idx)

            self.history.append({
                "epoch": ep, "phase": "obs",
                "loss_obs": m_obs, "loss_do": None, "loss_total": m_obs,
                "val_obs": m_val
            })
            if ep % 10 == 0:
                msg = f"[obs {ep:03d}] obs={m_obs:.6f}"
                if m_val is not None: msg += f" | val_obs={m_val:.6f}"
                print(msg)


        # -------- Phase 2: obs + teacher-free causal (one combined step per batch) --------
        # reset optimizer to avoid stale momentum from Phase 1
        opt = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)

        
        for ep in range(1, self.epochs_do + 1):
            model.train()
            obs_sum, do_sum, n_obs, n_do = 0.0, 0.0, 0, 0

            for g in loader:
                g = g.to(dev)
                x, edge_index, y = g.x, g.edge_index, g.y

                # observational term
                p_obs = model(x, edge_index)
                l_obs = self.loss(p_obs, y)
                obs_sum += float(l_obs.detach()); n_obs += 1

                # teacher-free causal term: do(parent) one at a time, aggregate to each child
                l_cau = self._causal_loss_do_parent_average(model, g, p_obs)
                do_sum += float(l_cau.detach()); n_do += 1

                # combine and step once
                total = (self.w_obs * l_obs) + (self.w_do * l_cau)
                opt.zero_grad()
                total.backward()
                if self.clip: torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
                opt.step()

            m_obs = obs_sum / max(n_obs, 1) if n_obs else 0.0
            m_do  = do_sum  / max(n_do,  1) if n_do  else 0.0
            total_epoch = (self.w_obs * m_obs) + (self.w_do * m_do if n_do else 0.0)

            m_val= self.evaluate_obs_mse(model, val_loader, node_idx=val_node_idx)

            ep_abs = self.epochs_obs + ep
            self.history.append({
                "epoch": ep_abs, "phase": "do",
                "loss_obs": m_obs, "loss_do": m_do, "loss_total": total_epoch,
                "val_obs": m_val
            })
            if ep % 10 == 0:
                msg = f"[do  {ep:03d}] total={total_epoch:.6f} (obs={m_obs:.6f}, do={m_do:.6f})"
                if m_val is not None: msg += f" | val_obs={m_val:.6f}"
                print(msg)


        return model

    def _causal_loss_do_parent_average(self, model, g, p_obs):
        """
        Implements your causal loss verbatim:
          - For each UNIQUE parent node p (from edges src->dst), compute prediction under do(p)
            where features of p are set to a neutral row ('zeros' or 'self+delta').
          - For each child v, average the do(p)[v] over all parents p of v to form a target.
          - Compare that target to TRUE y[v] (MSE), average over nodes that have parents.

        No teacher labels (y_do) and no masks are used.
        """
        dev = p_obs.device
        x, edge_index, y = g.x, g.edge_index, g.y
        N, F = x.size(0), x.size(1)

        # edges assumed in PyG convention: [2, E] with src = edge_index[0], dst = edge_index[1]
        src, dst = edge_index[0], edge_index[1]
        if src.numel() == 0:
            # no edges -> no parents -> causal term 0 (preserve grad graph/dtype)
            return 0.0 * p_obs.sum()

        unique_parents = torch.unique(src)

        # prepare delta row if needed
        if isinstance(self.delta, torch.Tensor):
            delta_row = self.delta.to(device=dev, dtype=x.dtype)
        else:
            delta_row = torch.full((F,), float(self.delta), device=dev, dtype=x.dtype)

        # compute do(parent) predictions once per parent
        p1_map = {}
        for p in unique_parents.tolist():
            p = int(p)
            if self.neutral == 'zeros':
                new_row = torch.zeros(F, device=dev, dtype=x.dtype)
            else:  # 'self+delta'
                new_row = x[p] + delta_row

            if hasattr(model, "do_intervention") and callable(getattr(model, "do_intervention")):
                p1 = model.do_intervention(
                    x, edge_index,
                    intervened_nodes=torch.tensor([p], dtype=torch.long, device=dev),
                    new_feature_values=new_row.unsqueeze(0)
                )  # [N, ...]
            else:
                # fallback: override node p's features and run a forward pass
                x_do = x.clone()
                x_do[p] = new_row
                p1 = model(x_do, edge_index)  # [N, ...]
            p1_map[p] = p1

        # aggregate targets per child and compare to TRUE y[v]
        loss_causal = 0.0
        count = 0
        for v in range(N):
            parents_v = src[dst == v]
            if parents_v.numel() == 0:
                continue
            vals = []
            for p in parents_v.tolist():
                if p in p1_map:
                    vals.append(p1_map[p][v])  # prediction at node v under do(p)
            if not vals:
                continue
            target_v = torch.stack(vals, dim=0).mean(dim=0)  # averaged target for node v
            loss_causal = loss_causal + self.loss(target_v, y[v])
            count += 1

        return (loss_causal / count) if count > 0 else (0.0 * p_obs.sum())

    @torch.no_grad()
    def evaluate_obs_mse(self, model, loader, node_idx=None):
        model.eval()
        dev = next(model.parameters()).device
        tot, n = 0.0, 0
        for g in loader:
            g = g.to(dev)
            p = model(g.x, g.edge_index)   # [N], [N,1], or [N,C]
            y = g.y                        # [N], [N,1], or [N,C]

            # --- safe, few-line fix for single-node eval ---
            if node_idx is not None and p.shape[0] == y.shape[0]:
                idx = node_idx if torch.is_tensor(node_idx) else torch.tensor([int(node_idx)], device=dev)
                idx = idx.to(torch.long).view(-1)
                p = p.index_select(0, idx)
                y = y.index_select(0, idx)
            # align [N,1] ↔ [N]
            if p.dim()==2 and p.size(-1)==1: p = p.squeeze(-1)
            if y.dim()==2 and y.size(-1)==1: y = y.squeeze(-1)
            # -----------------------------------------------

            tot += self.loss(p, y).item(); n += 1
        return tot / max(n, 1)

