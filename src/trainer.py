import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from src.models import TeacherGNN

class HybridCausalTrainer:
    """Trains a GNN-NCM using a hybrid loss for performance and causal consistency."""
    def __init__(self, model, epochs=200, lr=0.01, gamma=0.5):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _train_teacher(self, graph_data):
        teacher_model = TeacherGNN(graph_data.num_features, 16, 8).to(graph_data.x.device)
        optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=0.01)
        print("--- Pre-training Teacher Model ---")
        for _ in range(100):
            out = teacher_model(graph_data.x, graph_data.edge_index)
            loss = F.mse_loss(out, graph_data.y)
            loss.backward()
            optimizer_teacher.step()
        teacher_model.eval()
        print("Teacher model trained.")
        return teacher_model

    def train(self, graph_data):
        teacher_model = self._train_teacher(graph_data)
        
        print("\n--- Starting Hybrid Causal Training ---")
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # 1. Observational Loss (Direct-to-Label)
            obs_preds = self.model(graph_data.x, graph_data.edge_index)
            loss_obs = self.loss_fn(obs_preds, graph_data.y)
            
            # 2. Causal Consistency Loss (Structural Regularizer)
            loss_causal = 0
            with torch.no_grad():
                teacher_preds = teacher_model(graph_data.x, graph_data.edge_index).detach()
            
            num_causal_nodes = 0
            for v_idx in range(graph_data.num_nodes):
                parents = graph_data.edge_index[0][graph_data.edge_index[1] == v_idx]
                if len(parents) > 0:
                    num_causal_nodes += 1
                    interventional_preds = []
                    for parent_idx in parents:
                        pred = self.model.do_intervention(
                            graph_data.x, graph_data.edge_index,
                            intervened_nodes=torch.tensor([parent_idx]),
                            new_feature_values=torch.zeros(1, graph_data.num_features)
                        )
                        interventional_preds.append(pred[v_idx])
                    
                    causally_derived_pred = torch.stack(interventional_preds).mean(dim=0)
                    loss_causal += self.loss_fn(causally_derived_pred, teacher_preds[v_idx])

            if num_causal_nodes > 0:
                loss_causal /= num_causal_nodes
            
            # 3. Combined Loss
            total_loss = loss_obs + self.gamma * loss_causal
            total_loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:03d} | Total Loss: {total_loss.item():.4f} "
                      f"(Obs: {loss_obs.item():.4f}, Causal: {loss_causal.item():.4f})")
        print("Training finished.")
        return self.model