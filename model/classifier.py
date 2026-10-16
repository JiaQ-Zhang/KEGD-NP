"""Original classifier architecture, extracted without changing its forward logic."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kan2 import ImprovedKAAGCN

class ExplainableGlobalEnhancer(nn.Module):
    def __init__(self, input_dim=208, hidden_dim=208, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(input_dim)
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        weights = self.gate(x)
        weighted_x = x * weights
        weighted_x = self.norm(weighted_x)
        out = self.project(weighted_x)
        return out, weights, weighted_x

class GraphCCS_MD_Classifier_Explainable(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone = ImprovedKAAGCN(
            N=config["gnn_N"],
            dim=config["gnn_dim"],
            layer_hidden=config["gnn_hidden"],
            layer_output=config["gnn_output"],
            heads=config["gnn_heads"],
            grid_size=config["gnn_grid"],
            spline_order=3,
            dropout_rate=config["gnn_dropout"],
        )

        self.graph_embed_dim = config["gnn_dim"] * config["gnn_heads"]
        self.global_feat_size = config["global_feat_size"]

        print(f"🔗 GNN输出: {config['gnn_dim']}×{config['gnn_heads']}={self.graph_embed_dim}维")
        print(f"🔗 全局输入: {self.global_feat_size}维")

        self.global_enhancer = ExplainableGlobalEnhancer(
            input_dim=self.global_feat_size,
            hidden_dim=208,
            dropout=0.1
        )

        self.graph_norm = nn.LayerNorm(self.graph_embed_dim)
        self.global_norm = nn.LayerNorm(208)

        self.gate = nn.Sequential(
            nn.Linear(self.graph_embed_dim + 208, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        self.graph_project = nn.Linear(self.graph_embed_dim, 208)

        self.graph_project = nn.Linear(self.graph_embed_dim, 208)
        self.global_project = nn.Linear(208, 208)

        self.fuse_norm = nn.LayerNorm(208)


        self.out = nn.Sequential(
            nn.Linear(208, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, data_batch, global_feat, return_weights=False):
        graph_vec = self.backbone.forward_feature(data_batch)

        if isinstance(global_feat, np.ndarray):
            global_feat = torch.tensor(global_feat, dtype=torch.float32).to(graph_vec.device)
        else:
            global_feat = global_feat.to(graph_vec.device)

        enhanced_global, feature_weights, weighted_global = self.global_enhancer(global_feat)

        graph_vec = self.graph_norm(graph_vec)
        enhanced_global = self.global_norm(enhanced_global)

        gate_input = torch.cat([graph_vec, enhanced_global], dim=1)
        alpha = torch.sigmoid(self.gate(gate_input))

        graph_emb = self.graph_project(graph_vec)
        if self.graph_embed_dim >= 208:
            residual = graph_vec[:, :208]
        else:
            residual = F.linear(graph_vec, weight=self.graph_project.weight[:208], bias=None)
        graph_emb = graph_emb + residual


        global_emb = self.global_project(enhanced_global)

        x = alpha * graph_emb + (1 - alpha) * global_emb
        x = self.fuse_norm(x)
        logits = self.out(x)

        if return_weights:
            return logits, alpha.squeeze(1), (1 - alpha).squeeze(1), feature_weights, weighted_global
        return logits
