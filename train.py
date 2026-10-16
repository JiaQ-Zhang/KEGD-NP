import os
import numpy as np
import pandas as pd
from collections import Counter



os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import matplotlib.pyplot as plt
import math  # 🔥 添加这一行


import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from rdkit import Chem
from tqdm import tqdm


from torch.utils.data import Dataset, DataLoader



import os
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_DIR = str(ROOT / "model")

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
from kan2 import ImprovedKAAGCN



from rdkit.Chem import Descriptors
from sklearn.preprocessing import normalize

import torch.serialization


from preprocess import *

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import matplotlib.pyplot as plt






def smiles_to_dgl_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    node_featurizer = featurize_atoms
    edge_featurizer = BondFeaturizer(bond_data_field='e', self_loop=True)
    g = mol_to_bigraph(
        mol=mol,
        node_featurizer=node_featurizer,
        edge_featurizer=edge_featurizer,
        add_self_loop=True,
        explicit_hydrogens=True
    )
    return g




from rdkit.Chem import Descriptors
from sklearn.preprocessing import normalize
import numpy as np

DESC_NAMES = [d[0] for d in Descriptors.descList]


def smiles_to_global_features(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    values = []
    for name in DESC_NAMES:
        func = getattr(Descriptors, name)
        try:
            v = func(mol)
            if v is None or np.isnan(v) or np.isinf(v):
                v = 0.0
            v = float(max(min(v, 1e6), -1e6))
        except Exception:
            v = 0.0
        values.append(v)


    if not hasattr(smiles_to_global_features, '_debug_printed'):
        print(f"✅ RDKit描述符维度确认: {len(DESC_NAMES)}维")
        smiles_to_global_features._debug_printed = True

    arr = np.array(values, dtype=np.float32).reshape(1, -1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = normalize(arr, axis=1, norm="max")
    return arr.squeeze(0)




import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from rdkit import Chem

import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch


class NPBS_DGL_Dataset(Dataset):
    def __init__(self, csv_path, label_col="Marine", encoding="utf-8-sig", use_cache=True):
        super().__init__()
        self.cache_dir = csv_path.rsplit('.', 1)[0] + f"_{label_col}_dgl_cache"
        self.cache_file = os.path.join(self.cache_dir, "dgl_graphs.pt")
        self.csv_path = csv_path
        self.label_col = label_col


        self._load_or_build_fingerprint_dict(use_cache=use_cache)

        print(f"📖 读取数据集: {csv_path}")
        df = pd.read_csv(csv_path, encoding=encoding)

        label_num_col = f"{label_col}_num"
        if label_num_col not in df.columns:
            available_cols = [col for col in df.columns if col.endswith('_num')]
            raise ValueError(f"标签列 '{label_num_col}' 不存在! 可用列: {available_cols}")

        df = df[df[label_num_col].notna()].copy()
        df = df[df['SMILES'].notna()]
        print(f"✅ 找到 {len(df)} 条有效 {label_col} 记录")

        self.smiles = df['SMILES'].astype(str).tolist()
        self.labels = np.round(df[label_num_col].astype(float), 4).tolist()


        self.graphs = None
        self.globals = None
        self.y = None
        self.indices = None
        self.fingerprints_list = None
        self.adjacencies_list = None
        self.molecular_sizes_list = None


        if use_cache and os.path.exists(self.cache_file):
            self._try_load_cache()
        else:
            print(f"⚡ 尝试加载缓存: {self.cache_file}")
            self._build_data()

    def _load_or_build_fingerprint_dict(self, use_cache=True):

        import os
        import pickle
        from preprocess import fingerprint_dict

        dict_path = os.path.join(self.cache_dir, 'fingerprint_dict.pickle')

        if use_cache and os.path.exists(dict_path):
            print(f"✅ 从缓存加载 fingerprint_dict: {dict_path}")
            with open(dict_path, 'rb') as f:
                c = pickle.load(f)
                for k in c.keys():
                    fingerprint_dict[k] = c[k]
        else:

            print(f"🔍 正在为 {self.label_col} 构建专属 fingerprint_dict...")

            df = pd.read_csv(self.csv_path, encoding="gbk")
            label_num_col = f"{self.label_col}_num"
            df = df[df[label_num_col].notna()].copy()
            df = df[df['SMILES'].notna()]
            smiles_list = df['SMILES'].astype(str).tolist()

            for smi in tqdm(smiles_list, desc="Building fingerprint_dict"):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                mol = Chem.AddHs(mol)


                global_feat = smiles_to_global_features(smi)
                if global_feat is None or not np.any(global_feat):
                    continue

                atoms = create_atoms(mol, atom_dict)
                i_jbond_dict = create_ijbonddict(mol, bond_dict)
                fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict)


            os.makedirs(self.cache_dir, exist_ok=True)
            with open(dict_path, 'wb') as f:
                pickle.dump(dict(fingerprint_dict), f)
            print(f"✅ 专属 fingerprint_dict 已保存: {dict_path}")

    def _try_load_cache(self):
        """宽松缓存加载，只加载非图数据"""
        try:
            print(f"⚡ 尝试加载缓存: {self.cache_file}")
            cache_data = torch.load(self.cache_file, map_location='cpu', weights_only=False)

            # 简单校验：label_col 和文件路径
            if (cache_data.get('label_col') == self.label_col and
                    cache_data.get('csv_path') == self.csv_path):
                self.globals = cache_data['globals']
                self.y = cache_data['labels']
                self.indices = cache_data.get('indices', list(range(len(self.y))))
                self.fingerprints_list = cache_data.get('fingerprints_list', [])
                self.adjacencies_list = cache_data.get('adjacencies_list', [])
                self.molecular_sizes_list = cache_data.get('molecular_sizes_list', [])

                print(f"✅ 缓存加载成功！global_feat: {np.array(self.globals[0]).shape}")
                print(f"[ {self.label_col} ] 总有效样本数: {len(self.y)}")
                print(f"[ {self.label_col} ] 标签分布: {Counter(self.y)}")
                return

            print("⚠️ 缓存校验失败，重新构建...")
        except Exception as e:
            print(f"❌ 缓存读取失败: {e}")

        self._build_data()

    def _build_data(self):
        """构建数据（保持原始函数调用，完全兼容）"""
        print(f"🔄 开始构建数据 (缓存: {self.cache_file})...")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.globals, self.y, self.indices = [], [], []
        self.fingerprints_list, self.adjacencies_list, self.molecular_sizes_list = [], [], []

        for i, (smi, lab) in enumerate(tqdm(zip(self.smiles, self.labels), total=len(self.smiles))):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)


            global_feat = smiles_to_global_features(smi)
            if global_feat is None or not np.any(global_feat):
                continue

            atoms = create_atoms(mol, atom_dict)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict)
            adjacency = np.float32(Chem.GetAdjacencyMatrix(mol))


            fp_len = len(fingerprints)
            adj_size = adjacency.shape[0]

            if fp_len != adj_size:
                print(f"⚠️ idx={i} SMILES={smi[:50]}... 被调整，fp={fp_len}, adj={adj_size}")
                molecular_size = min(fp_len, adj_size)
                fingerprints = fingerprints[:molecular_size]
                adjacency = adjacency[:molecular_size, :molecular_size]
            else:
                molecular_size = adj_size
            fingerprints_tensor = torch.LongTensor(fingerprints)
            adjacency_tensor = torch.FloatTensor(adjacency)



            self.globals.append(global_feat)
            self.y.append(lab)
            self.indices.append(i)
            self.fingerprints_list.append(fingerprints_tensor)
            self.adjacencies_list.append(adjacency_tensor)
            self.molecular_sizes_list.append(molecular_size)


        cache_data = {
            'globals': self.globals,
            'labels': self.y,
            'indices': self.indices,
            'csv_path': self.csv_path,
            'label_col': self.label_col,
            'csv_mtime': os.path.getmtime(self.csv_path),
            'fingerprints_list': self.fingerprints_list,
            'adjacencies_list': self.adjacencies_list,
            'molecular_sizes_list': self.molecular_sizes_list
        }
        torch.save(cache_data, self.cache_file)
        print(f"💾 缓存已保存: {self.cache_file} ({len(self.y)} 样本)")
        print(f"[ {self.label_col} ] 总有效样本数: {len(self.y)}")
        print(f"[ {self.label_col} ] 标签分布: {Counter(self.y)}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        smi = self.smiles[self.indices[idx]]
        fingerprints_tensor = self.fingerprints_list[idx]  # .to('cpu')
        adjacency_tensor = self.adjacencies_list[idx]  # .to('cpu')
        molecular_size = self.molecular_sizes_list[idx]
        global_feat = self.globals[idx]

        if len(fingerprints_tensor) != adjacency_tensor.shape[0] != molecular_size:
            print(
                f"❌ idx={idx} 样本存在不一致: fp={len(fingerprints_tensor)}, adj={adjacency_tensor.shape[0]}, size={molecular_size}")
            raise ValueError("样本数据不一致，无法继续训练")

        assert len(fingerprints_tensor) == adjacency_tensor.shape[0] == molecular_size, \
            f"idx={idx}: fp={len(fingerprints_tensor)}, adj={adjacency_tensor.shape[0]}, size={molecular_size}"



        return (
            smi,
            fingerprints_tensor,
            adjacency_tensor,
            molecular_size,
            global_feat,
            float(self.y[idx])
        )

    def clear_cache(self):
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
            print(f"🗑️ 已删除缓存: {self.cache_file}")
            # print(f"✅ 保留字典文件: {os.path.join(self.cache_dir, 'fingerprint_dict.pickle')}")

    @staticmethod
    def clear_all_caches(csvbasepath, labelcols):
        for label in labelcols:
            cachedir = csvbasepath.rsplit('.', 1)[0] + f"_{label}_dgl_cache"
            if os.path.exists(cachedir):
                import shutil
                shutil.rmtree(cachedir)
                print(f"🗑️ 已删除 {label} 缓存目录: {cachedir}")

    @staticmethod
    def check_cache_status(csvpath, labelcol):
        cachedir = csvpath.rsplit('.', 1)[0] + f"_{labelcol}_dgl_cache"
        cachefile = os.path.join(cachedir, "dgl_graphs.pt")
        if os.path.exists(cachefile):
            size_gb = os.path.getsize(cachefile) / 1e9
            print(f"✅ {labelcol} 缓存存在，大小: {size_gb:.1f}GB")
        else:
            print(f"❌ {labelcol} 无缓存")


def clear_all_caches(csv_base_path, label_cols):

    for label in label_cols:
        cache_dir = csv_base_path.rsplit('.', 1)[0] + f"_{label}_dgl_cache"
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print(f"🗑️  已清理 {label} 缓存: {cache_dir}")


def check_cache_status(csv_path, label_col):

    cache_dir = csv_path.rsplit('.', 1)[0] + f"_{label_col}_dgl_cache"
    cache_file = os.path.join(cache_dir, "dgl_graphs.pt")
    if os.path.exists(cache_file):
        print(f"✅ {label_col} 缓存存在，大小: {os.path.getsize(cache_file) / 1e9:.1f}GB")
    else:
        print(f"❌ {label_col} 无缓存")


def dgl_collate_fn(batch):

    graphs, globals_arr, labels = zip(*batch)
    bg = dgl.batch(graphs)

    globals_arr = torch.tensor(np.stack(globals_arr, axis=0), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    return bg, globals_arr, y


def improved_kaagcn_collate_fn(batch):
    smiles, fingerprints, adjacencies, molecularsizes, globalfeat, labels = zip(*batch)


    data_batch = (
        list(smiles),
        list(fingerprints),
        list(adjacencies),
        list(molecularsizes)
    )

    global_tensor = torch.tensor(np.array(globalfeat), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    return data_batch, global_tensor, labels_tensor



import torch
import torch.nn as nn
import numpy as np

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




class GraphCCS_MD_Classifier_concat(nn.Module):
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



        self.graph_project = nn.Linear(self.graph_embed_dim, 208)

        self.graph_project = nn.Linear(self.graph_embed_dim, 208)
        self.global_project = nn.Linear(208, 208)

        self.fuse_norm = nn.LayerNorm(416)


        self.out = nn.Sequential(
            nn.Linear(416, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # 输出 1 个 logit
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

        graph_emb = self.graph_project(graph_vec)

        if self.graph_embed_dim >= 208:
            residual = graph_vec[:, :208]
        else:
            residual = F.linear(graph_vec, weight=self.graph_project.weight[:208], bias=None)

        graph_emb = graph_emb + residual

        global_emb = self.global_project(enhanced_global)

        # 硬 concat 融合：208 + 208 = 416
        x = torch.cat([graph_emb, global_emb], dim=1)

        x = self.fuse_norm(x)
        logits = self.out(x)

        if return_weights:
            return logits, feature_weights, weighted_global
        return logits




class GraphCCS_MD_Classifier_GlobalOnly(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.global_feat_size = config["global_feat_size"]

        print(f"🔗 全局输入: {self.global_feat_size}维")
        print("🔗 消融版本：仅使用全局描述符分支 + MLP 分类")

        self.global_enhancer = ExplainableGlobalEnhancer(
            input_dim=self.global_feat_size,
            hidden_dim=208,
            dropout=0.1
        )

        self.global_norm = nn.LayerNorm(208)
        self.fuse_norm = nn.LayerNorm(208)

        self.out = nn.Sequential(
            nn.Linear(208, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # 二分类输出 1 个 logit
        )

    def forward(self, data_batch, global_feat, return_weights=False):

        if isinstance(global_feat, np.ndarray):
            global_feat = torch.tensor(global_feat, dtype=torch.float32)


        device = next(self.parameters()).device
        global_feat = global_feat.float().to(device)

        enhanced_global, feature_weights, weighted_global = self.global_enhancer(global_feat)

        x = self.global_norm(enhanced_global)
        x = self.fuse_norm(x)

        logits = self.out(x)

        if return_weights:
            return logits, feature_weights, weighted_global

        return logits


class GraphCCS_MD_Classifier_GraphOnly(nn.Module):
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

        print(f"🔗 GNN输出: {config['gnn_dim']}×{config['gnn_heads']}={self.graph_embed_dim}维")
        print("🔗 消融版本：仅保留图分支")

        self.graph_norm = nn.LayerNorm(self.graph_embed_dim)
        self.graph_project = nn.Linear(self.graph_embed_dim, 208)
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

    def forward(self, data_batch, return_weights=False):
        graph_vec = self.backbone.forward_feature(data_batch)
        graph_vec = self.graph_norm(graph_vec)

        graph_emb = self.graph_project(graph_vec)

        if self.graph_embed_dim >= 208:
            residual = graph_vec[:, :208]
        else:
            residual = F.linear(graph_vec, weight=self.graph_project.weight[:208], bias=None)

        graph_emb = graph_emb + residual
        x = self.fuse_norm(graph_emb)
        logits = self.out(x)

        if return_weights:
            return logits, None, None, None, None
        return logits







def compute_metrics_from_logits(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits).view(-1)
    preds = (probs >= threshold).long().cpu().numpy()
    trues = targets.view(-1).long().cpu().numpy()

    acc = accuracy_score(trues, preds)
    pre = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)
    return acc, pre, rec, f1


import gc


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0



    all_labels = []
    all_preds = []

    pbar = tqdm(loader, desc="Train")
    for data_batch, global_feat, y in pbar:

        y = y.to(device)
        global_feat = global_feat.to(device)


        smiles, fingerprints, adjacencies, molecular_sizes = data_batch
        fingerprints = [fp.to(device) for fp in fingerprints]  # 关键修复！
        adjacencies = [adj.to(device) for adj in adjacencies]  # 关键修复！
        data_batch = (smiles, fingerprints, adjacencies, molecular_sizes)



        optimizer.zero_grad()
        logits = model(data_batch, global_feat)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = (torch.sigmoid(logits) >= 0.5).long()


        all_labels.extend(y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        total_correct += (preds == y.long()).sum().item()


        train_acc = total_correct / total_samples
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{train_acc:.4f}'
        })


    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)


    avg_loss = total_loss / total_samples
    torch.cuda.empty_cache()
    gc.collect()

    return avg_loss, acc, precision, recall, f1


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    if loader is None or len(loader.dataset) == 0:
        return float("nan"), None, None, None, None

    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    all_labels = []
    all_preds = []

    pbar = tqdm(loader, desc="Eval ")
    for data_batch, global_feat, y in pbar:

        y = y.to(device)
        global_feat = global_feat.to(device)

        smiles, fingerprints, adjacencies, molecular_sizes = data_batch
        fingerprints = [fp.to(device) for fp in fingerprints]
        adjacencies = [adj.to(device) for adj in adjacencies]
        data_batch = (smiles, fingerprints, adjacencies, molecular_sizes)

        logits = model(data_batch, global_feat)
        loss = loss_fn(logits, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = (torch.sigmoid(logits) >= 0.5).long()

        all_labels.extend(y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        total_correct += (preds == y.long()).sum().item()


        pbar.set_postfix({'Acc': f'{total_correct / total_samples:.4f}'})


    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    avg_loss = total_loss / total_samples
    torch.cuda.empty_cache()
    gc.collect()

    return avg_loss, acc, precision, recall, f1

def evaluate_with_gate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    all_labels = []
    all_preds = []
    all_graph_alphas = []
    all_global_alphas = []
    all_feature_weights = []
    all_weighted_global = []

    pbar = tqdm(loader, desc="Eval with Gate")
    with torch.no_grad():
        for data_batch, global_feat, y in pbar:
            y = y.to(device)
            global_feat = global_feat.to(device)

            smiles, fingerprints, adjacencies, molecular_sizes = data_batch
            fingerprints = [fp.to(device) for fp in fingerprints]
            adjacencies = [adj.to(device) for adj in adjacencies]
            data_batch = (smiles, fingerprints, adjacencies, molecular_sizes)

            logits, graph_alpha, global_alpha, feature_weights, weighted_global = model(
                data_batch, global_feat, return_weights=True
            )

            loss = loss_fn(logits.view(-1), y.float().view(-1))

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = (torch.sigmoid(logits.view(-1)) >= 0.5).long()

            all_labels.extend(y.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())

            total_correct += (preds == y.long().view(-1)).sum().item()

            all_graph_alphas.append(graph_alpha.detach().cpu())
            all_global_alphas.append(global_alpha.detach().cpu())
            all_feature_weights.append(feature_weights.detach().cpu())
            all_weighted_global.append(weighted_global.detach().cpu())

            pbar.set_postfix({'Acc': f'{total_correct / total_samples:.4f}'})

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    avg_loss = total_loss / total_samples

    all_graph_alphas = torch.cat(all_graph_alphas, dim=0)
    all_global_alphas = torch.cat(all_global_alphas, dim=0)
    all_feature_weights = torch.cat(all_feature_weights, dim=0)
    all_weighted_global = torch.cat(all_weighted_global, dim=0)

    print("图分支平均权重:", all_graph_alphas.mean().item())
    print("全局分支平均权重:", all_global_alphas.mean().item())
    print("图分支主导样本比例 (graph_alpha > global_alpha):",
          (all_graph_alphas > all_global_alphas).float().mean().item())
    print("全局分支主导样本比例 (global_alpha > graph_alpha):",
          (all_global_alphas > all_graph_alphas).float().mean().item())
    print("权重和校验 (应≈1.0):", (all_graph_alphas + all_global_alphas).mean().item())

    return avg_loss, acc, precision, recall, f1, all_graph_alphas, all_global_alphas, all_feature_weights, all_weighted_global

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_topk_attention(attention_weights, desc_names, save_dir, top_k=20, prefix="test"):
    mean_attention = attention_weights.mean(dim=0).detach().cpu().numpy()
    top_idx = np.argsort(mean_attention)[-top_k:][::-1]

    top_values = mean_attention[top_idx]
    top_names = [desc_names[i] for i in top_idx]

    df = pd.DataFrame({
        "descriptor": top_names,
        "mean_attention": top_values
    })
    csv_path = os.path.join(save_dir, f"{prefix}_top{top_k}_attention.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 6))
    plt.bar(top_names, top_values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Attention Weight")
    plt.title(f"Top {top_k} RDKit Descriptor Attention Weights")
    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"{prefix}_top{top_k}_attention.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.barh(top_names[::-1], top_values[::-1])
    plt.xlabel("Mean Attention Weight")
    plt.title(f"Top {top_k} RDKit Descriptor Attention Weights")
    plt.tight_layout()
    fig_path_h = os.path.join(save_dir, f"{prefix}_top{top_k}_attention_horizontal.png")
    plt.savefig(fig_path_h, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ attention 可视化已保存: {csv_path}")
    print(f"✅ 图像已保存: {fig_path}")
    print(f"✅ 横向图已保存: {fig_path_h}")



from sklearn.metrics import precision_score, recall_score, f1_score
import copy


def main():
    import os
    import gc
    import random
    import numpy as np
    import torch
    import torch.backends.cudnn as cudnn

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

    torch.cuda.empty_cache()
    gc.collect()


    CSV_PATH = str(ROOT / "data" / "train.csv")
    LABEL_COL = "Botanical"

    cache_dir = CSV_PATH.rsplit('.', 1)[0] + f"_{LABEL_COL}_dgl_cache"
    cache_file = os.path.join(cache_dir, "dgl_graphs.pt")


    print("🔍 先检查 NPBS_DGL_Dataset 是否正常...")
    tmp_dataset = NPBS_DGL_Dataset(
        CSV_PATH,
        label_col=LABEL_COL,
        encoding="gbk",
        use_cache=True
    )
    print("Dataset OK, size =", len(tmp_dataset))
    _ = tmp_dataset[0]  # 触发一次 __getitem__ 和 assert
    print("✅ __getitem__ 检查通过，开始正式训练")


    from preprocess import fingerprint_dict
    print(f"fingerprint_dict: {len(fingerprint_dict)} 个指纹")
    print(f"fingerprint_dict 最大 ID: {max(fingerprint_dict.values()) if len(fingerprint_dict) > 0 else 0}")




    full_dataset = tmp_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)


    print(f"[{LABEL_COL}] 总有效样本数: {len(full_dataset)}")
    all_y = [int(y) for y in full_dataset.y]
    counter = Counter(all_y)
    print(f"[{LABEL_COL}] 标签分布: {counter}")


    indices = np.arange(len(full_dataset))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.1, stratify=all_y, random_state=42)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.1111,
        stratify=np.array(all_y)[train_val_idx],
        random_state=42
    )

    def subset(dataset, idx_list):
        ds = [dataset[idx] for idx in idx_list]
        return ds

    train_set = subset(full_dataset, train_idx)
    val_set = subset(full_dataset, val_idx)
    test_set = subset(full_dataset, test_idx)

    print(f"[{LABEL_COL}] train/val/test: {len(train_set)} / {len(val_set)} / {len(test_set)}")


    train_loader = DataLoader(
        train_set,
        batch_size=128,  # 🔥 减小 batch_size，提升稳定性
        shuffle=True,
        collate_fn=improved_kaagcn_collate_fn,
        num_workers=0,  # 🔥 关键：避免大 list 多进程序列化
        pin_memory=True,  # 🔥 CPU→GPU 更快
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_set,
        batch_size=128,
        shuffle=False,
        collate_fn=improved_kaagcn_collate_fn,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )

    test_loader = DataLoader(  # 🔥 test_loader 也要优化！
        test_set,
        batch_size=128,
        shuffle=False,
        collate_fn=improved_kaagcn_collate_fn,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )

    print(f"✅ DataLoader 配置完成：")
    print(f"   train: {len(train_loader)} batches × 128")
    print(f"   val:   {len(val_loader)} batches × 128")
    print(f"   test:  {len(test_loader)} batches × 128")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)


    sample = train_set[0]
    _, _, _, _, sample_global, sample_label = sample
    global_feat_size = sample_global.shape[0]



    config = {
        "global_feat_size": global_feat_size,  # 217


        "gnn_N": 2000,
        "gnn_dim": 48,
        "gnn_heads": 4,
        "gnn_hidden": 5,
        "gnn_output": 2,
        "gnn_grid": 5,
        "gnn_dropout": 0.2,
    }



    model = GraphCCS_MD_Classifier_Explainable(config).to(device)

    # model = GraphCCS_MD_Classifier_concat(config).to(device)

    # model = GraphCCS_MD_Classifier_GraphOnly(config).to(device)

    # model = GraphCCS_MD_Classifier_GlobalOnly(config).to(device)



    print(f"✅ 模型创建完成！GNN: dim={config['gnn_dim']}×heads={config['gnn_heads']}")

    print(f"✅ 最终模型配置:")

    print(f"  Global feat: {config['global_feat_size']}维")
    print(f"  拼接维度: {200 + config['global_feat_size']}维")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")

    train_labels = [int(s[5]) for s in train_set]
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos
    pos_weight_value = num_neg / max(num_pos, 1) if num_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    print(f"pos_weight = {pos_weight.item():.4f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        # mode='min',
        mode='min',
        factor=0.5,
        patience=8,
        threshold=1e-4,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-6
    )


    patience = 12
    best_val_loss = float('inf')
    patience_counter = 0
    num_epochs = 200
    best_val_f1 = 0.0
    early_stop_epoch = None
    best_val_acc = 0.0


    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        "lr": [],
    }

    print("🚀 开始训练 (早停耐心度=20, 动态学习率)")



    save_dir = str(ROOT / "results" / "yesglo")
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    fig_dir = os.path.join(save_dir, "figs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    best_acc_model_path = os.path.join(ckpt_dir, f"best_by_val_acc_{LABEL_COL}.pt")
    best_loss_model_path = os.path.join(ckpt_dir, f"best_by_val_loss_{LABEL_COL}.pt")
    last_model_path = os.path.join(ckpt_dir, f"last_epoch_{LABEL_COL}.pt")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_pre, train_rec, train_f1 = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_loss, val_acc, val_pre, val_rec, val_f1 = evaluate(
            model, val_loader, loss_fn, device
        )

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }, best_acc_model_path)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }, best_loss_model_path)
        else:
            patience_counter += 1
            print(f"⏳ val_loss未改善: {val_loss:.4f} (耐心度: {patience_counter}/{patience})")

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
            f"LR: {current_lr:.2e} "
        )

        torch.save({
            "epoch": epoch,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
        }, last_model_path)

        if patience_counter >= patience:
            early_stop_epoch = epoch
            print(f"\n🛑 **早停触发！** 在第 {early_stop_epoch} 轮停止训练")
            print(f"   最佳val_loss: {best_val_loss:.4f}")
            break




    print(f"\n🎉 训练完成！最佳验证F1: {best_val_f1:.4f}")
    if early_stop_epoch is not None:
        print(f"🛑 早停轮数: {early_stop_epoch}")
    else:
        print("✅ 完整训练{num_epochs}轮")

    print(f"最佳验证 F1 = {best_val_f1:.4f}")

    print("===== 使用最佳验证ACC模型进行测试 =====")
    if os.path.exists(best_acc_model_path):
        best_acc_ckpt = torch.load(best_acc_model_path, map_location=device)
        model.load_state_dict(best_acc_ckpt["model_state_dict"])
        print("已加载验证 ACC 最优模型参数进行测试评估。")
        test_loss, test_acc, test_pre, test_rec, test_f1 = evaluate(
            model, test_loader, loss_fn, device
        )
        print(
            f"[Best Val ACC][Test] Loss: {test_loss:.4f} | "
            f"Acc: {test_acc:.4f} | Precision: {test_pre:.4f} | "
            f"Recall: {test_rec:.4f} | F1: {test_f1:.4f}"
        )

    print("===== 使用最佳验证Loss模型进行测试 =====")
    if os.path.exists(best_loss_model_path):
        best_loss_ckpt = torch.load(best_loss_model_path, map_location=device)
        model.load_state_dict(best_loss_ckpt["model_state_dict"])
        print("已加载验证 Loss 最优模型参数进行测试评估。")
        test_loss, test_acc, test_pre, test_rec, test_f1 = evaluate(
            model, test_loader, loss_fn, device
        )
        print(
            f"[Best Val Loss][Test] Loss: {test_loss:.4f} | "
            f"Acc: {test_acc:.4f} | Precision: {test_pre:.4f} | "
            f"Recall: {test_rec:.4f} | F1: {test_f1:.4f}"
        )

    print("===== 使用最后一轮模型进行测试 =====")
    if os.path.exists(last_model_path):
        last_ckpt = torch.load(last_model_path, map_location=device)
        model.load_state_dict(last_ckpt["model_state_dict"])
        print("已加载最后一轮模型参数进行测试评估。")
        test_loss, test_acc, test_pre, test_rec, test_f1 = evaluate(
            model, test_loader, loss_fn, device
        )
        print(
            f"[Last Epoch][Test] Loss: {test_loss:.4f} | "
            f"Acc: {test_acc:.4f} | Precision: {test_pre:.4f} | "
            f"Recall: {test_rec:.4f} | F1: {test_f1:.4f}"
        )




if __name__ == "__main__":
    main()



