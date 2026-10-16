import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train import improved_kaagcn_collate_fn

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_one_sample(sample):

    loader = DataLoader(
        [sample],
        batch_size=1,
        shuffle=False,
        collate_fn=improved_kaagcn_collate_fn,
        num_workers=0,
    )
    return next(iter(loader))

def encode_graph_once(model, sample, device):

    data_batch, _, _ = collate_one_sample(sample)

    model.eval()
    with torch.inference_mode():
        graph_vec = model.backbone.forward_feature(data_batch)
        graph_vec = model.graph_norm(graph_vec)

        graph_emb = model.graph_project(graph_vec)

        if model.graph_embed_dim >= 208:
            residual = graph_vec[:, :208]
        else:
            residual = F.linear(
                graph_vec,
                weight=model.graph_project.weight[:208],
                bias=None,
            )

        graph_emb = graph_emb + residual

    return (
        graph_vec.detach().to(device).squeeze(0),
        graph_emb.detach().to(device).squeeze(0),
    )

def predict_with_cached_graph(
    model,
    cached_graph_vec,
    cached_graph_emb,
    global_feats_np,
    device,
    batch_size=128,
):

    global_feats_np = np.asarray(global_feats_np, dtype=np.float32)

    if global_feats_np.ndim == 1:
        global_feats_np = global_feats_np.reshape(1, -1)

    all_probs = []

    model.eval()
    with torch.inference_mode():
        for start_index in range(0, len(global_feats_np), batch_size):
            end_index = min(
                start_index + batch_size,
                len(global_feats_np),
            )

            global_feat = torch.from_numpy(
                global_feats_np[start_index:end_index]
            ).to(
                device=device,
                dtype=torch.float32,
            )

            enhanced_global, _, _ = model.global_enhancer(global_feat)
            enhanced_global = model.global_norm(enhanced_global)

            current_batch_size = global_feat.shape[0]

            graph_vec_batch = cached_graph_vec.unsqueeze(0).expand(
                current_batch_size,
                -1,
            )
            graph_emb_batch = cached_graph_emb.unsqueeze(0).expand(
                current_batch_size,
                -1,
            )

            gate_input = torch.cat(
                [graph_vec_batch, enhanced_global],
                dim=1,
            )
            alpha = torch.sigmoid(model.gate(gate_input))

            global_emb = model.global_project(enhanced_global)

            fused = (
                alpha * graph_emb_batch
                + (1.0 - alpha) * global_emb
            )
            fused = model.fuse_norm(fused)
            logits = model.out(fused)

            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            all_probs.append(probs)

    if not all_probs:
        return np.empty((0,), dtype=np.float32)

    return np.concatenate(all_probs, axis=0)

def predict_original_model_one(model, sample, device):

    data_batch, global_feat, _ = collate_one_sample(sample)
    global_feat = global_feat.to(device)

    model.eval()
    with torch.inference_mode():
        logits = model(data_batch, global_feat)
        probability = torch.sigmoid(logits).cpu().numpy().reshape(-1)[0]

    return float(probability)

def validate_cached_graph_predictions(
    model,
    explain_samples,
    cached_graph_vecs,
    cached_graph_embs,
    device,
    check_num=3,
    tolerance=1e-6,
):

    check_num = min(check_num, len(explain_samples))

    print(f'开始验证缓存图分支，检查前 {check_num} 个样本...')

    for index in range(check_num):
        sample = explain_samples[index]
        sample_x = np.asarray(
            sample[4],
            dtype=np.float32,
        ).reshape(1, -1)

        original_probability = predict_original_model_one(
            model=model,
            sample=sample,
            device=device,
        )

        cached_probability = float(
            predict_with_cached_graph(
                model=model,
                cached_graph_vec=cached_graph_vecs[index],
                cached_graph_emb=cached_graph_embs[index],
                global_feats_np=sample_x,
                device=device,
                batch_size=1,
            )[0]
        )

        difference = abs(original_probability - cached_probability)

        print(
            f'缓存验证样本 {index + 1}: '
            f'原模型={original_probability:.10f}, '
            f'缓存模型={cached_probability:.10f}, '
            f'差值={difference:.3e}'
        )

        if difference > tolerance:
            raise RuntimeError(
                '缓存图分支后的预测与原始完整模型不一致。'
                f'样本={index}, 差值={difference:.6e}, '
                f'容许误差={tolerance:.6e}。'
                '请检查clean.py中的模型forward是否已经改变。'
            )

    print('✅ 缓存图分支与原始完整模型预测一致。')

def sample_has_edges(sample):

    adjacencies = sample[2]

    def has_edge_in_one_adj(adj):
        if adj is None:
            return False

        if torch.is_tensor(adj):
            if adj.numel() == 0:
                return False

            if adj.ndim == 2 and adj.shape[0] == 2:
                return adj.shape[1] > 0

            return torch.count_nonzero(adj).item() > 0

        if hasattr(adj, "nnz"):
            return adj.nnz > 0

        arr = np.asarray(adj)

        if arr.size == 0:
            return False


        if arr.ndim == 2 and arr.shape[0] == 2:
            return arr.shape[1] > 0


        return np.count_nonzero(arr) > 0

    if isinstance(adjacencies, (list, tuple)):
        return any(has_edge_in_one_adj(adj) for adj in adjacencies)

    return has_edge_in_one_adj(adjacencies)

def load_checkpoint_state_dict(checkpoint_path, device):

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    if isinstance(checkpoint, dict):
        return checkpoint

    raise TypeError(
        "无法识别检查点格式。检查点应为 state_dict，或包含 model_state_dict 键。"
    )

def extract_label(sample):

    label = sample[-1]

    if torch.is_tensor(label):
        return int(label.detach().cpu().reshape(-1)[0].item())

    label = np.asarray(label).reshape(-1)
    return int(label[0])
