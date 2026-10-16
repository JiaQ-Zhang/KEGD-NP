"""Botanical SHAP: original data/attribution methods, one beeswarm output.
Default analysis: 300 training backgrounds, 2000 balanced test explanations,
Kernel SHAP nsamples=1000, l1_reg=0; each molecule's graph is held fixed.
"""
import argparse
import contextlib
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from sklearn.model_selection import train_test_split


with contextlib.redirect_stdout(io.StringIO()):
    from train import (NPBS_DGL_Dataset, DESC_NAMES,
                       GraphCCS_MD_Classifier_Explainable)
from model.shap_helpers import (set_seed, encode_graph_once, predict_with_cached_graph,
                                validate_cached_graph_predictions, sample_has_edges,
                                load_checkpoint_state_dict, extract_label)

ROOT = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def split_dataset(full_dataset):
    all_y = np.asarray([int(y) for y in full_dataset.y], dtype=int)
    indices = np.arange(len(full_dataset))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.1, stratify=all_y, random_state=42)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.1111, stratify=all_y[train_val_idx], random_state=42)
    train_set = [full_dataset[i] for i in train_idx]
    test_set = [full_dataset[i] for i in test_idx]
    return train_set, test_set


def select_samples(train_set, test_set, background_num, explain_num):
    if background_num > len(train_set):
        raise ValueError(f'background={background_num} exceeds training size={len(train_set)}.')
    if explain_num > len(test_set):
        raise ValueError(f'explain={explain_num} exceeds test size={len(test_set)}.')
    rng = np.random.default_rng(42)
    background_indices = rng.choice(len(train_set), size=background_num, replace=False)
    background = [train_set[int(i)] for i in background_indices]
    test_labels = np.asarray([extract_label(sample) for sample in test_set], dtype=int)
    has_edge_mask = np.asarray([sample_has_edges(sample) for sample in test_set], dtype=bool)
    negative_pool = np.where((test_labels == 0) & has_edge_mask)[0]
    positive_pool = np.where((test_labels == 1) & has_edge_mask)[0]
    per_class_num = explain_num // 2
    if len(negative_pool) < per_class_num or len(positive_pool) < per_class_num:
        raise ValueError(f'Need {per_class_num} eligible test molecules per class; '
                         f'negative={len(negative_pool)}, positive={len(positive_pool)}.')
    selected_negative = rng.choice(negative_pool, size=per_class_num, replace=False)
    selected_positive = rng.choice(positive_pool, size=per_class_num, replace=False)
    explain_indices = np.concatenate([selected_negative, selected_positive]).astype(np.int64)
    rng.shuffle(explain_indices)
    explained = [test_set[int(i)] for i in explain_indices]
    selected_labels = np.asarray([extract_label(sample) for sample in explained], dtype=int)
    if np.sum(selected_labels == 0) != per_class_num or np.sum(selected_labels == 1) != per_class_num:
        raise RuntimeError('Explanation samples must be strictly balanced 1:1.')
    return background, explained


def compute_shap(model, background, explain_samples, nsamples=1000):
    DEVICE = torch.device('cpu')
    PREDICT_BATCH_SIZE = 128
    SHAP_NSAMPLES = nsamples
    background_X = np.stack([np.asarray(sample[4], dtype=np.float32) for sample in background], axis=0)
    explain_X = np.stack([np.asarray(sample[4], dtype=np.float32) for sample in explain_samples], axis=0)
    global_feat_size = explain_X.shape[1]
    cached = [encode_graph_once(model, sample, DEVICE) for sample in explain_samples]
    cached_graph_vecs = [pair[0] for pair in cached]
    cached_graph_embs = [pair[1] for pair in cached]
    validate_cached_graph_predictions(model, explain_samples, cached_graph_vecs,
                                     cached_graph_embs, DEVICE, check_num=3, tolerance=1e-6)
    all_shap_values = []
    expected_values = []
    for idx, sample in enumerate(explain_samples):
        sample_X = np.asarray(
            sample[4],
            dtype=np.float32,
        ).reshape(1, -1)

        fixed_graph_vec = cached_graph_vecs[idx]
        fixed_graph_emb = cached_graph_embs[idx]

        def single_predict_fn(
            x_np,
            cached_vec=fixed_graph_vec,
            cached_emb=fixed_graph_emb,
        ):
            x_np = np.asarray(x_np, dtype=np.float32)

            if x_np.ndim == 1:
                x_np = x_np.reshape(1, -1)

            return predict_with_cached_graph(
                model=model,
                cached_graph_vec=cached_vec,
                cached_graph_emb=cached_emb,
                global_feats_np=x_np,
                device=DEVICE,
                batch_size=PREDICT_BATCH_SIZE,
            )

        explainer = shap.KernelExplainer(
            single_predict_fn,
            background_X,
        )

        sample_shap_values = explainer.shap_values(
            sample_X,
            nsamples=SHAP_NSAMPLES,
            l1_reg=0,
        )

        sample_shap_values = np.asarray(
            sample_shap_values
        ).reshape(-1)

        if sample_shap_values.shape[0] != global_feat_size:
            raise ValueError(
                f'第 {idx} 个样本的 SHAP 维数异常：'
                f'{sample_shap_values.shape[0]} != {global_feat_size}'
            )

        expected_value = np.asarray(
            explainer.expected_value
        ).reshape(-1)[0]

        all_shap_values.append(sample_shap_values)
        expected_values.append(float(expected_value))

        print(f'已完成 {idx + 1}/{len(explain_samples)}')
    shap_values = np.stack(all_shap_values, axis=0)
    expected_values = np.asarray(expected_values, dtype=np.float64)
    return shap_values, explain_X, expected_values


def save_beeswarm(shap_values, explain_X, descriptor_names, output):

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    if len(mean_abs_shap) != len(descriptor_names):
        raise ValueError('Descriptor and SHAP dimensions do not match.')

    plt.figure()
    shap.summary_plot(
        shap_values,
        explain_X,
        feature_names=descriptor_names,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='data/train.csv', help='Complete labeled training CSV for this checkpoint.')
    parser.add_argument('--weights', default='weight/best_Botanical.pt')
    parser.add_argument('--background', type=int, default=300)
    parser.add_argument('--explain', type=int, default=2000)
    parser.add_argument('--nsamples', type=int, default=1000)
    parser.add_argument('--output', default='results/shap/shap_beeswarm_top20.png')
    args = parser.parse_args()
    if args.background <= 0 or args.explain <= 0 or args.explain % 2 or args.nsamples <= 0:
        parser.error('background/nsamples must be positive; explain must be a positive even number.')
    csv_path, weights = project_path(args.csv), project_path(args.weights)
    output = project_path(args.output)
    if not csv_path.is_file() or not weights.is_file():
        parser.error('CSV or checkpoint file does not exist.')
    if output.suffix.lower() != '.png':
        parser.error('output must be a .png path.')
    set_seed(42)

    full_dataset = NPBS_DGL_Dataset(str(csv_path), label_col='Botanical',
                                    encoding='gbk', use_cache=True)
    train_set, test_set = split_dataset(full_dataset)
    if args.background > len(train_set) or args.explain > len(test_set):
        raise ValueError('Requested SHAP sample counts exceed the train/test split. No automatic fallback.')
    sample_global = np.asarray(train_set[0][4], dtype=np.float32).reshape(-1)
    global_feat_size = int(sample_global.shape[0])
    descriptor_names = [str(name) for name in DESC_NAMES]
    if len(descriptor_names) != global_feat_size:
        raise ValueError('DESC_NAMES does not match the dataset feature size.')
    config = dict(global_feat_size=global_feat_size, gnn_N=2000, gnn_dim=48, gnn_heads=4,
                  gnn_hidden=5, gnn_output=2, gnn_grid=5, gnn_dropout=0.2)
    with contextlib.redirect_stdout(io.StringIO()):
        model = GraphCCS_MD_Classifier_Explainable(config).to(torch.device('cpu'))
    model.load_state_dict(load_checkpoint_state_dict(str(weights), torch.device('cpu')))
    model.eval()
    background, explained = select_samples(train_set, test_set, args.background, args.explain)
    print(f'Botanical | background={len(background)} | explained={len(explained)} (1:1) | CPU')
    values, features, _ = compute_shap(model, background, explained, args.nsamples)
    save_beeswarm(values, features, descriptor_names, output)
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
