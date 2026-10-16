"""Run Botanical inference: python test.py (no training or cache rebuilding)."""
import argparse
import contextlib
import csv
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import normalize

from model.classifier import GraphCCS_MD_Classifier_Explainable


with contextlib.redirect_stdout(io.StringIO()):
    from model.preprocess import create_atoms, create_ijbonddict, extract_fingerprints, radius

ROOT = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_smiles(path):
    """Accept a headerless one-column file, or a CSV with a SMILES column."""
    for encoding in ('utf-8-sig', 'gbk'):
        try:
            with path.open(encoding=encoding, newline='') as stream:
                rows = [row for row in csv.reader(stream) if row and any(x.strip() for x in row)]
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError('CSV encoding must be UTF-8 or GBK.')
    if not rows:
        raise ValueError('The input CSV is empty.')
    header = [cell.strip().lower() for cell in rows[0]]
    if 'smiles' in header:
        column = header.index('smiles')
        rows = rows[1:]
    else:
        if any(len(row) != 1 for row in rows):
            raise ValueError('A multi-column CSV must have a SMILES header.')
        column = 0
    return [row[column].strip() if len(row) > column else '' for row in rows]


def load_dictionaries(directory):
    dictionaries = []
    for name in ('atom', 'bond', 'fingerprint', 'edge'):
        with (directory / f'{name}_dict.pickle').open('rb') as stream:

            dictionaries.append(dict(pickle.load(stream)))
    return dictionaries


def global_features(smiles, descriptor_names):
    """Same descriptor clipping, missing-value handling and max norm as training."""
    mol = Chem.MolFromSmiles(smiles)
    values = []
    for name in descriptor_names:
        function = getattr(Descriptors, name)
        try:
            value = function(mol)
            if value is None or np.isnan(value) or np.isinf(value):
                value = 0.0
            value = float(max(min(value, 1e6), -1e6))
        except Exception:
            value = 0.0
        values.append(value)
    array = np.array(values, dtype=np.float32).reshape(1, -1)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return normalize(array, axis=1, norm='max').squeeze(0)


def prepare_sample(smiles, dictionaries, descriptor_names, embedding_size):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError('invalid_smiles')
    mol = Chem.AddHs(mol)
    atom_dict, bond_dict, fingerprint_dict, edge_dict = dictionaries
    try:
        atoms = create_atoms(mol, atom_dict)
    except KeyError:
        raise ValueError('unknown_atom_in_training_dictionary') from None
    try:
        bonds = create_ijbonddict(mol, bond_dict)
    except KeyError:
        raise ValueError('unknown_bond_in_training_dictionary') from None
    try:
        fingerprints = extract_fingerprints(radius, atoms, bonds, fingerprint_dict, edge_dict)
    except KeyError:
        raise ValueError('unknown_fingerprint_or_edge_in_training_dictionary') from None
    adjacency = np.float32(Chem.GetAdjacencyMatrix(mol))

    size = min(len(fingerprints), adjacency.shape[0])
    if size == 0:
        raise ValueError('empty_fingerprints')
    fingerprints = fingerprints[:size]
    if fingerprints.min() < 0 or fingerprints.max() >= embedding_size:
        raise ValueError('fingerprint_id_out_of_embedding_range')
    features = global_features(smiles, descriptor_names)
    if not np.any(features):
        raise ValueError('empty_global_features')
    return (smiles, torch.LongTensor(fingerprints),
            torch.FloatTensor(adjacency[:size, :size]), size, features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='data/test.csv')
    parser.add_argument('--weights', default='weight/best_Botanical.pt')
    parser.add_argument('--dict-dir', help='Default: dict/ beside the selected checkpoint.')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output', help='Optional CSV with SMILES, probabilities, predictions and status.')
    args = parser.parse_args()
    if args.batch_size <= 0 or not 0 <= args.threshold <= 1:
        parser.error('batch-size must be positive; threshold must be between 0 and 1.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available. Use --device cpu.')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    weights_path = project_path(args.weights)
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
    if checkpoint.get('label', 'Botanical') != 'Botanical':
        raise ValueError('This prediction entry uses Botanical column names; select a Botanical checkpoint.')
    state = checkpoint.get('model_state_dict', checkpoint)
    # Configuration of the supplied best_Botanical.pt; strict loading checks every parameter.
    config = dict(global_feat_size=208, gnn_N=2000, gnn_dim=48, gnn_heads=4,
                  gnn_hidden=5, gnn_output=2, gnn_grid=5, gnn_dropout=0.2)
    config = checkpoint.get('config', config)
    descriptor_names = checkpoint.get('descriptor_names')
    if descriptor_names is None:
        descriptor_names = json.loads((weights_path.parent / 'descriptor_names.json').read_text(encoding='utf-8'))
    if len(descriptor_names) != config['global_feat_size']:
        raise ValueError('Descriptor list does not match the model input dimension.')
    if any(not hasattr(Descriptors, name) for name in descriptor_names):
        raise ValueError('RDKit descriptor mismatch. Install the RDKit version in requirements.txt.')
    with contextlib.redirect_stdout(io.StringIO()):
        model = GraphCCS_MD_Classifier_Explainable(config)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    dictionaries = load_dictionaries(project_path(args.dict_dir) if args.dict_dir else weights_path.parent / 'dict')
    smiles_list = read_smiles(project_path(args.csv))
    results, valid = [], []
    for row, smiles in enumerate(smiles_list, start=1):
        result = dict(row=row, SMILES=smiles, Botanical_probability=None,
                      Botanical_prediction=None, status='ok')
        results.append(result)
        try:
            sample = prepare_sample(smiles, dictionaries, descriptor_names, config['gnn_N'])
        except ValueError as error:
            result['status'] = str(error)
            continue
        valid.append((row - 1, sample))
    with torch.inference_mode():
        for start in range(0, len(valid), args.batch_size):
            batch = valid[start:start + args.batch_size]
            _, samples = zip(*batch)
            smiles, fingerprints, adjacencies, sizes, features = zip(*samples)
            data = (list(smiles), [fp.to(device) for fp in fingerprints],
                    [adj.to(device) for adj in adjacencies], list(sizes))
            globals_tensor = torch.tensor(np.stack(features), dtype=torch.float32, device=device)
            probabilities = torch.sigmoid(model(data, globals_tensor)).flatten().cpu().tolist()
            for (index, _), probability in zip(batch, probabilities):
                if not np.isfinite(probability):
                    raise RuntimeError(f'Non-finite model output at row {index + 1}.')
                results[index]['Botanical_probability'] = probability
                results[index]['Botanical_prediction'] = int(probability >= args.threshold)
    frame = pd.DataFrame(results, columns=['row', 'SMILES', 'Botanical_probability',
                                         'Botanical_prediction', 'status'])
    frame['Botanical_prediction'] = frame['Botanical_prediction'].astype('Int64')
    print(frame.drop(columns='SMILES').to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print(f'Total: {len(results)} | Predicted: {len(valid)} | Failed: {len(results) - len(valid)}')
    if args.output:
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False, encoding='utf-8-sig')
    if not valid:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
