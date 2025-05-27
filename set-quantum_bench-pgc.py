
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum-Inspired Benchmark Datasets
==================================

Grover's Search  : unique marked item                 (binary class)
Deutsch–Jozsa    : constant-vs-balanced Boolean funcs (binary class)
Simon's Problem  : hidden XOR mask  s                 (multi-class)

Each generator returns NumPy arrays ready for a
PGC / classical-ML pipeline.

Author : (c) 2025  —  Cloudcell / Polymorphic Geometry Project
License: MIT
"""
import numpy as np
from itertools import product
from typing import Tuple, Optional
import torch
import pickle
import argparse

# ------------------------------------------------------------
# 1)  Grover's Search dataset
# ------------------------------------------------------------
def grover_dataset(n_bits: int,
                   pos_ratio: float = 0.5,
                   rng: Optional[np.random.Generator] = None
                   ) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Parameters
    ----------
    n_bits     : size of the binary search space (2^n items)
    pos_ratio  : fraction of positive (marked) samples to keep
    rng        : optional NumPy Generator for reproducibility

    Returns
    -------
    X   : (m, n_bits) binary array
    y   : (m,) binary labels (1 = marked)
    key : hidden string identifying the unique marked state
    """
    rng = rng or np.random.default_rng()
    hidden = rng.integers(0, 2, size=n_bits, dtype=int)
    key = ''.join(map(str, hidden))

    full_space = np.array(list(product([0, 1], repeat=n_bits)), dtype=int)
    labels = (full_space == hidden).all(axis=1).astype(int)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    rng.shuffle(neg_idx)

    k_neg = int((1 - pos_ratio) / pos_ratio * len(pos_idx))
    selected = np.concatenate([pos_idx, neg_idx[:k_neg]])
    rng.shuffle(selected)

    return full_space[selected], labels[selected], key


# ------------------------------------------------------------
# 2)  Deutsch–Jozsa dataset
# ------------------------------------------------------------
def deutsch_jozsa_dataset(n_bits: int,
                          n_funcs: int = 1024,
                          rng: Optional[np.random.Generator] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate truth-tables of random Boolean functions that are either
    constant or balanced; label = 0 (constant), 1 (balanced).
    """
    rng = rng or np.random.default_rng()
    tbl_len = 2 ** n_bits
    X = np.empty((n_funcs, tbl_len), dtype=int)
    y = np.empty(n_funcs, dtype=int)

    for i in range(n_funcs):
        is_balanced = rng.integers(0, 2)          # 0 = constant
        if is_balanced:
            tbl = np.array([0]*(tbl_len//2) + [1]*(tbl_len//2), dtype=int)
            rng.shuffle(tbl)
        else:
            tbl = np.full(tbl_len, rng.integers(0, 2), dtype=int)
        X[i], y[i] = tbl, is_balanced
    return X, y


# ------------------------------------------------------------
# 3)  Simon's Problem dataset
# ------------------------------------------------------------
def simon_dataset(n_bits: int,
                  n_pairs: int = 4096,
                  rng: Optional[np.random.Generator] = None
                  ) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Produce input–output pairs (x, f(x)) where
        f(x) = A·x  (mod 2)  and  f(x) = f(x ⊕ s)
    Hidden mask s is the multi-class label.
    """
    rng = rng or np.random.default_rng()

    # choose non-zero mask  s
    while True:
        s = rng.integers(0, 2, size=n_bits, dtype=int)
        if s.any():
            break
    s_int  = int(''.join(map(str, s)), 2)
    s_str  = ''.join(map(str, s))

    # random binary matrix A  (full row-rank not required here)
    A = rng.integers(0, 2, size=(n_bits, n_bits), dtype=int)
    f = lambda x: (A @ x) % 2

    X_raw = rng.integers(0, 2, size=(n_pairs, n_bits), dtype=int)
    Y_raw = np.apply_along_axis(f, 1, X_raw)

    X = np.hstack([X_raw, Y_raw]).astype(int)
    y = np.full(n_pairs, s_int, dtype=int)
    return X, y, s_str


# ------------------------------------------------------------
# Demo / sanity-check
# ------------------------------------------------------------
def _demo() -> None:
    rng = np.random.default_rng(42)

    Xg, yg, key = grover_dataset(4, rng=rng)
    print("Grover:", Xg.shape, yg.sum(), "marked =", key)

    Xd, yd = deutsch_jozsa_dataset(4, 8, rng)
    print("Deutsch–Jozsa:", Xd.shape, "balanced =", yd.sum())

    Xs, ys, mask = simon_dataset(4, 32, rng)
    print("Simon:", Xs.shape, "mask =", mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate quantum benchmark datasets in PGC/MNIST-compatible .pkl format."
    )
    parser.add_argument('--dataset', choices=['grover', 'deutsch-jozsa', 'simon'],
                        help='Which dataset to generate.')
    parser.add_argument('--n_bits', type=int, help='Number of bits (input size).')
    parser.add_argument('--output', type=str, help='Output .pkl filename.')
    parser.add_argument('--pos_ratio', type=float, default=0.5,
                        help='Grover: Fraction of positive samples to keep (default: 0.5).')
    parser.add_argument('--n_funcs', type=int, default=1024,
                        help="Deutsch–Jozsa: Number of functions (default: 1024).")
    parser.add_argument('--n_pairs', type=int, default=4096,
                        help="Simon: Number of input-output pairs (default: 4096).")
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--all', action='store_true',
                        help='Generate all datasets with default values.')
    args = parser.parse_args()

    def save_dataset(X, y, out_file):
        features = torch.from_numpy(X)
        labels = torch.from_numpy(y)
        with open(out_file, 'wb') as f:
            pickle.dump({'features': features, 'labels': labels}, f)
        print(f"Saved dataset to {out_file} (features: {features.shape}, labels: {labels.shape})")

    if args.all:
        print("Generating all datasets with default values...")
        rng = np.random.default_rng(args.seed)
        # Grover
        Xg, yg, key = grover_dataset(4, 0.5, rng)
        print(f"Grover: {Xg.shape[0]} samples, {Xg.shape[1]} bits. Marked key: {key}")
        save_dataset(Xg, yg, 'grover-pgc.pkl')
        # Deutsch–Jozsa
        Xd, yd = deutsch_jozsa_dataset(4, 1024, rng)
        print(f"Deutsch–Jozsa: {Xd.shape[0]} samples, {Xd.shape[1]} bits.")
        save_dataset(Xd, yd, 'deutsch-jozsa-pgc.pkl')
        # Simon
        Xs, ys, mask = simon_dataset(4, 4096, rng)
        print(f"Simon: {Xs.shape[0]} samples, {Xs.shape[1]} bits. Hidden mask: {mask}")
        save_dataset(Xs, ys, 'simon-pgc.pkl')
        exit(0)

    if not (args.dataset and args.n_bits and args.output):
        parser.error('You must specify --dataset, --n_bits, and --output unless using --all.')

    rng = np.random.default_rng(args.seed)

    if args.dataset == 'grover':
        X, y, key = grover_dataset(args.n_bits, args.pos_ratio, rng)
        print(f"Grover dataset: {X.shape[0]} samples, {X.shape[1]} bits. Marked key: {key}")
    elif args.dataset == 'deutsch-jozsa':
        X, y = deutsch_jozsa_dataset(args.n_bits, args.n_funcs, rng)
        print(f"Deutsch–Jozsa dataset: {X.shape[0]} samples, {X.shape[1]} bits.")
    elif args.dataset == 'simon':
        X, y, mask = simon_dataset(args.n_bits, args.n_pairs, rng)
        print(f"Simon dataset: {X.shape[0]} samples, {X.shape[1]} bits. Hidden mask: {mask}")
    else:
        raise ValueError("Unknown dataset type")

    # Convert to torch tensors
    features = torch.from_numpy(X)
    labels = torch.from_numpy(y)

    # Save as pickle file
    with open(args.output, 'wb') as f:
        pickle.dump({'features': features, 'labels': labels}, f)
    print(f"Saved dataset to {args.output} (features: {features.shape}, labels: {labels.shape})")

    parser = argparse.ArgumentParser(
        description="Generate quantum benchmark datasets in PGC/MNIST-compatible .pkl format."
    )
    parser.add_argument('--dataset', choices=['grover', 'deutsch-jozsa', 'simon'],
                        help='Which dataset to generate.')
    parser.add_argument('--n_bits', type=int, help='Number of bits (input size).')
    parser.add_argument('--pos_ratio', type=float, default=0.5,
                        help='Grover: Fraction of positive samples to keep (default: 0.5).')
    parser.add_argument('--n_funcs', type=int, default=1024,
                        help="Deutsch–Jozsa: Number of functions (default: 1024).")
    parser.add_argument('--n_pairs', type=int, default=4096,
                        help="Simon: Number of input-output pairs (default: 4096).")
    parser.add_argument('--output', type=str, help='Output .pkl filename.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')

    args = parser.parse_args()

    if not args.dataset:
        _demo()
        sys.exit(0)
    if not args.n_bits or not args.output:
        features = torch.from_numpy(X)
        labels = torch.from_numpy(y)
        with open(out_file, 'wb') as f:
            pickle.dump({'features': features, 'labels': labels}, f)
        print(f"Saved dataset to {out_file} (features: {features.shape}, labels: {labels.shape})")

    if args.dataset == 'grover':
        X, y, key = grover_dataset(args.n_bits, args.pos_ratio, rng)
        save_dataset(X, y, args.output)
        print(f"Grover dataset saved to {args.output}. Marked key: {key}")

    elif args.dataset == 'deutsch-jozsa':
        X, y = deutsch_jozsa_dataset(args.n_bits, args.n_funcs, rng)
        save_dataset(X, y, args.output)
        print(f"Deutsch–Jozsa dataset saved to {args.output}.")

    elif args.dataset == 'simon':
        X, y, mask = simon_dataset(args.n_bits, args.n_pairs, rng)
        save_dataset(X, y, args.output)
        print(f"Simon dataset saved to {args.output}. Hidden mask: {mask}")