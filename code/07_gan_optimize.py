from __future__ import annotations

import copy
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn


# ============================================================
# 1. Paths / global settings
# ============================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

INPUT_PATH = RESULTS_DIR / "04_normalized_data.xlsx"

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "Cannot find results/04_normalized_data.xlsx. "
        "Run the previous preprocessing/normalization step first."
    )

QUICK_MODE = os.getenv("GAN_QUICK", "0") == "1"

# Paper-oriented original K-means baseline is frozen here.
KMEANS_SEED = 76
N_CLUSTERS = 6

# Three independent training seeds for robust model selection.
TRAIN_SEEDS = [42] if QUICK_MODE else [42, 1234, 2026]

# Fixed generation seeds. These do not affect training.
POOL_GENERATION_SEED = 999
SOFT_SELECTION_SEED = 20260820

# 72 is retained as the paper-scale augmentation amount, but cluster
# counts are NOT forced to 12 each in the natural set.
N_SYNTHETIC = 72

# Soft balancing: 1.0 = fully empirical proportions,
#                 0.0 = fully uniform proportions.
# 0.70 keeps most of the natural real-data mixture while mildly
# enriching minority modes.
SOFT_BALANCE_ALPHA = 0.70

FEATURES = [
    "a",
    "b",
    "c",
    "V",
    "concentration",
    "Layer",
    "valence_electron",
    "IQE",
    "Lifetime",
]

DISCRETE_FEATURES = [
    "Layer",
    "valence_electron",
]

PAPER_ORIGINAL_COUNTS = np.array([37, 59, 3, 53, 114, 9])
PAPER_AFTER_COUNTS = np.array([49, 71, 15, 65, 126, 21])

# Our internal criteria, NOT criteria reported by the paper.
TARGET_MEAN_KS = 0.20
IDEAL_MEAN_KS = 0.15
TARGET_CORR_MAE = 0.15
IDEAL_CORR_MAE = 0.10


# ============================================================
# 2. Reproducibility helpers
# ============================================================

def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_counts(x: np.ndarray) -> str:
    return ",".join(str(int(v)) for v in x)


# ============================================================
# 3. Load and freeze the 276-row normalized real dataset
# ============================================================

df = pd.read_excel(INPUT_PATH)

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

X = df[FEATURES].astype(float).to_numpy(dtype=np.float32)

if np.isnan(X).any() or np.isinf(X).any():
    raise ValueError("Real normalized dataset contains NaN/Inf.")

if X.min() < -1e-8 or X.max() > 1 + 1e-8:
    raise ValueError("Input is expected to be Min-Max normalized to [0,1].")

print("=" * 80)
print("FROZEN ORIGINAL DATA")
print("=" * 80)
print("Samples:", len(X))
print("Dimensions:", X.shape[1])
print("Min / Max:", float(X.min()), float(X.max()))


# ============================================================
# 4. Standardized analysis / GAN representation
#
# The paper mentions feature standardization, and its Fig.2(b)/Fig.3(c)
# contain negative values. The real 276 rows remain unchanged; this is
# only the internal representation used for GAN/K-means/plots.
# ============================================================

standard_scaler = StandardScaler()
X_std = standard_scaler.fit_transform(X).astype(np.float32)


# ============================================================
# 5. Freeze the paper-oriented K-means original baseline
# ============================================================

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    init="random",
    n_init=1,
    random_state=KMEANS_SEED,
    max_iter=300,
)

real_raw_labels = kmeans.fit_predict(X_std)
real_raw_counts = np.bincount(real_raw_labels, minlength=N_CLUSTERS)
real_raw_prop = real_raw_counts / len(X)

# Align arbitrary raw K-means labels to the paper's class1~class6 names.
# This ONLY renames clusters; it never moves samples between clusters.
cost = np.abs(
    real_raw_counts[:, None]
    - PAPER_ORIGINAL_COUNTS[None, :]
)
raw_idx, paper_idx = linear_sum_assignment(cost)
RAW_TO_PAPER = {int(r): int(p) for r, p in zip(raw_idx, paper_idx)}

real_paper_labels = np.array([RAW_TO_PAPER[int(v)] for v in real_raw_labels])
real_paper_counts = np.bincount(real_paper_labels, minlength=N_CLUSTERS)

print()
print("=" * 80)
print("FROZEN PAPER-ORIENTED K-MEANS")
print("=" * 80)
print("Raw counts:", real_raw_counts)
print("Paper-aligned counts:", real_paper_counts)
print("Paper counts:", PAPER_ORIGINAL_COUNTS)
print("Difference:", real_paper_counts - PAPER_ORIGINAL_COUNTS)


# ============================================================
# 6. Legal values for discrete columns
# ============================================================

LEGAL_VALUES = {
    feature: np.sort(df[feature].astype(float).unique())
    for feature in DISCRETE_FEATURES
}


def postprocess_from_standardized(synthetic_std: np.ndarray) -> np.ndarray:
    """Inverse-transform, enforce [0,1], and snap discrete variables."""
    synthetic = standard_scaler.inverse_transform(synthetic_std)
    synthetic = np.clip(synthetic, 0.0, 1.0).astype(np.float32)

    for feature, values in LEGAL_VALUES.items():
        j = FEATURES.index(feature)
        distance = np.abs(synthetic[:, [j]] - values.reshape(1, -1))
        synthetic[:, j] = values[distance.argmin(axis=1)]

    return synthetic


# ============================================================
# 7. Reference statistics / nearest-neighbor model
# ============================================================

real_corr = pd.DataFrame(X, columns=FEATURES).corr(method="spearman").to_numpy()
upper_triangle = np.triu_indices(len(FEATURES), k=1)

nearest_real = NearestNeighbors(n_neighbors=1)
nearest_real.fit(X_std)

real_rows_rounded = {
    tuple(row)
    for row in np.round(X, 8)
}


# ============================================================
# 8. GAN quality metrics
# ============================================================

def evaluate_synthetic(synthetic: np.ndarray) -> dict:
    synthetic = np.asarray(synthetic, dtype=np.float32)

    nan_count = int(np.isnan(synthetic).sum())
    inf_count = int(np.isinf(synthetic).sum())
    range_violations = int(np.sum((synthetic < 0) | (synthetic > 1)))

    if nan_count or inf_count:
        return {
            "invalid": nan_count + inf_count + range_violations,
            "coverage": 0,
            "exact_duplicates": -1,
        }

    ks_values = np.array([
        ks_2samp(X[:, j], synthetic[:, j]).statistic
        for j in range(len(FEATURES))
    ])

    wasserstein_values = np.array([
        wasserstein_distance(X[:, j], synthetic[:, j])
        for j in range(len(FEATURES))
    ])

    synthetic_corr = pd.DataFrame(
        synthetic,
        columns=FEATURES,
    ).corr(method="spearman").to_numpy()

    corr_mae = float(np.nanmean(np.abs(
        real_corr[upper_triangle]
        - synthetic_corr[upper_triangle]
    )))

    synthetic_std = standard_scaler.transform(synthetic)
    raw_labels = kmeans.predict(synthetic_std)
    raw_counts = np.bincount(raw_labels, minlength=N_CLUSTERS)
    raw_prop = raw_counts / len(synthetic)

    mode_js = float(jensenshannon(
        real_raw_prop + 1e-12,
        raw_prop + 1e-12,
        base=2.0,
    ) ** 2)

    distances, _ = nearest_real.kneighbors(synthetic_std)

    duplicate_count = sum(
        tuple(row) in real_rows_rounded
        for row in np.round(synthetic, 8)
    )

    # Score is an internal model-selection score, not a paper metric.
    score = float(
        0.48 * ks_values.mean()
        + 0.38 * corr_mae
        + 0.08 * mode_js
        + 0.04 * wasserstein_values.mean()
        + 0.02 * (duplicate_count / max(1, len(synthetic)))
    )

    return {
        "score": score,
        "mean_ks": float(ks_values.mean()),
        "max_ks": float(ks_values.max()),
        "mean_wasserstein": float(wasserstein_values.mean()),
        "corr_mae": corr_mae,
        "mode_js": mode_js,
        "coverage": int(np.sum(raw_counts > 0)),
        "min_cluster_count": int(raw_counts.min()),
        "raw_cluster_counts": raw_counts,
        "mean_nearest_real_distance": float(distances.mean()),
        "min_nearest_real_distance": float(distances.min()),
        "exact_duplicates": int(duplicate_count),
        "invalid": int(nan_count + inf_count + range_violations),
        "ks_by_feature": ks_values,
        "wasserstein_by_feature": wasserstein_values,
    }


# ============================================================
# 9. GAN architecture
#
# Unconditional: G(z), not G(z,c).
# Objective: BCE adversarial loss.
# ============================================================

class Generator(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        layers = []
        d = latent_dim

        for h in hidden_dims:
            layers.extend([
                nn.Linear(d, h),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(h),
            ])
            d = h

        # Standardized output can be negative, so no Sigmoid here.
        layers.append(nn.Linear(d, len(FEATURES)))
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class Discriminator(nn.Module):
    def __init__(
        self,
        hidden_dims: tuple[int, ...],
        spectral_norm: bool,
    ):
        super().__init__()
        layers = []
        d = len(FEATURES)

        for h in hidden_dims:
            linear = nn.Linear(d, h)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)

            layers.extend([
                linear,
                nn.LeakyReLU(0.2),
            ])
            d = h

        linear = nn.Linear(d, 1)
        if spectral_norm:
            linear = nn.utils.spectral_norm(linear)

        layers.append(linear)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================
# 10. Candidate configurations
#
# baseline: close to the previous version
# sn_shallow: spectral-normalized D
# sn_deep: deeper G + spectral-normalized D + large batch
# sn_deep_strong_d: neighboring learning-rate check
# ============================================================

if QUICK_MODE:
    CONFIGS = {
        "quick_sn_deep": {
            "latent_dim": 16,
            "g_hidden": (128, 256, 128),
            "d_hidden": (128, 64),
            "batch_size": 128,
            "lr_g": 2e-4,
            "lr_d": 1.25e-4,
            "instance_noise": 0.02,
            "real_label": 0.90,
            "spectral_norm": True,
        },
    }
    CHECKPOINTS = [300, 600]
    EVAL_POOL_SIZE = 1500
else:
    CONFIGS = {
        "baseline": {
            "latent_dim": 16,
            "g_hidden": (64, 128, 64),
            "d_hidden": (128, 64),
            "batch_size": 64,
            "lr_g": 2e-4,
            "lr_d": 5e-5,
            "instance_noise": 0.05,
            "real_label": 0.90,
            "spectral_norm": False,
        },
        "sn_shallow": {
            "latent_dim": 16,
            "g_hidden": (64, 128, 64),
            "d_hidden": (128, 64),
            "batch_size": 256,
            "lr_g": 2e-4,
            "lr_d": 1.25e-4,
            "instance_noise": 0.02,
            "real_label": 0.90,
            "spectral_norm": True,
        },
        "sn_deep": {
            "latent_dim": 16,
            "g_hidden": (128, 256, 128),
            "d_hidden": (128, 64),
            "batch_size": 256,
            "lr_g": 2e-4,
            "lr_d": 1.25e-4,
            "instance_noise": 0.02,
            "real_label": 0.90,
            "spectral_norm": True,
        },
        "sn_deep_strong_d": {
            "latent_dim": 16,
            "g_hidden": (128, 256, 128),
            "d_hidden": (128, 64),
            "batch_size": 256,
            "lr_g": 2e-4,
            "lr_d": 1.50e-4,
            "instance_noise": 0.02,
            "real_label": 0.90,
            "spectral_norm": True,
        },
    }

    CHECKPOINTS = [
        600,
        900,
        1200,
        1500,
        1800,
        2100,
        2400,
    ]
    EVAL_POOL_SIZE = 5000


# ============================================================
# 11. Training function
#
# IMPORTANT: evaluation uses a separate torch.Generator so it never
# perturbs the training RNG stream.
# ============================================================

def train_one(
    config_name: str,
    config: dict,
    train_seed: int,
):
    set_training_seed(train_seed)
    np_rng = np.random.default_rng(train_seed)

    generator = Generator(
        latent_dim=config["latent_dim"],
        hidden_dims=config["g_hidden"],
    )

    discriminator = Discriminator(
        hidden_dims=config["d_hidden"],
        spectral_norm=config["spectral_norm"],
    )

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=config["lr_g"],
        betas=(0.5, 0.999),
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=config["lr_d"],
        betas=(0.5, 0.999),
    )

    criterion = nn.BCEWithLogitsLoss()
    real_tensor = torch.tensor(X_std, dtype=torch.float32)

    batch_size = config["batch_size"]
    max_steps = max(CHECKPOINTS)

    rows = []
    states = {}

    for step in range(1, max_steps + 1):
        # -----------------------------
        # Random real minibatch.
        # No K-means balancing is used.
        # -----------------------------
        indices = torch.tensor(
            np_rng.integers(0, len(real_tensor), size=batch_size),
            dtype=torch.long,
        )
        real_batch = real_tensor[indices]

        noise_std = (
            config["instance_noise"]
            * max(0.0, 1.0 - step / max_steps)
        )

        # -----------------------------
        # Train D
        # -----------------------------
        fake_detached = generator(torch.randn(
            batch_size,
            config["latent_dim"],
        )).detach()

        noisy_real = real_batch + noise_std * torch.randn_like(real_batch)
        noisy_fake = fake_detached + noise_std * torch.randn_like(fake_detached)

        real_logits = discriminator(noisy_real)
        fake_logits = discriminator(noisy_fake)

        d_loss = (
            criterion(
                real_logits,
                torch.full_like(real_logits, config["real_label"]),
            )
            + criterion(
                fake_logits,
                torch.zeros_like(fake_logits),
            )
        )

        optimizer_d.zero_grad(set_to_none=True)
        d_loss.backward()
        optimizer_d.step()

        # -----------------------------
        # Train G
        # -----------------------------
        fake = generator(torch.randn(
            batch_size,
            config["latent_dim"],
        ))
        fake_logits = discriminator(fake)

        g_loss = criterion(
            fake_logits,
            torch.ones_like(fake_logits),
        )

        optimizer_g.zero_grad(set_to_none=True)
        g_loss.backward()
        optimizer_g.step()

        # -----------------------------
        # Synthetic-only checkpoint evaluation
        # -----------------------------
        if step in CHECKPOINTS:
            generator.eval()

            eval_generator = torch.Generator().manual_seed(
                train_seed * 100000 + step
            )

            with torch.no_grad():
                synthetic_std = generator(torch.randn(
                    EVAL_POOL_SIZE,
                    config["latent_dim"],
                    generator=eval_generator,
                )).cpu().numpy()

            generator.train()

            synthetic = postprocess_from_standardized(synthetic_std)
            metrics = evaluate_synthetic(synthetic)

            hard_valid = (
                metrics["invalid"] == 0
                and metrics["exact_duplicates"] == 0
                and metrics["coverage"] == N_CLUSTERS
                and metrics["min_cluster_count"] >= max(5, int(0.005 * EVAL_POOL_SIZE))
            )

            row = {
                "config": config_name,
                "train_seed": train_seed,
                "step": step,
                "d_loss": float(d_loss.item()),
                "g_loss": float(g_loss.item()),
                "hard_valid": bool(hard_valid),
                "score": metrics["score"],
                "mean_ks": metrics["mean_ks"],
                "max_ks": metrics["max_ks"],
                "corr_mae": metrics["corr_mae"],
                "mean_wasserstein": metrics["mean_wasserstein"],
                "mode_js": metrics["mode_js"],
                "coverage": metrics["coverage"],
                "min_cluster_count": metrics["min_cluster_count"],
                "cluster_counts": format_counts(metrics["raw_cluster_counts"]),
                "mean_nearest_real_distance": metrics["mean_nearest_real_distance"],
                "exact_duplicates": metrics["exact_duplicates"],
                "invalid": metrics["invalid"],
            }

            rows.append(row)
            states[step] = copy.deepcopy(generator.state_dict())

            print(
                f"{config_name:18s} seed={train_seed:4d} "
                f"step={step:4d} "
                f"KS={metrics['mean_ks']:.3f} "
                f"Corr={metrics['corr_mae']:.3f} "
                f"JS={metrics['mode_js']:.4f} "
                f"coverage={metrics['coverage']}/6 "
                f"valid={hard_valid}"
            )

    return rows, states


# ============================================================
# 12. Run multi-seed tuning
# ============================================================

all_rows = []
all_states = {}

for config_name, config in CONFIGS.items():
    print()
    print("=" * 80)
    print("TRAIN:", config_name)
    print("=" * 80)

    for train_seed in TRAIN_SEEDS:
        rows, states = train_one(
            config_name,
            config,
            train_seed,
        )

        all_rows.extend(rows)

        for step, state in states.items():
            all_states[(config_name, train_seed, step)] = state


tuning_df = pd.DataFrame(all_rows)
tuning_df.to_excel(
    RESULTS_DIR / "07_gan_tuning_all_runs.xlsx",
    index=False,
)


# ============================================================
# 13. Robust config/checkpoint selection across seeds
# ============================================================

valid_df = tuning_df[tuning_df["hard_valid"]].copy()

if valid_df.empty:
    raise RuntimeError(
        "No GAN checkpoint passed the hard quality gates."
    )

summary = (
    valid_df
    .groupby(["config", "step"])
    .agg(
        n_valid_seeds=("train_seed", "nunique"),
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        mean_ks=("mean_ks", "mean"),
        std_ks=("mean_ks", "std"),
        mean_corr_mae=("corr_mae", "mean"),
        std_corr_mae=("corr_mae", "std"),
        mean_wasserstein=("mean_wasserstein", "mean"),
        mean_mode_js=("mode_js", "mean"),
        min_coverage=("coverage", "min"),
    )
    .reset_index()
)

# Require every planned training seed to pass the hard gates.
summary = summary[
    summary["n_valid_seeds"] == len(TRAIN_SEEDS)
].copy()

if summary.empty:
    raise RuntimeError(
        "No config+checkpoint passed the hard gates for every training seed."
    )

summary = summary.sort_values(
    ["mean_score", "mean_ks", "mean_corr_mae"]
).reset_index(drop=True)

summary.to_excel(
    RESULTS_DIR / "07_gan_tuning_summary.xlsx",
    index=False,
)

best = summary.iloc[0]
best_config_name = str(best["config"])
best_step = int(best["step"])
best_config = CONFIGS[best_config_name]

# Select a representative seed: closest individual score to the
# multi-seed mean, rather than the lucky best seed.
subset = valid_df[
    (valid_df["config"] == best_config_name)
    & (valid_df["step"] == best_step)
].copy()

subset["distance_to_group_mean"] = np.abs(
    subset["score"] - best["mean_score"]
)

representative = subset.sort_values(
    "distance_to_group_mean"
).iloc[0]

representative_seed = int(representative["train_seed"])

print()
print("=" * 80)
print("SELECTED ROBUST GAN")
print("=" * 80)
print("Config:", best_config_name)
print("Checkpoint:", best_step)
print("Multi-seed mean KS:", float(best["mean_ks"]))
print("Multi-seed mean CorrMAE:", float(best["mean_corr_mae"]))
print("Representative seed:", representative_seed)


# ============================================================
# 14. Restore selected representative model
# ============================================================

final_generator = Generator(
    latent_dim=best_config["latent_dim"],
    hidden_dims=best_config["g_hidden"],
)

final_generator.load_state_dict(
    all_states[(
        best_config_name,
        representative_seed,
        best_step,
    )]
)

final_generator.eval()


torch.save(
    {
        "generator_state_dict": final_generator.state_dict(),
        "config_name": best_config_name,
        "config": best_config,
        "checkpoint": best_step,
        "representative_seed": representative_seed,
        "features": FEATURES,
    },
    RESULTS_DIR / "07_generator_optimized.pt",
)


# ============================================================
# 15. Large natural candidate pool for final GAN validation
# ============================================================

FINAL_POOL_SIZE = 3000 if QUICK_MODE else 30000
pool_rng = torch.Generator().manual_seed(POOL_GENERATION_SEED)

with torch.no_grad():
    pool_std = final_generator(torch.randn(
        FINAL_POOL_SIZE,
        best_config["latent_dim"],
        generator=pool_rng,
    )).cpu().numpy()

pool = postprocess_from_standardized(pool_std)
pool_metrics = evaluate_synthetic(pool)

print()
print("=" * 80)
print("FINAL NATURAL POOL VALIDATION")
print("=" * 80)
print("Pool size:", len(pool))
print("Mean KS:", pool_metrics["mean_ks"])
print("CorrMAE:", pool_metrics["corr_mae"])
print("Mode JS:", pool_metrics["mode_js"])
print("Coverage:", pool_metrics["coverage"], "/ 6")
print("Raw cluster counts:", pool_metrics["raw_cluster_counts"])
print("Exact duplicates:", pool_metrics["exact_duplicates"])
print("Invalid:", pool_metrics["invalid"])


# ============================================================
# 16. Natural 72
#
# We do NOT force per-cluster counts.
# We only require hard validity + 6/6 coverage.
# The first generation seed that passes is accepted; we do NOT choose
# the set with the prettiest KS/Corr values.
# ============================================================

natural_72 = None
natural_seed = None
natural_metrics = None

for generation_seed in range(100, 1100):
    rng = torch.Generator().manual_seed(generation_seed)

    with torch.no_grad():
        synthetic_std = final_generator(torch.randn(
            N_SYNTHETIC,
            best_config["latent_dim"],
            generator=rng,
        )).cpu().numpy()

    candidate = postprocess_from_standardized(synthetic_std)
    metrics = evaluate_synthetic(candidate)

    if (
        metrics["invalid"] == 0
        and metrics["exact_duplicates"] == 0
        and metrics["coverage"] == N_CLUSTERS
    ):
        natural_72 = candidate
        natural_seed = generation_seed
        natural_metrics = metrics
        break

if natural_72 is None:
    raise RuntimeError(
        "Could not obtain a natural 72-sample draw covering all six modes."
    )

natural_raw_labels = kmeans.predict(
    standard_scaler.transform(natural_72)
)

natural_paper_labels = np.array([
    RAW_TO_PAPER[int(v)]
    for v in natural_raw_labels
])

natural_paper_counts = np.bincount(
    natural_paper_labels,
    minlength=N_CLUSTERS,
)

print()
print("=" * 80)
print("NATURAL 72")
print("=" * 80)
print("Generation seed:", natural_seed)
print("Paper-aligned counts:", natural_paper_counts)
print("Mean KS:", natural_metrics["mean_ks"])
print("CorrMAE:", natural_metrics["corr_mae"])
print("Exact duplicates:", natural_metrics["exact_duplicates"])


# ============================================================
# 17. Soft-balanced 72
#
# This is NOT forced 12-per-cluster sampling.
# Target mixture = alpha * empirical real proportions
#                + (1-alpha) * uniform proportions
#
# With alpha=0.70, most of the natural mixture is retained while the
# rare modes receive a modest boost, which is consistent with the
# augmentation intent of the paper without reproducing its bar chart by
# construction.
# ============================================================

pool_raw_labels = kmeans.predict(
    standard_scaler.transform(pool)
)

uniform_prop = np.ones(N_CLUSTERS) / N_CLUSTERS
soft_target_prop = (
    SOFT_BALANCE_ALPHA * real_raw_prop
    + (1.0 - SOFT_BALANCE_ALPHA) * uniform_prop
)

raw_expected = soft_target_prop * N_SYNTHETIC
soft_target_counts = np.floor(raw_expected).astype(int)
remainder = N_SYNTHETIC - soft_target_counts.sum()
fractional = raw_expected - soft_target_counts

for k in np.argsort(-fractional)[:remainder]:
    soft_target_counts[k] += 1

soft_rng = np.random.default_rng(SOFT_SELECTION_SEED)
selected_indices = []

for raw_cluster, n_needed in enumerate(soft_target_counts):
    candidates = np.where(pool_raw_labels == raw_cluster)[0]

    if len(candidates) < n_needed:
        raise RuntimeError(
            f"Pool cluster {raw_cluster} has {len(candidates)} rows, "
            f"but {n_needed} are required."
        )

    chosen = soft_rng.choice(
        candidates,
        size=n_needed,
        replace=False,
    )
    selected_indices.extend(chosen.tolist())

soft_72 = pool[np.array(selected_indices)]
soft_metrics = evaluate_synthetic(soft_72)

soft_raw_labels = kmeans.predict(
    standard_scaler.transform(soft_72)
)
soft_paper_labels = np.array([
    RAW_TO_PAPER[int(v)]
    for v in soft_raw_labels
])
soft_paper_counts = np.bincount(
    soft_paper_labels,
    minlength=N_CLUSTERS,
)

print()
print("=" * 80)
print("SOFT-BALANCED 72")
print("=" * 80)
print("Alpha:", SOFT_BALANCE_ALPHA)
print("Paper-aligned counts:", soft_paper_counts)
print("Mean KS:", soft_metrics["mean_ks"])
print("CorrMAE:", soft_metrics["corr_mae"])
print("Exact duplicates:", soft_metrics["exact_duplicates"])


# ============================================================
# 18. Save synthetic datasets
# ============================================================

def make_output_df(data, paper_labels, source_name):
    out = pd.DataFrame(data, columns=FEATURES)
    out.insert(0, "sample_id", [
        f"{source_name}_{i+1:03d}"
        for i in range(len(out))
    ])
    out["source"] = source_name
    out["paper_aligned_cluster"] = paper_labels + 1
    return out


natural_df = make_output_df(
    natural_72,
    natural_paper_labels,
    "GAN_natural",
)

soft_df = make_output_df(
    soft_72,
    soft_paper_labels,
    "GAN_softbalanced",
)

natural_df.to_excel(
    RESULTS_DIR / "07_synthetic_72_natural.xlsx",
    index=False,
)

soft_df.to_excel(
    RESULTS_DIR / "07_synthetic_72_softbalanced.xlsx",
    index=False,
)

# Augmented versions
real_output = pd.DataFrame(X, columns=FEATURES)
real_output["source"] = "real"
real_output["paper_aligned_cluster"] = real_paper_labels + 1

pd.concat([
    real_output,
    natural_df[FEATURES + ["source", "paper_aligned_cluster"]],
], ignore_index=True).to_excel(
    RESULTS_DIR / "07_augmented_natural.xlsx",
    index=False,
)

pd.concat([
    real_output,
    soft_df[FEATURES + ["source", "paper_aligned_cluster"]],
], ignore_index=True).to_excel(
    RESULTS_DIR / "07_augmented_softbalanced.xlsx",
    index=False,
)


# ============================================================
# 19. Per-feature diagnostic tables
# ============================================================

def per_feature_table(synthetic, label):
    rows = []

    for j, feature in enumerate(FEATURES):
        rows.append({
            "dataset": label,
            "feature": feature,
            "KS": ks_2samp(X[:, j], synthetic[:, j]).statistic,
            "Wasserstein": wasserstein_distance(X[:, j], synthetic[:, j]),
            "real_median": np.median(X[:, j]),
            "synthetic_median": np.median(synthetic[:, j]),
            "real_q05": np.quantile(X[:, j], 0.05),
            "real_q95": np.quantile(X[:, j], 0.95),
            "synthetic_q05": np.quantile(synthetic[:, j], 0.05),
            "synthetic_q95": np.quantile(synthetic[:, j], 0.95),
        })

    return pd.DataFrame(rows)


per_feature = pd.concat([
    per_feature_table(natural_72, "natural72"),
    per_feature_table(soft_72, "softbalanced72"),
], ignore_index=True)

per_feature.to_excel(
    RESULTS_DIR / "07_per_feature_validation.xlsx",
    index=False,
)


# ============================================================
# 20. Summary workbook
# ============================================================

pool_summary = pd.DataFrame({
    "metric": [
        "mean_ks",
        "corr_mae",
        "mean_wasserstein",
        "mode_js",
        "coverage",
        "mean_nearest_real_distance",
        "exact_duplicates",
        "invalid",
    ],
    "value": [
        pool_metrics["mean_ks"],
        pool_metrics["corr_mae"],
        pool_metrics["mean_wasserstein"],
        pool_metrics["mode_js"],
        pool_metrics["coverage"],
        pool_metrics["mean_nearest_real_distance"],
        pool_metrics["exact_duplicates"],
        pool_metrics["invalid"],
    ],
})

sample_summary = pd.DataFrame([
    {
        "dataset": "natural72",
        "mean_ks": natural_metrics["mean_ks"],
        "corr_mae": natural_metrics["corr_mae"],
        "mode_js": natural_metrics["mode_js"],
        "coverage": natural_metrics["coverage"],
        "mean_nearest_real_distance": natural_metrics["mean_nearest_real_distance"],
        "exact_duplicates": natural_metrics["exact_duplicates"],
        "counts_paper_aligned": format_counts(natural_paper_counts),
    },
    {
        "dataset": "softbalanced72",
        "mean_ks": soft_metrics["mean_ks"],
        "corr_mae": soft_metrics["corr_mae"],
        "mode_js": soft_metrics["mode_js"],
        "coverage": soft_metrics["coverage"],
        "mean_nearest_real_distance": soft_metrics["mean_nearest_real_distance"],
        "exact_duplicates": soft_metrics["exact_duplicates"],
        "counts_paper_aligned": format_counts(soft_paper_counts),
    },
])

count_comparison = pd.DataFrame({
    "class": [f"class{i+1}" for i in range(N_CLUSTERS)],
    "paper_original": PAPER_ORIGINAL_COUNTS,
    "our_original": real_paper_counts,
    "paper_after": PAPER_AFTER_COUNTS,
    "natural_added": natural_paper_counts,
    "natural_after": real_paper_counts + natural_paper_counts,
    "soft_added": soft_paper_counts,
    "soft_after": real_paper_counts + soft_paper_counts,
})

with pd.ExcelWriter(RESULTS_DIR / "07_validation_summary.xlsx") as writer:
    summary.to_excel(writer, sheet_name="tuning_summary", index=False)
    tuning_df.to_excel(writer, sheet_name="all_tuning_runs", index=False)
    pool_summary.to_excel(writer, sheet_name="final_pool", index=False)
    sample_summary.to_excel(writer, sheet_name="sample_sets", index=False)
    count_comparison.to_excel(writer, sheet_name="cluster_counts", index=False)
    per_feature.to_excel(writer, sheet_name="per_feature", index=False)


# ============================================================
# 21. Plot helpers
# ============================================================

real_plot_std = standard_scaler.transform(X)
natural_plot_std = standard_scaler.transform(natural_72)
soft_plot_std = standard_scaler.transform(soft_72)


def plot_histogram(synthetic_std, title, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        real_plot_std.ravel(),
        bins=35,
        density=True,
        alpha=0.55,
        label="Original",
    )
    ax.hist(
        synthetic_std.ravel(),
        bins=35,
        density=True,
        alpha=0.55,
        label="Synthetic",
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot(synthetic_std, title, path):
    feature_labels = [f"feature{i+1}" for i in range(len(FEATURES))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    axes[0].boxplot(
        [real_plot_std[:, j] for j in range(len(FEATURES))],
        tick_labels=feature_labels,
    )
    axes[1].boxplot(
        [synthetic_std[:, j] for j in range(len(FEATURES))],
        tick_labels=feature_labels,
    )

    axes[0].set_title("Original Data")
    axes[1].set_title(title)

    for ax in axes:
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_separate_tsne(
    synthetic_std,
    synthetic_paper_labels,
    synthetic_title,
    path,
):
    real_tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="pca",
        learning_rate="auto",
    ).fit_transform(real_plot_std)

    syn_perplexity = min(20, max(5, (len(synthetic_std) - 1) // 3))
    synthetic_tsne = TSNE(
        n_components=2,
        perplexity=syn_perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    ).fit_transform(synthetic_std)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for k in range(N_CLUSTERS):
        mr = real_paper_labels == k
        ms = synthetic_paper_labels == k

        axes[0].scatter(
            real_tsne[mr, 0],
            real_tsne[mr, 1],
            s=16,
            label=f"class{k+1}",
        )
        axes[1].scatter(
            synthetic_tsne[ms, 0],
            synthetic_tsne[ms, 1],
            s=25,
            label=f"class{k+1}",
        )

    axes[0].set_title("Original Data")
    axes[1].set_title(synthetic_title)

    for ax in axes:
        ax.set_xlabel("tsne dimension1")
        ax.set_ylabel("tsne dimension2")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_joint_tsne(synthetic_std, title, path):
    combined = np.vstack([real_plot_std, synthetic_std])

    embedded = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="pca",
        learning_rate="auto",
    ).fit_transform(combined)

    n_real = len(real_plot_std)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        embedded[:n_real, 0],
        embedded[:n_real, 1],
        s=18,
        alpha=0.65,
        label="Real",
    )
    ax.scatter(
        embedded[n_real:, 0],
        embedded[n_real:, 1],
        s=34,
        marker="x",
        label="Synthetic",
    )
    ax.set_title(title)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 22. Main paper-style figures
#
# The soft-balanced set is used for the main paper-oriented four figures.
# Natural-only figures are also saved so the GAN itself can be assessed
# without the soft balancing step.
# ============================================================

plot_histogram(
    soft_plot_std,
    "Original vs Soft-Balanced Synthetic Data",
    RESULTS_DIR / "07_fig2b_histogram.png",
)

# Fig.3(a): no forced +12. Show the actual soft-balanced increments.
soft_after_counts = real_paper_counts + soft_paper_counts
class_names = [f"class{i+1}" for i in range(N_CLUSTERS)]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, counts, title in [
    (axes[0], real_paper_counts, "Number of original samples"),
    (axes[1], soft_after_counts, "Number of samples after augmentation"),
]:
    bars = ax.bar(class_names, counts)
    ax.set_title(title)
    ax.set_ylabel("num")
    ax.tick_params(axis="x", rotation=35)

    for bar, value in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(value)),
            ha="center",
            va="bottom",
        )

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "07_fig3a_kmeans.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

plot_separate_tsne(
    soft_plot_std,
    soft_paper_labels,
    "Synthetic Data",
    RESULTS_DIR / "07_fig3b_tsne.png",
)

plot_boxplot(
    soft_plot_std,
    "Synthetic Data",
    RESULTS_DIR / "07_fig3c_boxplot.png",
)

# Additional scientific diagnostics
plot_histogram(
    natural_plot_std,
    "Original vs Natural Synthetic Data",
    RESULTS_DIR / "07_histogram_natural.png",
)

plot_joint_tsne(
    natural_plot_std,
    "Joint t-SNE: Real + Natural Synthetic",
    RESULTS_DIR / "07_joint_tsne_natural.png",
)

plot_joint_tsne(
    soft_plot_std,
    "Joint t-SNE: Real + Soft-Balanced Synthetic",
    RESULTS_DIR / "07_joint_tsne_softbalanced.png",
)


# ============================================================
# 23. Final textual assessment
# ============================================================

print()
print("=" * 80)
print("07 FINAL SUMMARY")
print("=" * 80)

print("Selected model:", best_config_name)
print("Selected checkpoint:", best_step)
print("Representative training seed:", representative_seed)
print()

print("Large natural pool:")
print("  mean KS:", pool_metrics["mean_ks"])
print("  CorrMAE:", pool_metrics["corr_mae"])
print("  coverage:", pool_metrics["coverage"], "/ 6")
print("  duplicates:", pool_metrics["exact_duplicates"])
print("  invalid:", pool_metrics["invalid"])
print()

print("Natural 72:")
print("  counts:", natural_paper_counts)
print("  mean KS:", natural_metrics["mean_ks"])
print("  CorrMAE:", natural_metrics["corr_mae"])
print()

print("Soft-balanced 72:")
print("  counts:", soft_paper_counts)
print("  mean KS:", soft_metrics["mean_ks"])
print("  CorrMAE:", soft_metrics["corr_mae"])
print()

pool_pass = (
    pool_metrics["mean_ks"] < TARGET_MEAN_KS
    and pool_metrics["corr_mae"] < TARGET_CORR_MAE
    and pool_metrics["coverage"] == N_CLUSTERS
    and pool_metrics["exact_duplicates"] == 0
    and pool_metrics["invalid"] == 0
)

print("Pool passes internal acceptance gates:", pool_pass)
print(
    "Pool reaches ideal mean-KS target:",
    pool_metrics["mean_ks"] < IDEAL_MEAN_KS,
)
print(
    "Pool reaches ideal CorrMAE target:",
    pool_metrics["corr_mae"] < IDEAL_CORR_MAE,
)

print()
print("Generated files are under results/ with prefix 07_.")