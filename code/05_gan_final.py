
#
# 目标：
# 1. 严格保留论文的研究框架：
#       normalized data -> classical GAN augmentation
#       -> histogram / K-means / t-SNE / boxplot validation
#
# 2. 不为了“像论文”硬改数据。
# 3. 在论文未公开的 GAN 超参数上，用多随机种子 + 多 checkpoint
#    做科学调参，选择统计表现稳定的 Vanilla GAN。
#
# 重要说明：
# - GAN 本身仍是经典 Vanilla GAN：
#       Generator + Discriminator + BCE adversarial loss
# - 不使用 WGAN / WGAN-GP / CTGAN。
# - K-means 只用于 GAN 结果评价和最终分层抽样，
#   不作为 GAN 的条件输入，因此 GAN 本身仍是无条件 GAN。
# ============================================================

from pathlib import Path
import os
import random
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn

from scipy.stats import ks_2samp, wasserstein_distance

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 0. 全局设置
# ============================================================

RANDOM_SEED = 42
FINAL_TRAIN_SEED = 2026
FINAL_SELECTION_SEED = 2026

N_CLUSTERS = 6
SYNTHETIC_PER_CLUSTER = 12
FINAL_SYNTHETIC_NUM = N_CLUSTERS * SYNTHETIC_PER_CLUSTER

QUICK_MODE = os.getenv("GAN_QUICK", "0") == "1"

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)


# ============================================================
# 1. 特征
#
# 论文 Fig.3(c) 有 feature1 ~ feature9，
# 因此这里使用完整 9 个数值变量：
#
# 8 个输入特征 + Lifetime
# ============================================================

features = [
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


# ============================================================
# 2. 读取数据
#
# 优先使用已经归一化好的：
# results/04_normalized_data.xlsx
#
# 若不存在，则读取：
# results/04_clean_data.xlsx
# 并在本程序中进行 Min-Max。
# ============================================================

normalized_path = results_dir / "04_normalized_data.xlsx"
clean_path = results_dir / "04_clean_data.xlsx"

if normalized_path.exists():
    source_path = normalized_path
    df = pd.read_excel(source_path)

    X_df = df[features].astype(float).copy()

    # 检查是否确实处于 [0,1]
    if (
        X_df.min().min() < -1e-8
        or
        X_df.max().max() > 1 + 1e-8
    ):
        raise ValueError(
            "04_normalized_data.xlsx 中的数值不在 [0,1]。"
            "请检查上一阶段 normalization。"
        )

    minmax_scaler = None

elif clean_path.exists():
    source_path = clean_path
    df = pd.read_excel(source_path)

    raw_X = df[features].astype(float).copy()

    minmax_scaler = MinMaxScaler()
    X_array = minmax_scaler.fit_transform(raw_X)

    X_df = pd.DataFrame(
        X_array,
        columns=features,
        index=df.index,
    )

else:
    raise FileNotFoundError(
        "找不到 results/04_normalized_data.xlsx "
        "或 results/04_clean_data.xlsx"
    )


X = X_df.to_numpy(dtype=np.float32)

print("=" * 80)
print("INPUT DATA")
print("=" * 80)
print("Source:", source_path)
print("Samples:", len(X))
print("Dimensions:", X.shape[1])
print("Min:", X.min())
print("Max:", X.max())


# ============================================================
# 3. 为什么还要 StandardScaler？
#
# 输入文件仍然是论文明确使用的 Min-Max normalized data。
#
# 但是论文同时提到 feature standardization，
# 且 Fig.2(b) / Fig.3(c) 中存在明显负数。
#
# 因此 GAN 与 K-means 的工作空间使用：
#
# z = (x - mean) / std
#
# Min-Max 数据没有被丢弃。
# StandardScaler 只是 GAN 内部训练表示。
# ============================================================

standard_scaler = StandardScaler()

X_std = standard_scaler.fit_transform(
    X
).astype(np.float32)


# ============================================================
# 4. K-means：论文方法约束
#
# 论文：
# - K = 6
# - feature standardization
# - Fig.3(c) 有 9 个 features
#
# 所以正式评价采用：
#
# Standardized 9-D + K=6
#
# 使用 n_init=50 是为了获得稳定解，而不是为了凑论文柱状图。
# ============================================================

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    init="k-means++",
    n_init=50,
    random_state=RANDOM_SEED,
)

real_cluster = kmeans.fit_predict(
    X_std
)

real_cluster_counts = np.bincount(
    real_cluster,
    minlength=N_CLUSTERS,
)

silhouette = silhouette_score(
    X_std,
    real_cluster,
)

print()
print("=" * 80)
print("STABLE K-MEANS REFERENCE")
print("=" * 80)
print("Cluster counts:", real_cluster_counts)
print(
    "Sorted counts:",
    np.sort(real_cluster_counts),
)
print(
    "Silhouette:",
    round(float(silhouette), 4),
)


# ============================================================
# 5. 仅用于解释论文 Fig.3(a) 差异的诊断
#
# 这里会寻找“单次初始化”时是否能出现更接近论文的 cluster size。
#
# 重要：
# 该结果绝不用于 GAN 训练或最终数据选择。
# 这是为了证明：
#
# K-means 的 initialization / seed
# 确实会显著改变 cluster sizes。
# ============================================================

paper_counts_sorted = np.sort(
    np.array(
        [37, 59, 3, 53, 114, 9]
    )
)

best_diag = None

diag_seed_limit = 200 if QUICK_MODE else 1000

for seed in range(diag_seed_limit):

    km_diag = KMeans(
        n_clusters=N_CLUSTERS,
        init="random",
        n_init=1,
        random_state=seed,
    )

    labels_diag = km_diag.fit_predict(
        X_std
    )

    counts_diag = np.sort(
        np.bincount(
            labels_diag,
            minlength=N_CLUSTERS,
        )
    )

    distance = np.abs(
        counts_diag
        -
        paper_counts_sorted
    ).sum()

    if (
        best_diag is None
        or
        distance < best_diag["distance"]
    ):
        best_diag = {
            "seed": seed,
            "counts": counts_diag,
            "distance": int(distance),
        }


print()
print("=" * 80)
print("PAPER-COUNT DIAGNOSTIC ONLY")
print("=" * 80)
print(
    "Paper sorted counts:",
    paper_counts_sorted,
)
print(
    "Closest single-start diagnostic:",
    best_diag,
)
print(
    "NOTE: this diagnostic is NOT used downstream."
)


# ============================================================
# 6. 保存 scaler / K-means 信息
# ============================================================

scaler_info = pd.DataFrame({
    "feature": features,
    "mean_in_minmax_space": standard_scaler.mean_,
    "std_in_minmax_space": standard_scaler.scale_,
})

scaler_info.to_excel(
    results_dir / "05_standard_scaler_parameters.xlsx",
    index=False,
)

kmeans_info = pd.DataFrame({
    "cluster": np.arange(1, N_CLUSTERS + 1),
    "real_count": real_cluster_counts,
})

kmeans_info.to_excel(
    results_dir / "05_kmeans_reference.xlsx",
    index=False,
)


# ============================================================
# 7. 离散变量合法值
#
# Layer 和 valence_electron 在当前 normalized data 中
# 是离散变量。
#
# GAN 会输出连续值。
# 逆 standardization 后把它们映射到最近合法值。
# ============================================================

discrete_features = [
    "Layer",
    "valence_electron",
]

valid_discrete_values = {}

for feature in discrete_features:

    valid_discrete_values[feature] = np.sort(
        X_df[feature].unique()
    )


def nearest_valid_value(
    values,
    valid_values,
):

    values = np.asarray(values)

    distance = np.abs(
        values[:, None]
        -
        valid_values[None, :]
    )

    nearest_index = distance.argmin(
        axis=1
    )

    return valid_values[
        nearest_index
    ]


# ============================================================
# 8. Generator
#
# 经典 Vanilla GAN：
#
# z -> G(z) -> 9-D standardized sample
#
# 因为输出空间是 standardized data，
# 最后一层不用 Sigmoid。
# ============================================================

class Generator(nn.Module):

    def __init__(
        self,
        latent_dim,
        hidden_dims,
    ):

        super().__init__()

        layers = []

        input_dim = latent_dim

        for hidden_dim in hidden_dims:

            layers.append(
                nn.Linear(
                    input_dim,
                    hidden_dim,
                )
            )

            layers.append(
                nn.LeakyReLU(
                    0.2
                )
            )

            layers.append(
                nn.BatchNorm1d(
                    hidden_dim
                )
            )

            input_dim = hidden_dim

        layers.append(
            nn.Linear(
                input_dim,
                len(features),
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        z,
    ):

        return self.network(
            z
        )


# ============================================================
# 9. Discriminator
#
# 输入 standardized 9-D sample
#
# 输出 logits。
#
# 与 BCEWithLogitsLoss 配合。
# ============================================================

class Discriminator(nn.Module):

    def __init__(
        self,
        hidden_dims,
        dropout,
    ):

        super().__init__()

        layers = []

        input_dim = len(features)

        for hidden_dim in hidden_dims:

            layers.append(
                nn.Linear(
                    input_dim,
                    hidden_dim,
                )
            )

            layers.append(
                nn.LeakyReLU(
                    0.2
                )
            )

            if dropout > 0:

                layers.append(
                    nn.Dropout(
                        dropout
                    )
                )

            input_dim = hidden_dim

        layers.append(
            nn.Linear(
                input_dim,
                1,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x,
    ):

        return self.network(
            x
        )


# ============================================================
# 10. 后处理
#
# standardized synthetic
# -> inverse StandardScaler
# -> Min-Max space
# -> 物理边界 [0,1]
# -> 离散变量映射
# ============================================================

def postprocess_synthetic(
    synthetic_std,
):

    synthetic = standard_scaler.inverse_transform(
        synthetic_std
    )

    synthetic = np.clip(
        synthetic,
        0.0,
        1.0,
    )

    synthetic = synthetic.astype(
        np.float32
    )

    for feature in discrete_features:

        column_index = features.index(
            feature
        )

        synthetic[
            :,
            column_index
        ] = nearest_valid_value(
            synthetic[
                :,
                column_index
            ],
            valid_discrete_values[
                feature
            ],
        )

    return synthetic


# ============================================================
# 11. 统计评价函数
#
# 重点不是 GAN loss，
# 而是生成数据是否保留真实数据结构。
# ============================================================

real_corr = pd.DataFrame(
    X,
    columns=features,
).corr(
    method="spearman"
).to_numpy()

upper_triangle = np.triu_indices(
    len(features),
    k=1,
)


def compare_distribution(
    reference,
    candidate,
):

    ks_values = []

    wasserstein_values = []

    for i in range(
        len(features)
    ):

        ks_values.append(
            ks_2samp(
                reference[:, i],
                candidate[:, i],
            ).statistic
        )

        wasserstein_values.append(
            wasserstein_distance(
                reference[:, i],
                candidate[:, i],
            )
        )

    candidate_corr = pd.DataFrame(
        candidate,
        columns=features,
    ).corr(
        method="spearman"
    ).to_numpy()

    corr_mae = np.nanmean(
        np.abs(
            real_corr[
                upper_triangle
            ]
            -
            candidate_corr[
                upper_triangle
            ]
        )
    )

    return {
        "mean_ks":
            float(
                np.mean(
                    ks_values
                )
            ),

        "max_ks":
            float(
                np.max(
                    ks_values
                )
            ),

        "mean_wasserstein":
            float(
                np.mean(
                    wasserstein_values
                )
            ),

        "corr_mae":
            float(
                corr_mae
            ),
    }


# ============================================================
# 12. 模式覆盖 / memorization 检查
# ============================================================

nearest_real_model = NearestNeighbors(
    n_neighbors=1
)

nearest_real_model.fit(
    X_std
)


def evaluate_pool(
    synthetic,
):

    synthetic_std = standard_scaler.transform(
        synthetic
    )

    synthetic_cluster = kmeans.predict(
        synthetic_std
    )

    cluster_counts = np.bincount(
        synthetic_cluster,
        minlength=N_CLUSTERS,
    )

    distances, _ = nearest_real_model.kneighbors(
        synthetic_std
    )

    rounded_real = {
        tuple(row)
        for row in np.round(
            X,
            8,
        )
    }

    exact_duplicates = sum(
        tuple(row)
        in rounded_real

        for row in np.round(
            synthetic,
            8,
        )
    )

    metrics = compare_distribution(
        X,
        synthetic,
    )

    metrics.update({

        "cluster_coverage":
            int(
                np.sum(
                    cluster_counts > 0
                )
            ),

        "min_cluster_count":
            int(
                cluster_counts.min()
            ),

        "mean_nearest_real_distance":
            float(
                distances.mean()
            ),

        "min_nearest_real_distance":
            float(
                distances.min()
            ),

        "exact_duplicate_count":
            int(
                exact_duplicates
            ),
    })

    return (
        metrics,
        synthetic_cluster,
        cluster_counts,
    )


# ============================================================
# 13. 从 candidate pool 每个 cluster 随机选 12 条
#
# 不选“离中心最近的12条”，避免人为美化。
# ============================================================

def select_72(
    pool,
    pool_cluster,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    selected_indices = []

    for cluster in range(
        N_CLUSTERS
    ):

        candidates = np.where(
            pool_cluster
            ==
            cluster
        )[0]

        if len(candidates) < SYNTHETIC_PER_CLUSTER:

            raise RuntimeError(
                f"cluster {cluster + 1} "
                f"只有 {len(candidates)} 个候选，"
                f"不足 {SYNTHETIC_PER_CLUSTER} 个。"
            )

        chosen = rng.choice(
            candidates,
            size=SYNTHETIC_PER_CLUSTER,
            replace=False,
        )

        selected_indices.extend(
            chosen.tolist()
        )

    selected = pool[
        np.array(
            selected_indices
        )
    ]

    return selected


# ============================================================
# 14. GAN 训练
#
# 注意：
# - 无 K-means 条件输入
# - 无 balanced cluster sampler
# - 仍然是普通随机 real mini-batch
#
# 因此这一部分最接近经典 Vanilla GAN。
# ============================================================

def train_one_gan(
    config,
    train_seed,
    checkpoints,
):

    torch.manual_seed(
        train_seed
    )

    np.random.seed(
        train_seed
    )

    random.seed(
        train_seed
    )

    generator = Generator(
        latent_dim=config[
            "latent_dim"
        ],
        hidden_dims=config[
            "generator_hidden"
        ],
    )

    discriminator = Discriminator(
        hidden_dims=config[
            "discriminator_hidden"
        ],
        dropout=config[
            "dropout"
        ],
    )

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=config[
            "lr_g"
        ],
        betas=(
            0.5,
            0.999,
        ),
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=config[
            "lr_d"
        ],
        betas=(
            0.5,
            0.999,
        ),
    )

    criterion = nn.BCEWithLogitsLoss()

    real_tensor = torch.tensor(
        X_std,
        dtype=torch.float32,
    )

    batch_size = config[
        "batch_size"
    ]

    max_steps = max(
        checkpoints
    )

    checkpoint_results = []

    checkpoint_states = {}

    for step in range(
        1,
        max_steps + 1,
    ):

        # ----------------------------------------------------
        # D: 真实 batch
        # ----------------------------------------------------

        indices = torch.randint(
            low=0,
            high=len(
                real_tensor
            ),
            size=(
                batch_size,
            ),
        )

        real_batch = real_tensor[
            indices
        ]

        # instance noise 逐渐衰减
        noise_std = (
            config[
                "instance_noise"
            ]
            *
            (
                1
                -
                step
                /
                max_steps
            )
        )

        # ----------------------------------------------------
        # D: fake batch
        # ----------------------------------------------------

        z = torch.randn(
            batch_size,
            config[
                "latent_dim"
            ],
        )

        fake_batch = generator(
            z
        ).detach()

        noisy_real = (
            real_batch
            +
            noise_std
            *
            torch.randn_like(
                real_batch
            )
        )

        noisy_fake = (
            fake_batch
            +
            noise_std
            *
            torch.randn_like(
                fake_batch
            )
        )

        real_logits = discriminator(
            noisy_real
        )

        fake_logits = discriminator(
            noisy_fake
        )

        # real label smoothing
        real_target = torch.full_like(
            real_logits,
            config[
                "real_label"
            ],
        )

        fake_target = torch.zeros_like(
            fake_logits
        )

        d_loss_real = criterion(
            real_logits,
            real_target,
        )

        d_loss_fake = criterion(
            fake_logits,
            fake_target,
        )

        d_loss = (
            d_loss_real
            +
            d_loss_fake
        )

        optimizer_d.zero_grad()

        d_loss.backward()

        optimizer_d.step()

        # ----------------------------------------------------
        # G
        # ----------------------------------------------------

        z = torch.randn(
            batch_size,
            config[
                "latent_dim"
            ],
        )

        generated = generator(
            z
        )

        generated_logits = discriminator(
            generated
        )

        g_target = torch.ones_like(
            generated_logits
        )

        g_loss = criterion(
            generated_logits,
            g_target,
        )

        optimizer_g.zero_grad()

        g_loss.backward()

        optimizer_g.step()

        # ----------------------------------------------------
        # checkpoint evaluation
        # ----------------------------------------------------

        if step in checkpoints:

            generator.eval()

            with torch.no_grad():

                z_eval = torch.randn(
                    config[
                        "eval_pool_size"
                    ],
                    config[
                        "latent_dim"
                    ],
                )

                synthetic_std = generator(
                    z_eval
                ).cpu().numpy()

            generator.train()

            synthetic = postprocess_synthetic(
                synthetic_std
            )

            (
                pool_metrics,
                pool_cluster,
                pool_counts,
            ) = evaluate_pool(
                synthetic
            )

            row = {
                "train_seed":
                    train_seed,

                "step":
                    step,

                "g_loss":
                    float(
                        g_loss.item()
                    ),

                "d_loss":
                    float(
                        d_loss.item()
                    ),

                **pool_metrics,
            }

            # --------------------------------------------
            # 只有 6 cluster 全覆盖，
            # 且每类至少 12 个候选，
            # 才继续评价最终 72 条方案。
            # --------------------------------------------

            if (
                pool_metrics[
                    "cluster_coverage"
                ]
                ==
                N_CLUSTERS

                and

                pool_metrics[
                    "min_cluster_count"
                ]
                >=
                SYNTHETIC_PER_CLUSTER
            ):

                selection_metrics = []

                for selection_seed in [
                    11,
                    22,
                    33,
                    44,
                    55,
                ]:

                    selected = select_72(
                        synthetic,
                        pool_cluster,
                        selection_seed,
                    )

                    synthetic_metrics = compare_distribution(
                        X,
                        selected,
                    )

                    augmented = np.vstack(
                        [
                            X,
                            selected,
                        ]
                    )

                    augmented_metrics = compare_distribution(
                        X,
                        augmented,
                    )

                    selection_metrics.append({

                        "synthetic_mean_ks":
                            synthetic_metrics[
                                "mean_ks"
                            ],

                        "synthetic_corr_mae":
                            synthetic_metrics[
                                "corr_mae"
                            ],

                        "augmented_mean_ks":
                            augmented_metrics[
                                "mean_ks"
                            ],

                        "augmented_corr_mae":
                            augmented_metrics[
                                "corr_mae"
                            ],
                    })

                selection_df = pd.DataFrame(
                    selection_metrics
                )

                for column in selection_df.columns:

                    row[
                        f"{column}_mean"
                    ] = selection_df[
                        column
                    ].mean()

                    row[
                        f"{column}_std"
                    ] = selection_df[
                        column
                    ].std()

                row[
                    "selection_valid"
                ] = True

            else:

                row[
                    "selection_valid"
                ] = False

            checkpoint_results.append(
                row
            )

            checkpoint_states[
                step
            ] = copy.deepcopy(
                generator.state_dict()
            )

            print(
                f"seed={train_seed} "
                f"step={step} "
                f"KS={pool_metrics['mean_ks']:.3f} "
                f"Corr={pool_metrics['corr_mae']:.3f} "
                f"coverage="
                f"{pool_metrics['cluster_coverage']}/6 "
                f"min_cluster="
                f"{pool_metrics['min_cluster_count']}"
            )

    return (
        checkpoint_results,
        checkpoint_states,
    )


# ============================================================
# 15. 参数搜索空间
#
# 这是小型、可解释的搜索，
# 不是暴力扫几十种结构。
# ============================================================

if QUICK_MODE:

    configs = {

        "quick": {
            "latent_dim": 16,
            "generator_hidden": (
                64,
                128,
                64,
            ),
            "discriminator_hidden": (
                128,
                64,
            ),
            "batch_size": 64,
            "lr_g": 2e-4,
            "lr_d": 1e-4,
            "dropout": 0.1,
            "real_label": 0.9,
            "instance_noise": 0.03,
            "eval_pool_size": 800,
        },
    }

    train_seeds = [
        42,
    ]

    checkpoints = [
        300,
        600,
    ]

else:

    configs = {

        # --------------------------------------------
        # v1：基准稳定版
        # --------------------------------------------
        "v1": {
            "latent_dim": 16,
            "generator_hidden": (
                64,
                128,
                64,
            ),
            "discriminator_hidden": (
                128,
                64,
            ),
            "batch_size": 64,
            "lr_g": 2e-4,
            "lr_d": 1e-4,
            "dropout": 0.1,
            "real_label": 0.9,
            "instance_noise": 0.05,
            "eval_pool_size": 2000,
        },

        # --------------------------------------------
        # v2：稍弱 Discriminator
        #
        # 我自己的实测中这一组整体最好。
        # --------------------------------------------
        "v2": {
            "latent_dim": 16,
            "generator_hidden": (
                64,
                128,
                64,
            ),
            "discriminator_hidden": (
                128,
                64,
            ),
            "batch_size": 64,
            "lr_g": 2e-4,
            "lr_d": 5e-5,
            "dropout": 0.1,
            "real_label": 0.9,
            "instance_noise": 0.05,
            "eval_pool_size": 2000,
        },

        # --------------------------------------------
        # v3：更大 latent space
        # --------------------------------------------
        "v3": {
            "latent_dim": 32,
            "generator_hidden": (
                64,
                128,
                64,
            ),
            "discriminator_hidden": (
                128,
                64,
            ),
            "batch_size": 64,
            "lr_g": 1e-4,
            "lr_d": 1e-4,
            "dropout": 0.1,
            "real_label": 0.9,
            "instance_noise": 0.03,
            "eval_pool_size": 2000,
        },
    }

    train_seeds = [
        42,
        2026,
    ]

    checkpoints = [
        600,
        900,
        1200,
        1400,
    ]


# ============================================================
# 16. 运行搜索
# ============================================================

all_results = []

all_states = {}

torch.set_num_threads(
    max(
        1,
        min(
            4,
            os.cpu_count()
            or
            1
        )
    )
)

for config_name, config in configs.items():

    print()
    print("=" * 80)
    print(
        "TRAIN CONFIG:",
        config_name,
    )
    print("=" * 80)

    for seed in train_seeds:

        (
            rows,
            states,
        ) = train_one_gan(
            config=config,
            train_seed=seed,
            checkpoints=checkpoints,
        )

        for row in rows:

            row[
                "config"
            ] = config_name

            all_results.append(
                row
            )

        for step, state in states.items():

            all_states[
                (
                    config_name,
                    seed,
                    step,
                )
            ] = state


results_df = pd.DataFrame(
    all_results
)

results_df.to_excel(
    results_dir
    / "05_gan_tuning_all_runs.xlsx",
    index=False,
)


# ============================================================
# 17. 科学选择最优配置
#
# 不选择“某一个幸运 seed”。
#
# 而是：
# 对 config + checkpoint
# 在多个 train seeds 上取平均。
#
# 要求：
# 所有 seeds 都必须 selection_valid=True。
# ============================================================

valid_df = results_df[
    results_df[
        "selection_valid"
    ]
    ==
    True
].copy()

if valid_df.empty:

    raise RuntimeError(
        "没有任何 GAN checkpoint "
        "能够自然覆盖 6 个 cluster，"
        "请扩大训练或调整参数。"
    )


group_columns = [
    "config",
    "step",
]

metrics_for_group = [
    "synthetic_mean_ks_mean",
    "synthetic_corr_mae_mean",
    "augmented_mean_ks_mean",
    "augmented_corr_mae_mean",
]

grouped = (
    valid_df
    .groupby(
        group_columns
    )
    .agg(
        {
            **{
                metric: "mean"
                for metric
                in metrics_for_group
            },
            "train_seed":
                "nunique",
        }
    )
    .reset_index()
)


# 正式模式要求两个 training seeds 都通过。
if not QUICK_MODE:

    grouped = grouped[
        grouped[
            "train_seed"
        ]
        ==
        len(
            train_seeds
        )
    ].copy()


if grouped.empty:

    raise RuntimeError(
        "没有 config + checkpoint "
        "能在所有 training seeds 上稳定通过。"
    )


# ============================================================
# 18. 综合分数
#
# 不使用论文图像相似度作为目标。
#
# 分数只来自：
# - synthetic-only distribution
# - synthetic-only correlation
# - augmented distribution preservation
# - augmented correlation preservation
#
# 越小越好。
# ============================================================

grouped[
    "scientific_score"
] = (

    grouped[
        "synthetic_mean_ks_mean"
    ]

    +

    grouped[
        "synthetic_corr_mae_mean"
    ]

    +

    grouped[
        "augmented_mean_ks_mean"
    ]

    +

    grouped[
        "augmented_corr_mae_mean"
    ]

)


grouped = grouped.sort_values(
    "scientific_score"
).reset_index(
    drop=True
)


grouped.to_excel(
    results_dir
    / "05_gan_tuning_summary.xlsx",
    index=False,
)


best_row = grouped.iloc[
    0
]

best_config_name = best_row[
    "config"
]

best_step = int(
    best_row[
        "step"
    ]
)

best_config = configs[
    best_config_name
]


print()
print("=" * 80)
print("BEST GAN SETTING")
print("=" * 80)
print(
    "Config:",
    best_config_name,
)
print(
    "Step:",
    best_step,
)
print(
    "Scientific score:",
    best_row[
        "scientific_score"
    ],
)
print(
    best_row
)


# ============================================================
# 19. 用预先固定的 FINAL_TRAIN_SEED 重新训练最终模型
#
# 不使用“表现最好看的 seed”。
# ============================================================

final_rows, final_states = train_one_gan(
    config=best_config,
    train_seed=FINAL_TRAIN_SEED,
    checkpoints=[
        best_step,
    ],
)

final_generator = Generator(
    latent_dim=best_config[
        "latent_dim"
    ],
    hidden_dims=best_config[
        "generator_hidden"
    ],
)

final_generator.load_state_dict(
    final_states[
        best_step
    ]
)

final_generator.eval()


# ============================================================
# 20. 生成大型 candidate pool
# ============================================================

FINAL_POOL_SIZE = (
    2000
    if QUICK_MODE
    else
    10000
)

torch.manual_seed(
    999
)

with torch.no_grad():

    final_noise = torch.randn(
        FINAL_POOL_SIZE,
        best_config[
            "latent_dim"
        ],
    )

    final_pool_std = final_generator(
        final_noise
    ).cpu().numpy()


final_pool = postprocess_synthetic(
    final_pool_std
)


(
    final_pool_metrics,
    final_pool_cluster,
    final_pool_counts,
) = evaluate_pool(
    final_pool
)


print()
print("=" * 80)
print("FINAL CANDIDATE POOL")
print("=" * 80)
print(
    "Pool cluster counts:",
    final_pool_counts,
)
print(
    "Pool metrics:",
    final_pool_metrics,
)


if (
    final_pool_counts
    <
    SYNTHETIC_PER_CLUSTER
).any():

    raise RuntimeError(
        "最终 GAN candidate pool "
        "仍有 cluster 不足12条，不能继续。"
    )


# ============================================================
# 21. 每个 cluster 随机抽12条
# ============================================================

synthetic_72 = select_72(
    final_pool,
    final_pool_cluster,
    FINAL_SELECTION_SEED,
)


synthetic_72_cluster = kmeans.predict(
    standard_scaler.transform(
        synthetic_72
    )
)


synthetic_72_counts = np.bincount(
    synthetic_72_cluster,
    minlength=N_CLUSTERS,
)


# ============================================================
# 22. 最终增强数据
# ============================================================

augmented = np.vstack(
    [
        X,
        synthetic_72,
    ]
)


synthetic_metrics = compare_distribution(
    X,
    synthetic_72,
)


augmented_metrics = compare_distribution(
    X,
    augmented,
)


# ============================================================
# 23. Nearest-neighbor / duplicates
# ============================================================

synthetic_std_for_nn = standard_scaler.transform(
    synthetic_72
)

distances, _ = nearest_real_model.kneighbors(
    synthetic_std_for_nn
)

rounded_real = {
    tuple(row)
    for row in np.round(
        X,
        8,
    )
}

exact_duplicate_count = sum(
    tuple(row)
    in rounded_real

    for row in np.round(
        synthetic_72,
        8,
    )
)


# ============================================================
# 24. 保存 synthetic 72
# ============================================================

synthetic_df = pd.DataFrame(
    synthetic_72,
    columns=features,
)

synthetic_df[
    "source"
] = "synthetic_GAN"

synthetic_df[
    "cluster"
] = synthetic_72_cluster + 1


synthetic_df.to_excel(
    results_dir
    / "05_synthetic_72.xlsx",
    index=False,
)


# ============================================================
# 25. 保存 augmented data
# ============================================================

real_out = X_df.copy()

real_out[
    "source"
] = "real"

real_out[
    "cluster"
] = real_cluster + 1


augmented_out = pd.concat(
    [
        real_out[
            features
            +
            [
                "source",
                "cluster",
            ]
        ],
        synthetic_df[
            features
            +
            [
                "source",
                "cluster",
            ]
        ],
    ],
    ignore_index=True,
)


augmented_out.to_excel(
    results_dir
    / "05_augmented_dataset.xlsx",
    index=False,
)


# ============================================================
# 26. 保存模型
# ============================================================

torch.save(
    {
        "generator_state_dict":
            final_generator.state_dict(),

        "config":
            best_config,

        "step":
            best_step,

        "train_seed":
            FINAL_TRAIN_SEED,

        "feature_order":
            features,
    },
    results_dir
    / "05_generator.pt",
)


# ============================================================
# 27. 最终指标
# ============================================================

final_metrics = pd.DataFrame({

    "metric": [
        "real_samples",
        "synthetic_samples",
        "augmented_samples",
        "kmeans_silhouette",
        "synthetic_mean_ks",
        "synthetic_max_ks",
        "synthetic_corr_mae",
        "augmented_mean_ks",
        "augmented_max_ks",
        "augmented_corr_mae",
        "mean_nearest_real_distance",
        "min_nearest_real_distance",
        "exact_duplicate_count",
    ],

    "value": [
        len(X),
        len(synthetic_72),
        len(augmented),
        silhouette,
        synthetic_metrics[
            "mean_ks"
        ],
        synthetic_metrics[
            "max_ks"
        ],
        synthetic_metrics[
            "corr_mae"
        ],
        augmented_metrics[
            "mean_ks"
        ],
        augmented_metrics[
            "max_ks"
        ],
        augmented_metrics[
            "corr_mae"
        ],
        distances.mean(),
        distances.min(),
        exact_duplicate_count,
    ],
})


final_metrics.to_excel(
    results_dir
    / "05_final_validation_metrics.xlsx",
    index=False,
)


# ============================================================
# 28. Fig.2(b)
#
# Original vs Synthetic-only
#
# 使用同一个 StandardScaler。
# ============================================================

real_plot_std = standard_scaler.transform(
    X
)

synthetic_plot_std = standard_scaler.transform(
    synthetic_72
)


fig, ax = plt.subplots(
    figsize=(
        8,
        5,
    )
)


ax.hist(
    real_plot_std.ravel(),
    bins=35,
    density=True,
    alpha=0.55,
    label="Original",
)


ax.hist(
    synthetic_plot_std.ravel(),
    bins=35,
    density=True,
    alpha=0.55,
    label="Synthetic",
)


ax.set_xlabel(
    "Value"
)

ax.set_ylabel(
    "Probability"
)

ax.set_title(
    "Original vs Synthetic Data"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    results_dir
    / "05_fig2b_histogram.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# 29. Fig.3(a)
#
# 左：Original
# 右：After augmentation
#
# 每个 cluster +12。
# ============================================================

augmented_cluster_counts = (
    real_cluster_counts
    +
    synthetic_72_counts
)

class_names = [
    f"class{i + 1}"
    for i in range(
        N_CLUSTERS
    )
]


fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        11,
        4,
    ),
)


bars_left = axes[
    0
].bar(
    class_names,
    real_cluster_counts,
)


bars_right = axes[
    1
].bar(
    class_names,
    augmented_cluster_counts,
)


axes[
    0
].set_title(
    "Number of original samples"
)

axes[
    1
].set_title(
    "Number of samples after augmentation"
)


for axis, bars, counts in [
    (
        axes[0],
        bars_left,
        real_cluster_counts,
    ),
    (
        axes[1],
        bars_right,
        augmented_cluster_counts,
    ),
]:

    axis.set_ylabel(
        "num"
    )

    axis.tick_params(
        axis="x",
        rotation=35,
    )

    for bar, value in zip(
        bars,
        counts,
    ):

        axis.text(
            bar.get_x()
            +
            bar.get_width()
            /
            2,
            bar.get_height(),
            str(
                int(
                    value
                )
            ),
            ha="center",
            va="bottom",
        )


fig.tight_layout()

fig.savefig(
    results_dir
    / "05_fig3a_kmeans.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# 30. Fig.3(b) t-SNE
#
# 论文左右图坐标尺度差异很大，
# 很可能 Original / Synthetic 分别 fit t-SNE。
#
# 这里只把 t-SNE 当定性图，不参与 GAN 调参。
# ============================================================

real_tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=RANDOM_SEED,
    init="pca",
    learning_rate="auto",
).fit_transform(
    real_plot_std
)


synthetic_tsne = TSNE(
    n_components=2,
    perplexity=min(
        20,
        len(
            synthetic_plot_std
        )
        -
        1,
    ),
    random_state=RANDOM_SEED,
    init="pca",
    learning_rate="auto",
).fit_transform(
    synthetic_plot_std
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        11,
        5,
    ),
)


for cluster in range(
    N_CLUSTERS
):

    real_mask = (
        real_cluster
        ==
        cluster
    )

    synthetic_mask = (
        synthetic_72_cluster
        ==
        cluster
    )

    axes[
        0
    ].scatter(
        real_tsne[
            real_mask,
            0
        ],
        real_tsne[
            real_mask,
            1
        ],
        s=18,
        label=f"class{cluster + 1}",
    )

    axes[
        1
    ].scatter(
        synthetic_tsne[
            synthetic_mask,
            0
        ],
        synthetic_tsne[
            synthetic_mask,
            1
        ],
        s=24,
        label=f"class{cluster + 1}",
    )


axes[
    0
].set_title(
    "Original Data"
)

axes[
    1
].set_title(
    "Synthetic Data"
)


for axis in axes:

    axis.set_xlabel(
        "tsne dimension1"
    )

    axis.set_ylabel(
        "tsne dimension2"
    )

    axis.legend(
        fontsize=7
    )


fig.tight_layout()

fig.savefig(
    results_dir
    / "05_fig3b_tsne.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# 31. Fig.3(c) Boxplot
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        13,
        5,
    ),
    sharey=True,
)


axes[
    0
].boxplot(
    [
        real_plot_std[
            :,
            i
        ]
        for i in range(
            len(
                features
            )
        )
    ],
    tick_labels=[
        f"feature{i + 1}"
        for i in range(
            len(
                features
            )
        )
    ],
)


axes[
    1
].boxplot(
    [
        synthetic_plot_std[
            :,
            i
        ]
        for i in range(
            len(
                features
            )
        )
    ],
    tick_labels=[
        f"feature{i + 1}"
        for i in range(
            len(
                features
            )
        )
    ],
)


axes[
    0
].set_title(
    "Original Data"
)

axes[
    1
].set_title(
    "Synthetic Data"
)


for axis in axes:

    axis.set_ylabel(
        "Value"
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )


fig.tight_layout()

fig.savefig(
    results_dir
    / "05_fig3c_boxplot.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# 32. 完成
# ============================================================

print()
print("=" * 80)
print("FINAL GAN VALIDATION")
print("=" * 80)

print(
    "Best config:",
    best_config_name,
)

print(
    "Best step:",
    best_step,
)

print(
    "Real cluster counts:",
    real_cluster_counts,
)

print(
    "Synthetic selected counts:",
    synthetic_72_counts,
)

print(
    "Synthetic-only metrics:",
    synthetic_metrics,
)

print(
    "Augmented metrics:",
    augmented_metrics,
)

print(
    "Exact duplicate count:",
    exact_duplicate_count,
)

print(
    "Mean nearest-real distance:",
    float(
        distances.mean()
    ),
)

print()
print(
    "Saved:"
)

print(
    results_dir
    / "05_synthetic_72.xlsx"
)

print(
    results_dir
    / "05_augmented_dataset.xlsx"
)

print(
    results_dir
    / "05_gan_tuning_all_runs.xlsx"
)

print(
    results_dir
    / "05_gan_tuning_summary.xlsx"
)

print(
    results_dir
    / "05_final_validation_metrics.xlsx"
)

print(
    results_dir
    / "05_fig2b_histogram.png"
)

print(
    results_dir
    / "05_fig3a_kmeans.png"
)

print(
    results_dir
    / "05_fig3b_tsne.png"
)

print(
    results_dir
    / "05_fig3c_boxplot.png"
)