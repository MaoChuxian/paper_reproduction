# ============================================================
# 08_split_dataset_recovered.py
# GAN 增强数据 -> Train / Validation / Test = 8 : 1 : 1
#
# 恢复自之前的 08 阶段：
# - 输入：results/07_augmented_softbalanced.xlsx
# - random_state = 42
# - 第一次：80% Train，20% Temp
# - 第二次：Temp 对半分 -> 10% Validation + 10% Test
# - 348 条数据 -> 278 / 35 / 35
#
# source、paper_aligned_cluster 只用于追踪和审计，
# 后续机器学习绝不能把它们作为输入特征。
# ============================================================

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. 路径
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (
    SCRIPT_DIR.parent
    if SCRIPT_DIR.name.lower() == "code"
    else SCRIPT_DIR
)

RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_PATH = RESULTS_DIR / "07_augmented_softbalanced.xlsx"

TRAIN_PATH = RESULTS_DIR / "08_train.xlsx"
VAL_PATH = RESULTS_DIR / "08_validation.xlsx"
TEST_PATH = RESULTS_DIR / "08_test.xlsx"
SUMMARY_PATH = RESULTS_DIR / "08_split_summary.xlsx"


# ============================================================
# 2. 固定随机种子
# ============================================================

RANDOM_STATE = 42


# ============================================================
# 3. 读取 07 增强数据
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"没有找到：{INPUT_PATH}\n"
        "请确认 07_augmented_softbalanced.xlsx 仍在 results/ 中。"
    )

df = pd.read_excel(INPUT_PATH)

print("=" * 70)
print("08 Split Dataset")
print("=" * 70)
print("Input :", INPUT_PATH)
print("Rows  :", len(df))
print()


# ============================================================
# 4. 第一次划分：80% Train + 20% Temp
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    shuffle=True,
)


# ============================================================
# 5. 第二次划分：Temp 对半
#    -> 10% Validation + 10% Test
# ============================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    shuffle=True,
)


# ============================================================
# 6. 重置索引
# ============================================================

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# 7. 保存三个数据集
# ============================================================

train_df.to_excel(
    TRAIN_PATH,
    index=False,
)

val_df.to_excel(
    VAL_PATH,
    index=False,
)

test_df.to_excel(
    TEST_PATH,
    index=False,
)


# ============================================================
# 8. 输出基本信息
# ============================================================

print("Split result")
print("-" * 70)
print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")
print(f"Total      : {len(train_df) + len(val_df) + len(test_df)}")
print()


# ============================================================
# 9. 审计 source 分布
# ============================================================

summary_rows = []

for split_name, split_df in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df),
]:
    row = {
        "Split": split_name,
        "N": len(split_df),
    }

    if "source" in split_df.columns:
        source_counts = split_df["source"].value_counts()

        for source_name, count in source_counts.items():
            row[f"source::{source_name}"] = int(count)

    summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)


# ============================================================
# 10. 审计聚类分布
# ============================================================

# 之前的 07 数据优先使用 paper_aligned_cluster。
# 如果没有，则退回 cluster。
if "paper_aligned_cluster" in df.columns:
    cluster_col = "paper_aligned_cluster"
elif "cluster" in df.columns:
    cluster_col = "cluster"
else:
    cluster_col = None


if cluster_col is not None:

    cluster_summary = []

    for split_name, split_df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        counts = (
            split_df[cluster_col]
            .value_counts()
            .sort_index()
        )

        for cluster_id, count in counts.items():
            cluster_summary.append({
                "Split": split_name,
                "ClusterColumn": cluster_col,
                "Cluster": cluster_id,
                "N": int(count),
            })

    cluster_summary_df = pd.DataFrame(cluster_summary)

else:
    cluster_summary_df = pd.DataFrame(
        columns=[
            "Split",
            "ClusterColumn",
            "Cluster",
            "N",
        ]
    )


# ============================================================
# 11. 保存划分摘要
# ============================================================

with pd.ExcelWriter(SUMMARY_PATH) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="split_summary",
        index=False,
    )

    cluster_summary_df.to_excel(
        writer,
        sheet_name="cluster_summary",
        index=False,
    )


# ============================================================
# 12. 完成
# ============================================================

print("Saved:")
print(" ", TRAIN_PATH)
print(" ", VAL_PATH)
print(" ", TEST_PATH)
print(" ", SUMMARY_PATH)
print()

print(
    "IMPORTANT:\n"
    "source 和 paper_aligned_cluster / cluster 仅用于追踪，\n"
    "后续模型输入只使用真实 8 个物理特征。"
)
