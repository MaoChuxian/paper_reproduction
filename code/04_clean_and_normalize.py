# ============================================================
# 04_clean_and_normalize.py
#
# Candidate reproduction of data preprocessing
#
# 当前候选方法：
#
# 1. 读取 S1 + S2 原始实验数据
# 2. 使用 1% - 99% percentile trimming
#    删除 extreme-outlier candidates
# 3. Min-Max normalization
# 4. Spearman correlation
# 5. 生成 Fig. 2(a) 候选复现图
#
# 注意：
# 原论文仅说明：
# "removing extreme outliers"
# 但没有公开具体异常值判定方法。
#
# 因此 percentile trimming 是本复现采用的候选方法，
# 不是作者明确公开的方法。
# ============================================================


# ============================================================
# 1. 导入库
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 2. 路径
# ============================================================

data_path = Path(
    "results/00_raw_data.xlsx"
)

results_dir = Path(
    "results"
)

results_dir.mkdir(
    exist_ok=True
)


# ============================================================
# 3. 读取 S1 + S2
# ============================================================

df = pd.read_excel(
    data_path,
    sheet_name="S1_S2_Combined"
)


print(
    "清洗前样本数：",
    len(df)
)


# ============================================================
# 4. 数值变量
#
# 前 8 个为模型输入特征
# Lifetime 为预测目标
#
# Fig.2(a) 与 GAN 数据增强均涉及完整数据关系，
# 因此当前预处理阶段将 9 个数值变量一起考虑。
# ============================================================

numeric_columns = [

    "a",
    "b",
    "c",
    "V",
    "concentration",
    "Layer",
    "valence_electron",
    "IQE",
    "Lifetime"

]


# ============================================================
# 5. 计算 1% 与 99% 分位数
# ============================================================

lower_limits = (

    df[numeric_columns]
    .quantile(0.01)

)


upper_limits = (

    df[numeric_columns]
    .quantile(0.99)

)


print()

print(
    "=" * 70
)

print(
    "1% 分位数"
)

print(
    "=" * 70
)

print(
    lower_limits
)


print()

print(
    "=" * 70
)

print(
    "99% 分位数"
)

print(
    "=" * 70
)

print(
    upper_limits
)


# ============================================================
# 6. 标记 extreme-outlier candidates
#
# 对于某一条样本：
#
# 只要任意一个数值变量：
#
# x < P1
#
# 或
#
# x > P99
#
# 就标记为候选 extreme outlier。
# ============================================================

within_lower = (

    df[numeric_columns]
    >= lower_limits

)


within_upper = (

    df[numeric_columns]
    <= upper_limits

)


normal_mask = (

    within_lower
    &
    within_upper

).all(
    axis=1
)


# ============================================================
# 7. 分成保留数据和删除候选数据
# ============================================================

clean_df = (

    df[
        normal_mask
    ]
    .copy()
)


removed_df = (

    df[
        ~normal_mask
    ]
    .copy()
)


print()

print(
    "删除候选样本数：",
    len(removed_df)
)


print(
    "清洗后样本数：",
    len(clean_df)
)


# ============================================================
# 8. 查看被删除的数据
# ============================================================

print()

print(
    "=" * 70
)

print(
    "Extreme-outlier candidates"
)

print(
    "=" * 70
)


print(

    removed_df[
        [
            "Name",
            "a",
            "b",
            "c",
            "V",
            "concentration",
            "Layer",
            "valence_electron",
            "IQE",
            "Lifetime",
            "dopant",
            "Ref"
        ]
    ]

)


# ============================================================
# 9. 保存清洗结果
# ============================================================

clean_output_path = (

    results_dir
    / "04_clean_data.xlsx"

)


removed_output_path = (

    results_dir
    / "04_removed_extreme_candidates.xlsx"

)


clean_df.to_excel(

    clean_output_path,

    index=False

)


removed_df.to_excel(

    removed_output_path,

    index=False

)


# ============================================================
# 10. 保存分位数阈值
# ============================================================

threshold_df = pd.DataFrame({

    "P01": lower_limits,

    "P99": upper_limits

})


threshold_output_path = (

    results_dir
    / "04_percentile_thresholds.xlsx"

)


threshold_df.to_excel(
    threshold_output_path
)


# ============================================================
# 11. 按论文 Fig.2(a) 顺序排列变量
# ============================================================

paper_order = [

    "b",
    "c",
    "V",
    "concentration",
    "Layer",
    "valence_electron",
    "IQE",
    "Lifetime",
    "a"

]


# ============================================================
# 12. Min-Max normalization
#
# x_norm = (x - x_min) / (x_max - x_min)
#
# 每一个变量分别缩放到 [0, 1]
# ============================================================

scaler = MinMaxScaler()


normalized_array = scaler.fit_transform(

    clean_df[
        paper_order
    ]

)


normalized_df = pd.DataFrame(

    normalized_array,

    columns=paper_order,

    index=clean_df.index

)


# ============================================================
# 13. 把材料信息重新加回来
#
# 注意：
# Name、dopant、Ref 不参与 normalization。
# 它们只是样本身份信息。
# ============================================================

normalized_full_df = pd.concat(

    [

        clean_df[
            [
                "Name",
                "dopant",
                "Ref"
            ]
        ],

        normalized_df

    ],

    axis=1

)


# ============================================================
# 14. 检查 Min-Max 是否正确
# ============================================================

print()

print(
    "=" * 70
)

print(
    "归一化后最小值"
)

print(
    "=" * 70
)


print(
    normalized_df.min()
)


print()

print(
    "=" * 70
)

print(
    "归一化后最大值"
)

print(
    "=" * 70
)


print(
    normalized_df.max()
)


# ============================================================
# 15. 保存 normalized data
# ============================================================

normalized_output_path = (

    results_dir
    / "04_normalized_data.xlsx"

)


normalized_full_df.to_excel(

    normalized_output_path,

    index=False

)


# ============================================================
# 16. 保存 Min-Max 参数
#
# 后面如果需要把 normalized data
# 还原回实际物理单位，
# 必须知道每个变量的 min 和 max。
# ============================================================

scaler_parameters = pd.DataFrame({

    "feature": paper_order,

    "min": scaler.data_min_,

    "max": scaler.data_max_

})


scaler_output_path = (

    results_dir
    / "04_minmax_parameters.xlsx"

)


scaler_parameters.to_excel(

    scaler_output_path,

    index=False

)


# ============================================================
# 17. Spearman correlation
#
# 注意：
# 原文没有明确说明相关系数类型。
#
# 当前使用 Spearman 是根据论文 Fig.2(a)
# 公布的相关系数反推得到的候选复现方案。
# ============================================================

corr_matrix = normalized_df.corr(
    method="spearman"
)


print()

print(
    "=" * 70
)

print(
    "Spearman correlation matrix"
)

print(
    "=" * 70
)


print(
    corr_matrix.round(3)
)


# ============================================================
# 18. 保存 correlation matrix
# ============================================================

corr_output_path = (

    results_dir
    / "04_spearman_correlation.xlsx"

)


corr_matrix.to_excel(
    corr_output_path
)


# ============================================================
# 19. 绘制 Fig.2(a) 候选复现图
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)


image = ax.imshow(

    corr_matrix.values,

    cmap="bwr",

    vmin=-1,

    vmax=1

)


# ============================================================
# 20. 坐标轴
# ============================================================

ax.set_xticks(

    np.arange(
        len(paper_order)
    )

)


ax.set_yticks(

    np.arange(
        len(paper_order)
    )

)


ax.set_xticklabels(
    paper_order
)


ax.set_yticklabels(
    paper_order
)


plt.setp(

    ax.get_xticklabels(),

    rotation=45,

    ha="right",

    rotation_mode="anchor"

)


# ============================================================
# 21. 在每个格子写入相关系数
# ============================================================

for i in range(
    len(paper_order)
):

    for j in range(
        len(paper_order)
    ):

        value = corr_matrix.iloc[
            i,
            j
        ]


        ax.text(

            j,
            i,

            f"{value:.2f}",

            ha="center",

            va="center",

            fontsize=8

        )


# ============================================================
# 22. Colorbar
# ============================================================

colorbar = fig.colorbar(

    image,

    ax=ax

)


colorbar.set_label(
    "Spearman correlation"
)


# ============================================================
# 23. 标题
# ============================================================

ax.set_title(

    "Heatmap of Correlation Matrix for Normalized Data"

)


fig.tight_layout()


# ============================================================
# 24. 保存 Fig.2(a)
# ============================================================

figure_output_path = (

    results_dir
    / "04_fig2a_reproduction.png"

)


fig.savefig(

    figure_output_path,

    dpi=300,

    bbox_inches="tight"

)


plt.close(
    fig
)


# ============================================================
# 25. 最终输出
# ============================================================

print()

print(
    "=" * 70
)

print(
    "PREPROCESSING COMPLETED"
)

print(
    "=" * 70
)


print(
    "清洗后数据：",
    clean_output_path
)


print(
    "删除候选数据：",
    removed_output_path
)


print(
    "分位数阈值：",
    threshold_output_path
)


print(
    "归一化数据：",
    normalized_output_path
)


print(
    "Min-Max 参数：",
    scaler_output_path
)


print(
    "相关矩阵：",
    corr_output_path
)


print(
    "Fig.2(a)：",
    figure_output_path
)