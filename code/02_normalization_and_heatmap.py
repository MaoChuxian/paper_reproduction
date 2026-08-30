# ============================================================
# 02_normalization_and_heatmap.py
#
# Min-Max normalization
# + Fig. 2(a) correlation heatmap baseline
#
# 当前版本：
# 不删除任何异常值
# 用全部 288 条 S1 + S2 数据建立 baseline
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
# 3. 读取 S1 + S2 数据
# ============================================================

df = pd.read_excel(
    data_path,
    sheet_name="S1_S2_Combined"
)


print(
    "原始数据形状：",
    df.shape
)


# ============================================================
# 4. 论文使用的 9 个数值变量
#
# 其中前 8 个是模型输入 X
# Lifetime 是预测目标 y
#
# 为了复现 Fig. 2(a)，Lifetime 也参与相关性分析
# ============================================================

features = [

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
# 5. 取出数值数据
# ============================================================

data = df[
    features
].copy()


# ============================================================
# 6. Min-Max normalization
# ============================================================

scaler = MinMaxScaler()


normalized_array = scaler.fit_transform(
    data
)


normalized_df = pd.DataFrame(

    normalized_array,

    columns=features

)


# ============================================================
# 7. 查看归一化后的范围
# ============================================================

print()

print(
    "归一化后最小值："
)

print(
    normalized_df.min()
)


print()

print(
    "归一化后最大值："
)

print(
    normalized_df.max()
)


# ============================================================
# 8. 计算 Pearson correlation matrix
# ============================================================

corr_matrix = normalized_df.corr(
    method="pearson"
)


print()

print(
    "=" * 70
)

print(
    "当前数据相关系数矩阵"
)

print(
    "=" * 70
)

print(
    corr_matrix.round(3)
)


# ============================================================
# 9. 保存归一化数据
# ============================================================

normalized_output_path = (

    results_dir
    / "02_normalized_baseline.xlsx"

)


normalized_df.to_excel(

    normalized_output_path,

    index=False

)


# ============================================================
# 10. 保存相关系数矩阵
# ============================================================

corr_output_path = (

    results_dir
    / "02_correlation_baseline.xlsx"

)


corr_matrix.to_excel(
    corr_output_path
)


# ============================================================
# 11. 绘制热图
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)


image = ax.imshow(

    corr_matrix.values,

    vmin=-1,
    vmax=1,

    cmap="bwr"

)


# ============================================================
# 12. 坐标轴
# ============================================================

ax.set_xticks(
    np.arange(
        len(features)
    )
)

ax.set_yticks(
    np.arange(
        len(features)
    )
)


ax.set_xticklabels(
    features
)

ax.set_yticklabels(
    features
)


plt.setp(

    ax.get_xticklabels(),

    rotation=45,

    ha="right",

    rotation_mode="anchor"

)


# ============================================================
# 13. 在每个格子中写相关系数
# ============================================================

for i in range(
    len(features)
):

    for j in range(
        len(features)
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
# 14. Colorbar
# ============================================================

colorbar = fig.colorbar(

    image,

    ax=ax

)


colorbar.set_label(
    "Correlation"
)


# ============================================================
# 15. 标题
# ============================================================

ax.set_title(

    "Correlation Matrix of Normalized Data\n"
    "(Baseline: No Outlier Removal)"

)


# ============================================================
# 16. 自动调整布局
# ============================================================

fig.tight_layout()


# ============================================================
# 17. 保存图片
# ============================================================

figure_path = (

    results_dir
    / "02_fig2a_baseline.png"

)


fig.savefig(

    figure_path,

    dpi=300,

    bbox_inches="tight"

)


plt.close(
    fig
)


# ============================================================
# 18. 输出
# ============================================================

print()

print(
    "归一化数据：",
    normalized_output_path
)

print(
    "相关矩阵：",
    corr_output_path
)

print(
    "热图：",
    figure_path
)