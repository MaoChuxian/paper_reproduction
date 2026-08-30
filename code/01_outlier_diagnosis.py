# ============================================================
# 01_outlier_diagnosis.py
# 原始数据异常值初步诊断
# ============================================================

from pathlib import Path

import pandas as pd


# ============================================================
# 1. 路径
# ============================================================

data_path = Path(
    "results/00_raw_data.xlsx"
)


# ============================================================
# 2. 读取 S1 + S2 原始数据
# ============================================================

df = pd.read_excel(
    data_path,
    sheet_name="S1_S2_Combined"
)


print(
    "数据形状：",
    df.shape
)


# ============================================================
# 3. 需要检查的数值变量
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
    "Lifetime"
]


# ============================================================
# 4. 查看基本统计量
# ============================================================

summary = (
    df[features]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99
        ]
    )
    .T
)


print()

print(
    "=" * 70
)

print(
    "各变量统计范围"
)

print(
    "=" * 70
)

print(
    summary
)


# ============================================================
# 5. 分别查看每个变量最大和最小的样本
# ============================================================

for feature in features:

    print()
    print(
        "=" * 70
    )

    print(
        "Feature:",
        feature
    )

    print(
        "=" * 70
    )


    print()

    print(
        "最小的 5 条："
    )

    print(

        df[
            [
                "Name",
                feature,
                "dopant",
                "Ref"
            ]
        ]
        .sort_values(
            feature
        )
        .head(5)

    )


    print()

    print(
        "最大的 5 条："
    )

    print(

        df[
            [
                "Name",
                feature,
                "dopant",
                "Ref"
            ]
        ]
        .sort_values(
            feature,
            ascending=False
        )
        .head(5)

    )


# ============================================================
# 6. 保存统计结果
# ============================================================

output_path = Path(
    "results/01_data_summary.xlsx"
)


summary.to_excel(
    output_path
)


print()

print(
    "统计结果已保存：",
    output_path
)