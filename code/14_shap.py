from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"

train_path = results_dir / "08_train_final.xlsx"
model_path = results_dir / "11_1_RF" / "11_1_RF.joblib"

output_dir = results_dir / "14_RF_SHAP"
output_dir.mkdir(exist_ok=True)

features = [
    "a", "b", "c", "V",
    "concentration",
    "Layer",
    "valence_electron",
    "IQE",
]


train_df = pd.read_excel(train_path)
X = train_df[features]

model = joblib.load(model_path)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

importance = np.abs(shap_values.values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": features,
    "Mean_abs_SHAP": importance,
}).sort_values("Mean_abs_SHAP", ascending=False)

print()
print("RF SHAP feature importance")
print(importance_df.to_string(index=False))

importance_df.to_excel(
    output_dir / "14_RF_SHAP_importance.xlsx",
    index=False,
)

shap_df = pd.DataFrame(
    shap_values.values,
    columns=features,
)

shap_df.to_excel(
    output_dir / "14_RF_SHAP_values.xlsx",
    index=False,
)


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]


# Fig. 9a
plt.figure(figsize=(7.0, 5.2))

shap.summary_plot(
    shap_values.values,
    X,
    feature_names=features,
    plot_type="dot",
    show=False,
    max_display=8,
)

plt.xlabel("SHAP value")
plt.tight_layout()
plt.savefig(
    output_dir / "14_Fig9a_SHAP_scatter.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# Fig. 9b
plt.figure(figsize=(6.5, 5.0))

ordered = importance_df.sort_values("Mean_abs_SHAP")

plt.barh(
    ordered["Feature"],
    ordered["Mean_abs_SHAP"],
)

plt.xlabel("mean(|SHAP value|)")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig(
    output_dir / "14_Fig9b_SHAP_importance.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# Fig. 9c
# 使用 SHAP 官方 heatmap：
# 1. 横轴样本按 SHAP explanation similarity 自动聚类排序
# 2. 纵轴按 mean(|SHAP|) 排列特征
# 3. 红色表示正 SHAP，蓝色表示负 SHAP
# 4. 顶部显示模型输出 f(x)
# 5. 右侧显示全局 mean(|SHAP|) 重要性
fig, ax = plt.subplots(figsize=(9.5, 5.5))

shap.plots.heatmap(
    shap_values,
    max_display=8,
    show=False,
    ax=ax,
)

ax.set_xlabel("Sample index")

fig.savefig(
    output_dir / "14_Fig9c_SHAP_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Fig. 9d
ordered_features = importance_df["Feature"].tolist()
ordered_index = [features.index(name) for name in ordered_features]

fig, ax = plt.subplots(figsize=(8.0, 5.4))

y_positions = np.arange(len(ordered_features))

ax.barh(
    y_positions,
    importance_df["Mean_abs_SHAP"].to_numpy(),
    alpha=0.22,
    label="mean(|SHAP|)",
)

rng = np.random.default_rng(42)

for y_pos, feature_index in zip(
    y_positions,
    ordered_index,
):
    values = shap_values.values[:, feature_index]
    feature_values = X.iloc[:, feature_index].to_numpy()

    jitter = rng.normal(
        loc=0.0,
        scale=0.08,
        size=len(values),
    )

    scatter = ax.scatter(
        values,
        y_pos + jitter,
        c=feature_values,
        cmap="coolwarm",
        s=14,
        alpha=0.75,
        edgecolors="none",
    )

ax.axvline(0, color="gray", linewidth=0.8)

ax.set_yticks(y_positions)
ax.set_yticklabels(ordered_features)
ax.invert_yaxis()

ax.set_xlabel("SHAP value")
ax.set_ylabel("Feature")

cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("Feature value")

fig.tight_layout()
fig.savefig(
    output_dir / "14_Fig9d_SHAP_bar_beeswarm.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Combined Fig. 9
fig = plt.figure(figsize=(13.5, 9.5))
gs = fig.add_gridspec(
    2,
    2,
    hspace=0.38,
    wspace=0.32,
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

ordered_features = importance_df["Feature"].tolist()
ordered_index = [features.index(name) for name in ordered_features]


# Fig. 9a
for y_pos, feature_index in enumerate(ordered_index):
    values = shap_values.values[:, feature_index]
    feature_values = X.iloc[:, feature_index].to_numpy()
    jitter = rng.normal(0, 0.08, size=len(values))

    ax_a.scatter(
        values,
        y_pos + jitter,
        c=feature_values,
        cmap="coolwarm",
        s=13,
        alpha=0.75,
        edgecolors="none",
    )

ax_a.axvline(0, color="gray", linewidth=0.8)
ax_a.set_yticks(np.arange(len(ordered_features)))
ax_a.set_yticklabels(ordered_features)
ax_a.invert_yaxis()
ax_a.set_xlabel("SHAP value")


# Fig. 9b
bar_order = importance_df.sort_values("Mean_abs_SHAP")

ax_b.barh(
    bar_order["Feature"],
    bar_order["Mean_abs_SHAP"],
)

ax_b.set_xlabel("mean(|SHAP value|)")


# Fig. 9c
shap.plots.heatmap(
    shap_values,
    max_display=8,
    show=False,
    ax=ax_c,
)

ax_c.set_xlabel("Sample index")


# Fig. 9d
ax_d.barh(
    np.arange(len(ordered_features)),
    importance_df["Mean_abs_SHAP"].to_numpy(),
    alpha=0.22,
)

for y_pos, feature_index in enumerate(ordered_index):
    values = shap_values.values[:, feature_index]
    feature_values = X.iloc[:, feature_index].to_numpy()
    jitter = rng.normal(0, 0.08, size=len(values))

    scatter_d = ax_d.scatter(
        values,
        y_pos + jitter,
        c=feature_values,
        cmap="coolwarm",
        s=13,
        alpha=0.75,
        edgecolors="none",
    )

ax_d.axvline(0, color="gray", linewidth=0.8)
ax_d.set_yticks(np.arange(len(ordered_features)))
ax_d.set_yticklabels(ordered_features)
ax_d.invert_yaxis()
ax_d.set_xlabel("SHAP value")

fig.colorbar(
    scatter_d,
    ax=ax_d,
    fraction=0.046,
    pad=0.04,
    label="Feature value",
)


for label, ax in zip(
    ["a", "b", "c", "d"],
    [ax_a, ax_b, ax_c, ax_d],
):
    ax.text(
        -0.16,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=25,
        fontweight="bold",
    )


fig.savefig(
    output_dir / "14_Fig9_RF_SHAP.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print()
print(f"Saved to: {output_dir}")
