from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"

clean_path = results_dir / "04_clean_data.xlsx"

output_dir = results_dir / "16_Tb3_generalization"
output_dir.mkdir(exist_ok=True)

features = [
    "a", "b", "c", "V",
    "concentration",
    "Layer",
    "valence_electron",
    "IQE",
]

model_paths = {
    "LSBoost": results_dir / "11_4_LSBoost" / "11_4_LSBoost.joblib",
    "DT": results_dir / "11_2_DT" / "11_2_DT.joblib",
    "RF": results_dir / "11_1_RF" / "11_1_RF.joblib",
    "GAM": results_dir / "09_4_GAM" / "09_4_GAM.joblib",
    "XGBoost": results_dir / "11_3_XGBoost" / "11_3_XGBoost.joblib",
}


# ============================================================
# 1. Supporting Information Table S4
# ============================================================
# 按原补充材料 Table S4 的顺序录入。
#
# 注意：
# 原 Table S4 中 CGW:50%Tb3+ 出现两次，
# Lifetime 分别为 0.551 和 0.541。
# 这里不自行纠正，严格保留原表。
# ============================================================

tb3_df = pd.DataFrame([
    {
        "Name": "CGW:50%Tb3+ (1)",
        "a": 5.2202,
        "b": 5.238,
        "c": 11.4185,
        "V": 311.95,
        "concentration": 50.0,
        "Layer": 6,
        "valence_electron": 11,
        "IQE": 69.0,
        "Lifetime": 0.551,
    },
    {
        "Name": "CGW:10%Tb3+",
        "a": 5.2202,
        "b": 5.238,
        "c": 11.4185,
        "V": 311.95,
        "concentration": 10.0,
        "Layer": 6,
        "valence_electron": 11,
        "IQE": 69.0,
        "Lifetime": 0.571,
    },
    {
        "Name": "CGW:30%Tb3+",
        "a": 5.2202,
        "b": 5.238,
        "c": 11.4185,
        "V": 311.95,
        "concentration": 30.0,
        "Layer": 6,
        "valence_electron": 11,
        "IQE": 69.0,
        "Lifetime": 0.566,
    },
    {
        "Name": "CGW:90%Tb3+",
        "a": 5.2202,
        "b": 5.238,
        "c": 11.4185,
        "V": 311.95,
        "concentration": 90.0,
        "Layer": 6,
        "valence_electron": 11,
        "IQE": 69.0,
        "Lifetime": 0.463,
    },
    {
        "Name": "CGW:20%Tb3+",
        "a": 5.2202,
        "b": 5.238,
        "c": 11.4185,
        "V": 311.95,
        "concentration": 20.0,
        "Layer": 6,
        "valence_electron": 11,
        "IQE": 69.0,
        "Lifetime": 0.547,
    },
    {
        "Name": "CGW:40%Tb3+",
        "a": 5.2202,
        "b": 5.238,
        "c": 11.4185,
        "V": 311.95,
        "concentration": 40.0,
        "Layer": 6,
        "valence_electron": 11,
        "IQE": 69.0,
        "Lifetime": 0.544,
    },
    {
        "Name": "CGW:50%Tb3+ (2)",
        "a": 5.2202,
        "b": 5.238,
        "c": 11.4185,
        "V": 311.95,
        "concentration": 50.0,
        "Layer": 6,
        "valence_electron": 11,
        "IQE": 69.0,
        "Lifetime": 0.541,
    },
    {
        "Name": "CGW:60%Tb3+",
        "a": 5.2202,
        "b": 5.238,
        "c": 11.4185,
        "V": 311.95,
        "concentration": 60.0,
        "Layer": 6,
        "valence_electron": 11,
        "IQE": 69.0,
        "Lifetime": 0.539,
    },
    {
        "Name": "CGW:80%Tb3+",
        "a": 5.2202,
        "b": 5.238,
        "c": 11.4185,
        "V": 311.95,
        "concentration": 80.0,
        "Layer": 6,
        "valence_electron": 11,
        "IQE": 69.0,
        "Lifetime": 0.526,
    },
    {
        "Name": "CGAB:0.3%Tb3+",
        "a": 10.41726,
        "b": 10.41726,
        "c": 5.69949,
        "V": 535.642,
        "concentration": 0.3,
        "Layer": 6,
        "valence_electron": 11,
        "IQE": 17.0,
        "Lifetime": 1.736,
    },
    {
        "Name": "CGAB:0.5%Tb3+",
        "a": 10.41726,
        "b": 10.41726,
        "c": 5.69949,
        "V": 535.642,
        "concentration": 0.5,
        "Layer": 6,
        "valence_electron": 11,
        "IQE": 17.0,
        "Lifetime": 1.733,
    },
    {
        "Name": "CGAB:0.7%Tb3+",
        "a": 10.41726,
        "b": 10.41726,
        "c": 5.69949,
        "V": 535.642,
        "concentration": 0.7,
        "Layer": 6,
        "valence_electron": 11,
        "IQE": 17.0,
        "Lifetime": 1.686,
    },
])


# ============================================================
# 2. Use exactly the same Min-Max scale as the training data
# ============================================================

clean_df = pd.read_excel(clean_path)

scaler = MinMaxScaler()
scaler.fit(
    clean_df[features].astype(float)
)

X_tb3_norm = scaler.transform(
    tb3_df[features].astype(float)
)

tb3_norm_df = pd.DataFrame(
    X_tb3_norm,
    columns=features,
)

print()
print("Normalized Tb3 feature ranges")
for feature in features:
    print(
        f"{feature:18s}: "
        f"{tb3_norm_df[feature].min():.6f} "
        f"~ {tb3_norm_df[feature].max():.6f}"
    )

print()
print(
    "Values outside [0, 1] are kept. "
    "They represent true extrapolation to the new Tb3+ dopant."
)


# ============================================================
# 3. Load the five models selected in the paper
# ============================================================

models = {
    name: joblib.load(path)
    for name, path in model_paths.items()
}


# ============================================================
# 4. Predict and calculate APE
# ============================================================

y_true = tb3_df["Lifetime"].to_numpy(dtype=float)

prediction_df = tb3_df.copy()
ape_df = pd.DataFrame({
    "Name": tb3_df["Name"],
    "Lifetime": y_true,
})

summary_rows = []

for name, model in models.items():
    pred = np.asarray(
        model.predict(tb3_norm_df[features]),
        dtype=float,
    )

    ape = np.abs(
        (pred - y_true) / y_true
    )

    prediction_df[f"Pred_{name}"] = pred
    ape_df[f"APE_{name}"] = ape

    summary_rows.append({
        "Model": name,
        "Mean_APE": ape.mean(),
        "Median_APE": np.median(ape),
        "Min_APE_Count": 0,
    })


# ============================================================
# 5. Count how many samples each model wins
# ============================================================

ape_columns = [
    f"APE_{name}"
    for name in models
]

ape_matrix = ape_df[ape_columns].to_numpy()

winner_index = np.argmin(
    ape_matrix,
    axis=1,
)

model_names = list(models.keys())

ape_df["Best_Model"] = [
    model_names[i]
    for i in winner_index
]

for row in summary_rows:
    row["Min_APE_Count"] = int(
        (ape_df["Best_Model"] == row["Model"]).sum()
    )

summary_df = pd.DataFrame(
    summary_rows
).sort_values(
    ["Min_APE_Count", "Mean_APE"],
    ascending=[False, True],
)


print()
print("Tb3+ external validation summary")
print(summary_df.to_string(index=False))

print()
print("Best model for each Tb3+ sample")
print(
    ape_df[
        ["Name", "Best_Model"]
    ].to_string(index=False)
)


# ============================================================
# 6. Save tables
# ============================================================

tb3_df.to_excel(
    output_dir / "16_TableS4_Tb3_raw.xlsx",
    index=False,
)

tb3_norm_save = tb3_df[["Name", "Lifetime"]].copy()

for feature in features:
    tb3_norm_save[feature] = tb3_norm_df[feature]

tb3_norm_save.to_excel(
    output_dir / "16_TableS4_Tb3_normalized.xlsx",
    index=False,
)

prediction_df.to_excel(
    output_dir / "16_Tb3_predictions.xlsx",
    index=False,
)

ape_df.to_excel(
    output_dir / "16_Tb3_APE.xlsx",
    index=False,
)

summary_df.to_excel(
    output_dir / "16_Tb3_summary.xlsx",
    index=False,
)


# ============================================================
# 7. Reproduce paper Fig. 11
# ============================================================

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Times New Roman",
    "DejaVu Serif",
]
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


colors = {
    "LSBoost": "#5578C5",
    "DT": "#D88755",
    "RF": "#A94B3D",
    "GAM": "#314B9B",
    "XGBoost": "#E3A26F",
}

names = tb3_df["Name"].tolist()
x = np.arange(len(names))

width = 0.15

fig, ax = plt.subplots(
    figsize=(13.0, 6.0)
)

offsets = np.linspace(
    -2 * width,
    2 * width,
    len(models),
)

for offset, name in zip(
    offsets,
    model_names,
):
    ax.bar(
        x + offset,
        ape_df[f"APE_{name}"],
        width,
        label=name,
        color=colors[name],
        edgecolor="none",
    )


ax.set_ylabel("Absolute Percentage Error (APE)")
ax.set_xlabel("")

ax.set_xticks(x)
ax.set_xticklabels(
    names,
    rotation=45,
    ha="right",
)

ax.legend(
    ncol=5,
    frameon=False,
    loc="upper center",
)

ax.grid(
    axis="y",
    alpha=0.22,
    linewidth=0.8,
)

ax.set_xlim(
    -0.65,
    len(names) - 0.35,
)

fig.tight_layout()

fig.savefig(
    output_dir / "16_Fig11_Tb3_APE.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print()
print(f"Saved to: {output_dir}")