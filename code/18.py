from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler


project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"

clean_path = results_dir / "04_clean_data.xlsx"

model_paths = {
    "LSBoost": (
        results_dir
        / "16_3_LSBoost_MATLAB_style"
        / "16_3_LSBoost_MATLAB_style.joblib"
    ),
    "DT": (
        results_dir
        / "11_2_DT"
        / "11_2_DT.joblib"
    ),
    "RF": (
        results_dir
        / "15_RF_Bayesian_TPE"
        / "15_Bayesian_RF_TPE.joblib"
    ),
    "GAM": (
        results_dir
        / "09_4_GAM"
        / "09_4_GAM.joblib"
    ),
    "XGBoost": (
        results_dir
        / "11_3_XGBoost"
        / "11_3_XGBoost.joblib"
    ),
}

output_dir = results_dir / "16_4_Tb3_final_figure"
output_dir.mkdir(exist_ok=True)

features = [
    "a", "b", "c", "V",
    "concentration",
    "Layer",
    "valence_electron",
    "IQE",
]


tb3_df = pd.DataFrame([
    ["CGW:50%Tb3+ (1)", 5.2202, 5.23800, 11.41850, 311.950, 50.0, 6, 11, 69.0, 0.551],
    ["CGW:10%Tb3+",     5.2202, 5.23800, 11.41850, 311.950, 10.0, 6, 11, 69.0, 0.571],
    ["CGW:30%Tb3+",     5.2202, 5.23800, 11.41850, 311.950, 30.0, 6, 11, 69.0, 0.566],
    ["CGW:90%Tb3+",     5.2202, 5.23800, 11.41850, 311.950, 90.0, 6, 11, 69.0, 0.463],
    ["CGW:20%Tb3+",     5.2202, 5.23800, 11.41850, 311.950, 20.0, 6, 11, 69.0, 0.547],
    ["CGW:40%Tb3+",     5.2202, 5.23800, 11.41850, 311.950, 40.0, 6, 11, 69.0, 0.544],
    ["CGW:50%Tb3+ (2)", 5.2202, 5.23800, 11.41850, 311.950, 50.0, 6, 11, 69.0, 0.541],
    ["CGW:60%Tb3+",     5.2202, 5.23800, 11.41850, 311.950, 60.0, 6, 11, 69.0, 0.539],
    ["CGW:80%Tb3+",     5.2202, 5.23800, 11.41850, 311.950, 80.0, 6, 11, 69.0, 0.526],
    ["CGAB:0.3%Tb3+",  10.41726, 10.41726, 5.69949, 535.642, 0.3, 6, 11, 17.0, 1.736],
    ["CGAB:0.5%Tb3+",  10.41726, 10.41726, 5.69949, 535.642, 0.5, 6, 11, 17.0, 1.733],
    ["CGAB:0.7%Tb3+",  10.41726, 10.41726, 5.69949, 535.642, 0.7, 6, 11, 17.0, 1.686],
], columns=[
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
])


clean_df = pd.read_excel(clean_path)

scaler = MinMaxScaler()
scaler.fit(clean_df[features].astype(float))

tb3_norm = pd.DataFrame(
    scaler.transform(
        tb3_df[features].astype(float)
    ),
    columns=features,
)


models = {
    name: joblib.load(path)
    for name, path in model_paths.items()
}

y_true = tb3_df["Lifetime"].to_numpy(dtype=float)

result_df = tb3_df[
    ["Name", "Lifetime"]
].copy()

for name, model in models.items():
    pred = np.asarray(
        model.predict(
            tb3_norm[features]
        ),
        dtype=float,
    )

    ape = (
        np.abs(
            (pred - y_true)
            / y_true
        )
        * 100
    )

    result_df[f"Pred_{name}"] = pred
    result_df[f"APE_{name}"] = ape


ape_columns = [
    f"APE_{name}"
    for name in model_paths
]

ape_matrix = result_df[
    ape_columns
].to_numpy()

best_index = np.argmin(
    ape_matrix,
    axis=1,
)

model_names = list(
    model_paths.keys()
)

result_df["Best_Model"] = [
    model_names[i]
    for i in best_index
]

result_df["Min_APE"] = (
    ape_matrix.min(axis=1)
)


summary_rows = []

for name in model_paths:
    ape = result_df[
        f"APE_{name}"
    ].to_numpy()

    summary_rows.append({
        "Model": name,
        "Mean_APE": ape.mean(),
        "Median_APE": np.median(ape),
        "CGW_Mean_APE": ape[:9].mean(),
        "CGAB_Mean_APE": ape[9:].mean(),
        "Min_APE_Count": int(
            (
                result_df["Best_Model"]
                == name
            ).sum()
        ),
    })

summary_df = (
    pd.DataFrame(summary_rows)
    .sort_values(
        [
            "Min_APE_Count",
            "Mean_APE",
        ],
        ascending=[
            False,
            True,
        ],
    )
)


print()
print("Tb3+ external validation summary")
print(
    summary_df.to_string(
        index=False
    )
)

print()
print("Best model for each Tb3+ sample")
print(
    result_df[
        [
            "Name",
            "Best_Model",
            "Min_APE",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


with pd.ExcelWriter(
    output_dir
    / "16_4_Tb3_final_results.xlsx"
) as writer:
    result_df.to_excel(
        writer,
        sheet_name="predictions",
        index=False,
    )

    summary_df.to_excel(
        writer,
        sheet_name="summary",
        index=False,
    )


plot_order = [
    "LSBoost",
    "DT",
    "RF",
    "XGBoost",
    "GAM",
]

colors = {
    "LSBoost": "#4C63C7",
    "DT": "#56BCEB",
    "RF": "#94D83F",
    "GAM": "#5BC9AE",
    "XGBoost": "#EB7A17",
}

x = np.arange(
    len(tb3_df)
)

y = np.arange(
    len(plot_order)
)

dx = 0.42
dy = 0.52


fig = plt.figure(
    figsize=(13.5, 7.5)
)

ax = fig.add_subplot(
    111,
    projection="3d",
)


for j, model_name in enumerate(
    plot_order
):
    values = result_df[
        f"APE_{model_name}"
    ].to_numpy()

    ax.bar3d(
        x - dx / 2,
        np.full(
            len(tb3_df),
            j,
        ) - dy / 2,
        np.zeros(
            len(tb3_df)
        ),
        dx,
        dy,
        values,
        color=colors[
            model_name
        ],
        edgecolor="none",
        shade=True,
        alpha=0.98,
    )


ax.set_title(
    "Absolute Percentage Error of different algorithms",
    pad=22,
    fontsize=16,
    fontweight="bold",
)

ax.set_xlim(
    -0.6,
    len(tb3_df) - 0.4,
)

ax.set_ylim(
    -0.7,
    len(plot_order) - 0.2,
)

z_max = max(
    35,
    np.ceil(
        ape_matrix.max()
        / 10
    )
    * 10,
)

ax.set_zlim(
    0,
    z_max,
)

ax.set_xticks(x)

ax.set_xticklabels(
    tb3_df["Name"],
    rotation=45,
    ha="right",
    fontsize=9,
)

ax.set_yticks([])

ax.set_zlabel(
    "APE (%)",
    labelpad=10,
)

ax.view_init(
    elev=24,
    azim=-64,
)


ax.xaxis.pane.set_facecolor(
    (1, 1, 1, 1)
)

ax.yaxis.pane.set_facecolor(
    (1, 1, 1, 1)
)

ax.zaxis.pane.set_facecolor(
    (1, 1, 1, 1)
)


ax.xaxis._axinfo[
    "grid"
]["color"] = (
    0.86,
    0.86,
    0.86,
    1,
)

ax.yaxis._axinfo[
    "grid"
]["color"] = (
    0.86,
    0.86,
    0.86,
    1,
)

ax.zaxis._axinfo[
    "grid"
]["color"] = (
    0.82,
    0.82,
    0.82,
    1,
)


handles = [
    Patch(
        facecolor=colors[name],
        edgecolor="none",
        label=name,
    )
    for name in plot_order
]

ax.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        -0.04,
    ),
    ncol=5,
    frameon=False,
    fontsize=11,
)


fig.tight_layout()

fig.savefig(
    output_dir
    / "16_4_Fig11_Tb3_final.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print()
print(
    f"Saved to: {output_dir}"
)
