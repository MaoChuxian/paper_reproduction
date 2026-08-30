from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"

test_path = results_dir / "08_test_final.xlsx"

output_dir = results_dir / "13_top5_test"
output_dir.mkdir(exist_ok=True)

features = [
    "a", "b", "c", "V",
    "concentration",
    "Layer",
    "valence_electron",
    "IQE",
]

model_paths = {
    "Random Forest": results_dir / "11_1_RF" / "11_1_RF.joblib",
    "Decision Tree": results_dir / "11_2_DT" / "11_2_DT.joblib",
    "XGBoost": results_dir / "11_3_XGBoost" / "11_3_XGBoost.joblib",
    "LSBoost": results_dir / "11_4_LSBoost" / "11_4_LSBoost.joblib",
    "GAM": results_dir / "09_4_GAM" / "09_4_GAM.joblib",
}

test_df = pd.read_excel(test_path)

X_test = test_df[features]
y_test = test_df["Lifetime"].to_numpy(dtype=float)


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    mask = y_true != 0
    mape = np.mean(
        np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "MAPE": mape,
        "RMSE": rmse,
        "R2": r2,
    }


models = {
    name: joblib.load(path)
    for name, path in model_paths.items()
}

predictions = {}
records = []

for name, model in models.items():
    pred = model.predict(X_test)
    predictions[name] = pred

    metrics = evaluate(y_test, pred)

    records.append({
        "Model": name,
        **metrics,
    })


metrics_df = pd.DataFrame(records)
metrics_df = metrics_df.sort_values("RMSE").reset_index(drop=True)

print()
print(metrics_df.to_string(index=False))

best_model = metrics_df.iloc[0]["Model"]
print(f"\nBest model on Test = {best_model}")


metrics_df.to_excel(
    output_dir / "13_top5_test_metrics.xlsx",
    index=False,
)

prediction_df = test_df.copy()

for name, pred in predictions.items():
    column = name.replace(" ", "_")
    prediction_df[f"Pred_{column}"] = pred

prediction_df.to_excel(
    output_dir / "13_top5_test_predictions.xlsx",
    index=False,
)


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

colors = {
    "True value": "#8B1A2B",
    "Random Forest": "#A94B3D",
    "Decision Tree": "#D88755",
    "XGBoost": "#E3A26F",
    "LSBoost": "#5578C5",
    "GAM": "#314B9B",
}


def add_border(ax):
    border = FancyBboxPatch(
        (-0.08, -0.12),
        1.16,
        1.22,
        transform=ax.transAxes,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        fill=False,
        edgecolor="#1296DB",
        linewidth=2,
        linestyle=(0, (5, 4)),
        clip_on=False,
    )
    ax.add_patch(border)


fig = plt.figure(figsize=(11.5, 8.8))
gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[1.35, 1],
    hspace=0.35,
    wspace=0.28,
)

ax_a = fig.add_subplot(gs[0, :])
ax_b = fig.add_subplot(gs[1, 0], polar=True)
ax_c = fig.add_subplot(gs[1, 1])

sample = np.arange(1, len(y_test) + 1)

ax_a.plot(
    sample,
    y_test,
    marker="^",
    markersize=3,
    linewidth=1.2,
    color=colors["True value"],
    label="True value",
)

for name in model_paths:
    ax_a.plot(
        sample,
        predictions[name],
        marker="o",
        markersize=2.5,
        linewidth=1.0,
        color=colors[name],
        label=name,
    )

ax_a.set_xlim(1, len(y_test))
ax_a.set_xlabel("sample")
ax_a.set_ylabel("Luminescence lifetime")
ax_a.grid(alpha=0.25)
ax_a.legend(
    loc="lower right",
    fontsize=8,
    frameon=True,
    fancybox=False,
    edgecolor="black",
)
ax_a.tick_params(direction="in", top=True, right=True)
ax_a.text(
    -0.07,
    1.03,
    "a",
    transform=ax_a.transAxes,
    fontsize=28,
    fontweight="bold",
)
add_border(ax_a)


radar_metrics = ["MAPE", "MAE", "1-R2", "RMSE", "MSE"]
angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False)
angles_closed = np.append(angles, angles[0])

for name in model_paths:
    row = metrics_df.loc[metrics_df["Model"] == name].iloc[0]

    values = np.array([
        row["MAPE"],
        row["MAE"],
        1 - row["R2"],
        row["RMSE"],
        row["MSE"],
    ])

    values_closed = np.append(values, values[0])

    ax_b.plot(
        angles_closed,
        values_closed,
        linewidth=1.1,
        marker="o",
        markersize=2.5,
        color=colors[name],
        label=name,
    )

ax_b.set_xticks(angles)
ax_b.set_xticklabels(
    ["A-MAPE", "A-MAE", "1-R2", "A-RMSE", "A-MSE"],
    fontsize=8,
)
ax_b.tick_params(axis="y", labelsize=7)
ax_b.grid(alpha=0.35)
ax_b.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.35),
    fontsize=6,
    frameon=False,
)
ax_b.text(
    -0.18,
    1.08,
    "b",
    transform=ax_b.transAxes,
    fontsize=28,
    fontweight="bold",
)
add_border(ax_b)


bar_metrics = ["MAE", "MAPE", "RMSE"]
x = np.arange(len(bar_metrics))
width = 0.15

for i, name in enumerate(model_paths):
    row = metrics_df.loc[metrics_df["Model"] == name].iloc[0]

    values = [
        row["MAE"],
        row["MAPE"],
        row["RMSE"],
    ]

    ax_c.bar(
        x + (i - 2) * width,
        values,
        width,
        label=name,
        color=colors[name],
        alpha=0.85,
    )

ax_c.set_xticks(x)
ax_c.set_xticklabels(bar_metrics)
ax_c.grid(axis="y", alpha=0.25)
ax_c.tick_params(direction="in")
ax_c.legend(
    loc="upper right",
    fontsize=7,
    frameon=False,
)
ax_c.text(
    -0.14,
    1.03,
    "c",
    transform=ax_c.transAxes,
    fontsize=28,
    fontweight="bold",
)
add_border(ax_c)


fig.savefig(
    output_dir / "13_Fig8_top5_test.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"\nSaved to: {output_dir}")