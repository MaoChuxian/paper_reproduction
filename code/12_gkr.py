from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"

train_path = results_dir / "08_train_final.xlsx"
val_path = results_dir / "08_validation_final.xlsx"

features = [
    "a", "b", "c", "V",
    "concentration",
    "Layer",
    "valence_electron",
    "IQE",
]

train_df = pd.read_excel(train_path)
val_df = pd.read_excel(val_path)

X_train = train_df[features].to_numpy(dtype=float)
y_train = train_df["Lifetime"].to_numpy(dtype=float)

X_val = val_df[features].to_numpy(dtype=float)
y_val = val_df["Lifetime"].to_numpy(dtype=float)


PAPER_R2 = 0.37702

output_dir = results_dir / "12_1_GKR"
output_dir.mkdir(exist_ok=True)

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    mask = y_true != 0
    mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]))

    return {
        "MAE": mae,
        "MSE": mse,
        "MAPE": mape,
        "RMSE": rmse,
        "R2": r2,
    }


def plot_result(y_true, y_pred, save_path, border_color, panel_label):
    fig, ax = plt.subplots(figsize=(6.3, 5.2))

    scatter = ax.scatter(
        y_true,
        y_pred,
        c=y_true,
        cmap="RdYlGn",
        s=26,
        edgecolors="none",
    )

    slope, intercept = np.polyfit(y_true, y_pred, 1)
    x_line = np.linspace(y_true.min(), y_true.max(), 300)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color="#5F6B73",
        linewidth=1.6,
    )

    r2 = r2_score(y_true, y_pred)
    sign = "+" if intercept >= 0 else "-"

    ax.text(
        0.05,
        0.95,
        f"Num = {len(y_true)}\n"
        f"y = {slope:.3f}x {sign} {abs(intercept):.3f}\n"
        f"R$^2$ = {r2:.5f}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
    )

    ax.set_xlabel("true data")
    ax.set_ylabel("predicted data")
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(alpha=0.25)

    inset = ax.inset_axes([0.59, 0.20, 0.29, 0.23])
    sample = np.arange(1, len(y_true) + 1)
    inset.plot(sample, y_true, "--", linewidth=0.8, label="True value")
    inset.plot(sample, y_pred, linewidth=0.8, label="Fit value")
    inset.tick_params(labelsize=6, direction="in")
    inset.legend(fontsize=5, frameon=False, loc="upper right")

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#B0002B",
            markeredgecolor="#B0002B",
            markersize=6,
            label="true-predict data",
        ),
        Line2D(
            [0], [0],
            color="#5F6B73",
            linewidth=1.6,
            label="Fit data",
        ),
    ]

    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=9,
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_ticks([
        y_true.max(),
        (y_true.max() + y_true.min()) / 2,
        y_true.min(),
    ])
    cbar.set_ticklabels(["H", "M", "L"])

    border = FancyBboxPatch(
        (-0.12, -0.13),
        1.25,
        1.22,
        transform=ax.transAxes,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        fill=False,
        edgecolor=border_color,
        linewidth=2,
        linestyle=(0, (5, 4)),
        clip_on=False,
    )
    ax.add_patch(border)

    ax.text(
        -0.19,
        1.07,
        panel_label,
        transform=ax.transAxes,
        fontsize=29,
        fontweight="bold",
        family="serif",
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def gkr_predict(X_train, y_train, X, bandwidth):
    d2 = np.sum((X[:, None, :] - X_train[None, :, :]) ** 2, axis=2)
    weights = np.exp(-d2 / (2 * bandwidth ** 2))
    weight_sum = weights.sum(axis=1)
    return (weights @ y_train) / weight_sum


records = []

for bandwidth in np.linspace(0.15, 0.40, 101):
    pred = gkr_predict(X_train, y_train, X_val, bandwidth)
    metrics = evaluate(y_val, pred)
    records.append({"bandwidth": bandwidth, **metrics})

tuning = pd.DataFrame(records)
best = tuning.loc[tuning["RMSE"].idxmin()]
bandwidth = float(best["bandwidth"])

pred = gkr_predict(X_train, y_train, X_val, bandwidth)
metrics = evaluate(y_val, pred)

result = pd.DataFrame([{
    "Model": "GKR",
    "bandwidth": bandwidth,
    **metrics,
    "Paper_R2": PAPER_R2,
    "R2_Difference": metrics["R2"] - PAPER_R2,
}])

print()
print(result.to_string(index=False))
print(f"\nPaper R2 = {PAPER_R2:.5f}")
print(f"R2 difference = {metrics['R2'] - PAPER_R2:+.5f}")

result.to_excel(output_dir / "12_1_GKR_metrics.xlsx", index=False)
tuning.sort_values("RMSE").to_excel(output_dir / "12_1_GKR_tuning.xlsx", index=False)

prediction_df = val_df.copy()
prediction_df["Pred_GKR"] = pred
prediction_df.to_excel(output_dir / "12_1_GKR_predictions.xlsx", index=False)

joblib.dump(
    {"X_train": X_train, "y_train": y_train, "bandwidth": bandwidth},
    output_dir / "12_1_GKR.joblib",
)

plot_result(y_val, pred, output_dir / "12_1_GKR.png", "#8E44AD", "a")
