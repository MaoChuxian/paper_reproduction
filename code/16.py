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

model_dir = results_dir / "15_RF_Bayesian_TPE"
rf_path = model_dir / "15_Original_RF_refit.joblib"
bayes_rf_path = model_dir / "15_Bayesian_RF_TPE.joblib"

output_dir = model_dir
output_dir.mkdir(exist_ok=True)

features = [
    "a", "b", "c", "V",
    "concentration",
    "Layer",
    "valence_electron",
    "IQE",
]


test_df = pd.read_excel(test_path)

X_test = test_df[features]
y_test = test_df["Lifetime"].to_numpy()

rf_model = joblib.load(rf_path)
bayes_rf_model = joblib.load(bayes_rf_path)

pred_rf = rf_model.predict(X_test)
pred_bayes = bayes_rf_model.predict(X_test)


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    mask = y_true != 0
    mape = np.mean(
        np.abs(
            (y_pred[mask] - y_true[mask])
            / y_true[mask]
        )
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "MAPE": mape,
        "RMSE": rmse,
        "R2": r2,
    }


rf_metrics = evaluate(y_test, pred_rf)
bayes_metrics = evaluate(y_test, pred_bayes)

metrics_df = pd.DataFrame([
    {
        "Model": "Random Forest",
        **rf_metrics,
    },
    {
        "Model": "Bayesian optimized RF",
        **bayes_metrics,
    },
])

print()
print("Fig.10 data")
print(metrics_df.to_string(index=False))

metrics_df.to_excel(
    output_dir / "15_Fig10_metrics.xlsx",
    index=False,
)

prediction_df = test_df.copy()
prediction_df["Random_Forest"] = pred_rf
prediction_df["Bayesian_optimized_RF"] = pred_bayes

prediction_df.to_excel(
    output_dir / "15_Fig10_predictions.xlsx",
    index=False,
)


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Times New Roman",
    "DejaVu Serif",
]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 1.0


sample = np.arange(1, len(y_test) + 1)

y_min = min(
    np.min(y_test),
    np.min(pred_rf),
    np.min(pred_bayes),
)
y_max = max(
    np.max(y_test),
    np.max(pred_rf),
    np.max(pred_bayes),
)
y_pad = 0.10 * (y_max - y_min)


# Fig. 10a
fig, ax = plt.subplots(figsize=(9.2, 4.8))

ax.plot(
    sample,
    y_test,
    color="#9E1025",
    linewidth=1.5,
    marker="^",
    markersize=4.2,
    label="True value",
)

ax.plot(
    sample,
    pred_rf,
    color="#8EB6E8",
    linewidth=1.5,
    marker="*",
    markersize=5.0,
    label="Random Forest",
)

ax.plot(
    sample,
    pred_bayes,
    color="#315EBA",
    linewidth=1.5,
    marker="*",
    markersize=5.0,
    label="Bayesian optimized RF",
)

ax.set_xlim(0, len(y_test) + 2)
ax.set_ylim(
    max(0, y_min - y_pad),
    y_max + y_pad,
)

ax.grid(True, linewidth=0.8, alpha=0.30)

ax.legend(
    loc="lower right",
    frameon=True,
    fancybox=False,
    edgecolor="black",
    framealpha=1.0,
    fontsize=10,
)

fig.tight_layout()

fig.savefig(
    output_dir / "15_Fig10a_prediction_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# Fig. 10b
bar_names = ["MAE", "MAPE", "RMSE"]

bayes_values = [
    bayes_metrics["MAE"],
    bayes_metrics["MAPE"],
    bayes_metrics["RMSE"],
]

rf_values = [
    rf_metrics["MAE"],
    rf_metrics["MAPE"],
    rf_metrics["RMSE"],
]

x = np.arange(len(bar_names))
width = 0.23

fig, ax = plt.subplots(figsize=(5.6, 4.7))

ax.bar(
    x - width / 2,
    bayes_values,
    width,
    color="#2F7FB5",
    edgecolor="0.45",
    linewidth=0.5,
    label="Bayesian optimized RF",
)

ax.bar(
    x + width / 2,
    rf_values,
    width,
    color="#93C7D8",
    edgecolor="0.45",
    linewidth=0.5,
    label="Random Forest",
)

for sep in [0.5, 1.5]:
    ax.axvline(
        sep,
        color="0.45",
        linestyle=(0, (3, 3)),
        linewidth=0.8,
    )

ax.set_xticks(x)
ax.set_xticklabels(bar_names)
ax.set_xlim(-0.55, 2.55)

upper = max(max(bayes_values), max(rf_values)) * 1.13
ax.set_ylim(0, upper)

ax.legend(
    loc="upper right",
    frameon=False,
    fontsize=8.5,
)

ax.tick_params(
    direction="in",
    top=False,
    right=False,
)

fig.tight_layout()

fig.savefig(
    output_dir / "15_Fig10b_error_metrics.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# Fig. 10c
fig, ax = plt.subplots(figsize=(5.6, 4.7))

ax.scatter(
    bayes_metrics["MAE"],
    bayes_metrics["R2"],
    marker="s",
    s=38,
    color="#1F77B4",
    label="Bayesian optimized RF",
    zorder=3,
)

ax.scatter(
    rf_metrics["MAE"],
    rf_metrics["R2"],
    marker="o",
    s=40,
    color="#D95F02",
    label="Random Forest",
    zorder=3,
)

mae_values = np.array([
    rf_metrics["MAE"],
    bayes_metrics["MAE"],
])

r2_values = np.array([
    rf_metrics["R2"],
    bayes_metrics["R2"],
])

x_span = max(mae_values.max() - mae_values.min(), 0.005)
y_span = max(r2_values.max() - r2_values.min(), 0.01)

ax.set_xlim(
    mae_values.min() - 0.22 * x_span,
    mae_values.max() + 0.22 * x_span,
)

ax.set_ylim(
    r2_values.min() - 0.22 * y_span,
    min(
        1.0,
        r2_values.max() + 0.22 * y_span,
    ),
)

ax.set_xlabel("MAE")
ax.set_ylabel(r"$R^2$")

ax.grid(True, linewidth=0.8, alpha=0.30)

ax.legend(
    loc="upper right",
    frameon=False,
    fontsize=8.5,
)

fig.tight_layout()

fig.savefig(
    output_dir / "15_Fig10c_R2_MAE.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# Combined Fig. 10
fig = plt.figure(figsize=(12.8, 9.6))

gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[1.28, 1.0],
    hspace=0.28,
    wspace=0.28,
)

ax_a = fig.add_subplot(gs[0, :])
ax_b = fig.add_subplot(gs[1, 0])
ax_c = fig.add_subplot(gs[1, 1])


ax_a.plot(
    sample,
    y_test,
    color="#9E1025",
    linewidth=1.5,
    marker="^",
    markersize=4.2,
    label="True value",
)

ax_a.plot(
    sample,
    pred_rf,
    color="#8EB6E8",
    linewidth=1.5,
    marker="*",
    markersize=5.0,
    label="Random Forest",
)

ax_a.plot(
    sample,
    pred_bayes,
    color="#315EBA",
    linewidth=1.5,
    marker="*",
    markersize=5.0,
    label="Bayesian optimized RF",
)

ax_a.set_xlim(0, len(y_test) + 2)
ax_a.set_ylim(
    max(0, y_min - y_pad),
    y_max + y_pad,
)

ax_a.grid(True, linewidth=0.8, alpha=0.30)

ax_a.legend(
    loc="lower right",
    frameon=True,
    fancybox=False,
    edgecolor="black",
    framealpha=1.0,
    fontsize=10,
)


ax_b.bar(
    x - width / 2,
    bayes_values,
    width,
    color="#2F7FB5",
    edgecolor="0.45",
    linewidth=0.5,
    label="Bayesian optimized RF",
)

ax_b.bar(
    x + width / 2,
    rf_values,
    width,
    color="#93C7D8",
    edgecolor="0.45",
    linewidth=0.5,
    label="Random Forest",
)

for sep in [0.5, 1.5]:
    ax_b.axvline(
        sep,
        color="0.45",
        linestyle=(0, (3, 3)),
        linewidth=0.8,
    )

ax_b.set_xticks(x)
ax_b.set_xticklabels(bar_names)
ax_b.set_xlim(-0.55, 2.55)
ax_b.set_ylim(0, upper)

ax_b.legend(
    loc="upper right",
    frameon=False,
    fontsize=8.5,
)


ax_c.scatter(
    bayes_metrics["MAE"],
    bayes_metrics["R2"],
    marker="s",
    s=38,
    color="#1F77B4",
    label="Bayesian optimized RF",
    zorder=3,
)

ax_c.scatter(
    rf_metrics["MAE"],
    rf_metrics["R2"],
    marker="o",
    s=40,
    color="#D95F02",
    label="Random Forest",
    zorder=3,
)

ax_c.set_xlim(
    mae_values.min() - 0.22 * x_span,
    mae_values.max() + 0.22 * x_span,
)

ax_c.set_ylim(
    r2_values.min() - 0.22 * y_span,
    min(
        1.0,
        r2_values.max() + 0.22 * y_span,
    ),
)

ax_c.set_xlabel("MAE")
ax_c.set_ylabel(r"$R^2$")

ax_c.grid(True, linewidth=0.8, alpha=0.30)

ax_c.legend(
    loc="upper right",
    frameon=False,
    fontsize=8.5,
)


for label, ax in zip(
    ["a", "b", "c"],
    [ax_a, ax_b, ax_c],
):
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=28,
        fontweight="bold",
        va="bottom",
    )


for ax in [ax_a, ax_b, ax_c]:
    box = ax.get_position()

    pad_x = 0.025
    pad_y = 0.030

    patch = FancyBboxPatch(
        (
            box.x0 - pad_x,
            box.y0 - pad_y,
        ),
        box.width + 2 * pad_x,
        box.height + 2 * pad_y,
        boxstyle="round,pad=0.015,rounding_size=0.045",
        transform=fig.transFigure,
        fill=False,
        edgecolor="#00A7E1",
        linewidth=2.0,
        linestyle=(0, (6, 5)),
        clip_on=False,
        zorder=10,
    )

    fig.add_artist(patch)


fig.savefig(
    output_dir / "15_Fig10_RF_Bayesian_reproduction.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print()
print(f"Saved to: {output_dir}")