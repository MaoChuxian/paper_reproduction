from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


PAPER_R2 = 0.96877

project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"

train_path = results_dir / "08_train_final.xlsx"
val_path = results_dir / "08_validation_final.xlsx"

output_dir = results_dir / "11_2_DT"
output_dir.mkdir(exist_ok=True)

features = [
    "a", "b", "c", "V",
    "concentration",
    "Layer",
    "valence_electron",
    "IQE",
]

train_df = pd.read_excel(train_path)
val_df = pd.read_excel(val_path)

X_train = train_df[features]
y_train = train_df["Lifetime"].to_numpy()

X_val = val_df[features]
y_val = val_df["Lifetime"].to_numpy()


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


best_model = None
best_pred = None
best_result = None
best_rmse = np.inf
records = []

for max_depth in [None, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30]:
    for min_samples_split in [2, 3, 4, 5, 6, 8, 10]:
        for min_samples_leaf in [1, 2, 3, 4]:
            for max_features in [None, 0.75, 0.875, "sqrt"]:
                for ccp_alpha in [0.0, 1e-5, 1e-4, 1e-3]:

                    model = DecisionTreeRegressor(
                        criterion="squared_error",
                        splitter="best",
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        ccp_alpha=ccp_alpha,
                        random_state=42,
                    )

                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    metrics = evaluate(y_val, pred)

                    row = {
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf,
                        "max_features": max_features,
                        "ccp_alpha": ccp_alpha,
                        **metrics,
                    }

                    records.append(row)

                    if metrics["RMSE"] < best_rmse:
                        best_rmse = metrics["RMSE"]
                        best_model = model
                        best_pred = pred
                        best_result = row


result = pd.DataFrame([{
    "Model": "Decision Tree",
    **best_result,
    "Paper_R2": PAPER_R2,
    "R2_Difference": best_result["R2"] - PAPER_R2,
}])

print()
print(result.to_string(index=False))
print(f"\nPaper R2 = {PAPER_R2:.5f}")
print(f"R2 difference = {best_result['R2'] - PAPER_R2:+.5f}")

result.to_excel(
    output_dir / "11_2_DT_metrics.xlsx",
    index=False,
)

pd.DataFrame(records).sort_values("RMSE").to_excel(
    output_dir / "11_2_DT_tuning.xlsx",
    index=False,
)

prediction_df = val_df.copy()
prediction_df["Pred_DT"] = best_pred

prediction_df.to_excel(
    output_dir / "11_2_DT_predictions.xlsx",
    index=False,
)

joblib.dump(
    best_model,
    output_dir / "11_2_DT.joblib",
)


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]


def plot_result(y_true, y_pred, save_path):
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

    inset.plot(
        sample,
        y_true,
        "--",
        linewidth=0.8,
        label="True value",
    )

    inset.plot(
        sample,
        y_pred,
        linewidth=0.8,
        label="Fit value",
    )

    inset.tick_params(
        labelsize=6,
        direction="in",
    )

    inset.legend(
        fontsize=5,
        frameon=False,
        loc="upper right",
    )

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

    cbar = fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.046,
        pad=0.025,
    )

    cbar.set_ticks([
        y_true.max(),
        (y_true.max() + y_true.min()) / 2,
        y_true.min(),
    ])

    cbar.set_ticklabels([
        "H",
        "M",
        "L",
    ])

    border = FancyBboxPatch(
        (-0.12, -0.13),
        1.25,
        1.22,
        transform=ax.transAxes,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        fill=False,
        edgecolor="#8E44AD",
        linewidth=2,
        linestyle=(0, (5, 4)),
        clip_on=False,
    )

    ax.add_patch(border)

    ax.text(
        -0.19,
        1.07,
        "b",
        transform=ax.transAxes,
        fontsize=29,
        fontweight="bold",
        family="serif",
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


plot_result(
    y_val,
    best_pred,
    output_dir / "11_2_DT.png",
)

print()
print("Figure saved to:")
print(output_dir / "11_2_DT.png")