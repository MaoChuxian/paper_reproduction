from pathlib import Path
import copy
import random

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


torch.set_num_threads(1)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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

target = "Lifetime"

train_df = pd.read_excel(train_path)
val_df = pd.read_excel(val_path)

X_train = torch.tensor(
    train_df[features].to_numpy(),
    dtype=torch.float32,
)

y_train = torch.tensor(
    train_df[target].to_numpy(),
    dtype=torch.float32,
).view(-1, 1)

X_val = torch.tensor(
    val_df[features].to_numpy(),
    dtype=torch.float32,
)

y_val = torch.tensor(
    val_df[target].to_numpy(),
    dtype=torch.float32,
).view(-1, 1)


output_dir = results_dir / "10_3_CNN"
output_dir.mkdir(exist_ok=True)


class CNNRegressor(nn.Module):
    def __init__(self, channels1, channels2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, channels1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels1, channels2, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.output = nn.Linear(channels2 * 8, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.flatten(1)
        return self.output(x)

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


def plot_result(y_true, y_pred, save_path, panel_label):
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
    inset.legend(fontsize=5, frameon=False)

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
        edgecolor="#1296DB",
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

def train_model(model, lr, weight_decay, max_epochs=1500, patience=120):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state = None
    best_val_loss = np.inf
    best_epoch = 0
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()

        optimizer.zero_grad()
        train_pred = model(X_train)
        train_loss = loss_fn(train_pred, y_train)

        train_loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred = model(X_val).squeeze(1).numpy()

    return model, pred, best_epoch


configs = [
    {"channels1": 8, "channels2": 16, "lr": 0.001, "weight_decay": 1e-4},
    {"channels1": 16, "channels2": 32, "lr": 0.001, "weight_decay": 1e-4},
    {"channels1": 32, "channels2": 64, "lr": 0.0005, "weight_decay": 1e-4},
]

best_model = None
best_pred = None
best_config = None
best_epoch = None
best_rmse = np.inf
records = []

for config in configs:
    torch.manual_seed(SEED)

    model = CNNRegressor(
        channels1=config["channels1"],
        channels2=config["channels2"],
    )

    model, pred, epoch = train_model(
        model,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    metrics = evaluate(
        y_val.squeeze(1).numpy(),
        pred,
    )

    records.append({
        **config,
        "best_epoch": epoch,
        **metrics,
    })

    if metrics["RMSE"] < best_rmse:
        best_rmse = metrics["RMSE"]
        best_model = model
        best_pred = pred
        best_config = config
        best_epoch = epoch

metrics = evaluate(
    y_val.squeeze(1).numpy(),
    best_pred,
)

result = pd.DataFrame([{
    "Model": "CNN",
    **best_config,
    "best_epoch": best_epoch,
    **metrics,
}])

print(result.to_string(index=False))

result.to_excel(
    output_dir / "10_3_CNN_metrics.xlsx",
    index=False,
)

pd.DataFrame(records).sort_values("RMSE").to_excel(
    output_dir / "10_3_CNN_tuning.xlsx",
    index=False,
)

prediction_df = val_df.copy()
prediction_df["Pred_CNN"] = best_pred
prediction_df.to_excel(
    output_dir / "10_3_CNN_predictions.xlsx",
    index=False,
)

torch.save(
    best_model.state_dict(),
    output_dir / "10_3_CNN.pt",
)

plot_result(
    y_val.squeeze(1).numpy(),
    best_pred,
    output_dir / "10_3_CNN.png",
    "c",
)