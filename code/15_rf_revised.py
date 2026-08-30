from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"

train_path = results_dir / "08_train_final.xlsx"
val_path = results_dir / "08_validation_final.xlsx"
test_path = results_dir / "08_test_final.xlsx"
original_model_path = results_dir / "11_1_RF" / "11_1_RF.joblib"

output_dir = results_dir / "15_RF_Bayesian_TPE"
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
test_df = pd.read_excel(test_path)

dev_df = pd.concat([train_df, val_df], ignore_index=True)

X_dev = dev_df[features]
y_dev = dev_df["Lifetime"].to_numpy()

X_test = test_df[features]
y_test = test_df["Lifetime"].to_numpy()

original_model = joblib.load(original_model_path)
original_params = original_model.get_params()


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


kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


def cv_rmse(model_params):
    fold_rmse = []

    for train_index, val_index in kf.split(X_dev):
        X_train_fold = X_dev.iloc[train_index]
        y_train_fold = y_dev[train_index]

        X_val_fold = X_dev.iloc[val_index]
        y_val_fold = y_dev[val_index]

        model = RandomForestRegressor(**model_params)
        model.fit(X_train_fold, y_train_fold)

        pred = model.predict(X_val_fold)

        rmse = np.sqrt(
            mean_squared_error(y_val_fold, pred)
        )

        fold_rmse.append(rmse)

    return (
        float(np.mean(fold_rmse)),
        float(np.std(fold_rmse)),
        fold_rmse,
    )


baseline_params = original_params.copy()
baseline_params["random_state"] = 42
baseline_params["n_jobs"] = -1

baseline_mean_cv, baseline_std_cv, baseline_folds = cv_rmse(
    baseline_params
)

print()
print("Original tuned RF")
print(f"Mean 5-fold CV RMSE = {baseline_mean_cv:.6f}")
print(f"Std  5-fold CV RMSE = {baseline_std_cv:.6f}")


history = []


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int(
            "Tree_Num", 20, 600
        ),
        "max_depth": trial.suggest_categorical(
            "MaxDepth",
            [None, 6, 8, 10, 12, 14, 16, 20, 24, 30, 40],
        ),
        "min_samples_split": trial.suggest_int(
            "MinSamplesSplit", 2, 20
        ),
        "min_samples_leaf": trial.suggest_int(
            "MinLeafSize", 1, 10
        ),
        "max_features": trial.suggest_int(
            "NumPredictors", 1, 8
        ),
        "bootstrap": True,
        "max_samples": trial.suggest_categorical(
            "MaxSamples",
            [None, 0.6, 0.7, 0.8, 0.9, 1.0],
        ),
        "criterion": "squared_error",
        "random_state": 42,
        "n_jobs": -1,
    }

    mean_rmse, std_rmse, fold_rmse = cv_rmse(params)

    history.append({
        "Trial": trial.number,
        "Tree_Num": params["n_estimators"],
        "MaxDepth": params["max_depth"],
        "MinSamplesSplit": params["min_samples_split"],
        "MinLeafSize": params["min_samples_leaf"],
        "NumPredictors": params["max_features"],
        "MaxSamples": params["max_samples"],
        "Fold1_RMSE": fold_rmse[0],
        "Fold2_RMSE": fold_rmse[1],
        "Fold3_RMSE": fold_rmse[2],
        "Fold4_RMSE": fold_rmse[3],
        "Fold5_RMSE": fold_rmse[4],
        "Mean_CV_RMSE": mean_rmse,
        "Std_CV_RMSE": std_rmse,
    })

    return mean_rmse


sampler = optuna.samplers.TPESampler(
    seed=42,
    n_startup_trials=40,
    multivariate=True,
)

study = optuna.create_study(
    direction="minimize",
    sampler=sampler,
)

study.optimize(
    objective,
    n_trials=300,
    show_progress_bar=True,
)


best = study.best_params

bayesian_params = {
    "n_estimators": best["Tree_Num"],
    "max_depth": best["MaxDepth"],
    "min_samples_split": best["MinSamplesSplit"],
    "min_samples_leaf": best["MinLeafSize"],
    "max_features": best["NumPredictors"],
    "bootstrap": True,
    "max_samples": best["MaxSamples"],
    "criterion": "squared_error",
    "random_state": 42,
    "n_jobs": -1,
}

bayesian_mean_cv, bayesian_std_cv, bayesian_folds = cv_rmse(
    bayesian_params
)

print()
print("Full 5-fold Bayesian optimization result")
print(f"Tree_Num        = {best['Tree_Num']}")
print(f"MaxDepth        = {best['MaxDepth']}")
print(f"MinSamplesSplit = {best['MinSamplesSplit']}")
print(f"MinLeafSize     = {best['MinLeafSize']}")
print(f"NumPredictors   = {best['NumPredictors']}")
print(f"MaxSamples      = {best['MaxSamples']}")
print(f"Best mean CV RMSE = {bayesian_mean_cv:.6f}")
print(f"Best CV RMSE std  = {bayesian_std_cv:.6f}")

print()
print("Paper optimized RF parameters")
print("Tree_Num = 65")
print("MinLeafSize = 1")
print("NumPredictors = 8")

if bayesian_mean_cv < baseline_mean_cv:
    selected_name = "Bayesian RF"
else:
    selected_name = "Original tuned RF"

print()
print(f"Selected by CV = {selected_name}")


baseline_final = RandomForestRegressor(**baseline_params)
bayesian_final = RandomForestRegressor(**bayesian_params)

baseline_final.fit(X_dev, y_dev)
bayesian_final.fit(X_dev, y_dev)

pred_baseline = baseline_final.predict(X_test)
pred_bayesian = bayesian_final.predict(X_test)

metrics_baseline = evaluate(
    y_test,
    pred_baseline,
)

metrics_bayesian = evaluate(
    y_test,
    pred_bayesian,
)

metrics_df = pd.DataFrame([
    {
        "Model": "Original tuned RF",
        "CV_RMSE": baseline_mean_cv,
        "CV_RMSE_STD": baseline_std_cv,
        **metrics_baseline,
    },
    {
        "Model": "Bayesian RF",
        "CV_RMSE": bayesian_mean_cv,
        "CV_RMSE_STD": bayesian_std_cv,
        **metrics_bayesian,
    },
])

print()
print("Final Test results")
print(metrics_df.to_string(index=False))

print()
if metrics_bayesian["RMSE"] < metrics_baseline["RMSE"]:
    print("Bayesian RF improved Test performance.")
else:
    print("Bayesian RF did not improve Test performance.")

print("Model selection was based on CV, not Test.")


history_df = pd.DataFrame(history)

history_df["Best_CV_RMSE_so_far"] = (
    history_df["Mean_CV_RMSE"].cummin()
)

history_df.to_excel(
    output_dir / "15_TPE_search_history.xlsx",
    index=False,
)

metrics_df.to_excel(
    output_dir / "15_TPE_test_metrics.xlsx",
    index=False,
)

prediction_df = test_df.copy()
prediction_df["Pred_Original_RF"] = pred_baseline
prediction_df["Pred_Bayesian_RF"] = pred_bayesian

prediction_df.to_excel(
    output_dir / "15_TPE_predictions.xlsx",
    index=False,
)

joblib.dump(
    baseline_final,
    output_dir / "15_Original_RF_refit.joblib",
)

joblib.dump(
    bayesian_final,
    output_dir / "15_Bayesian_RF_TPE.joblib",
)


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Times New Roman",
    "DejaVu Serif",
]

fig, ax = plt.subplots(figsize=(7.5, 5.0))

x = np.arange(1, len(history_df) + 1)

ax.scatter(
    x,
    history_df["Mean_CV_RMSE"],
    s=12,
    alpha=0.45,
    label="Observed CV RMSE",
)

ax.plot(
    x,
    history_df["Best_CV_RMSE_so_far"],
    linewidth=1.5,
    label="Best CV RMSE",
)

ax.axhline(
    baseline_mean_cv,
    linestyle="--",
    linewidth=1.0,
    label="Original tuned RF",
)

ax.set_xlabel("Trial")
ax.set_ylabel("Mean 5-fold CV RMSE")
ax.legend(frameon=False)
ax.grid(alpha=0.25)

fig.tight_layout()

fig.savefig(
    output_dir / "15_TPE_convergence.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print()
print(f"Saved to: {output_dir}")
