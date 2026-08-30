from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PAPER_R2 = 0.94476

project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"

train_path = results_dir / "08_train_final.xlsx"
val_path = results_dir / "08_validation_final.xlsx"

output_dir = results_dir / "11_1_RF"
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

for n_estimators in [65, 100, 200, 500, 800, 1200]:
    for max_depth in [None, 8, 12, 16, 20, 30]:
        for min_samples_split in [2, 3, 4, 6]:
            for min_samples_leaf in [1, 2]:
                for max_features in [0.5, 0.75, 1.0]:
                    for max_samples in [None, 0.8, 0.9]:

                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features,
                            bootstrap=True,
                            max_samples=max_samples,
                            criterion="squared_error",
                            random_state=42,
                            n_jobs=-1,
                        )

                        model.fit(X_train, y_train)
                        pred = model.predict(X_val)
                        metrics = evaluate(y_val, pred)

                        row = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "min_samples_leaf": min_samples_leaf,
                            "max_features": max_features,
                            "max_samples": max_samples,
                            **metrics,
                        }

                        records.append(row)

                        if metrics["RMSE"] < best_rmse:
                            best_rmse = metrics["RMSE"]
                            best_model = model
                            best_pred = pred
                            best_result = row


result = pd.DataFrame([{
    "Model": "Random Forest",
    **best_result,
    "Paper_R2": PAPER_R2,
    "R2_Difference": best_result["R2"] - PAPER_R2,
}])

print()
print(result.to_string(index=False))


result.to_excel(
    output_dir / "11_1_RF_metrics.xlsx",
    index=False,
)

pd.DataFrame(records).sort_values("RMSE").to_excel(
    output_dir / "11_1_RF_tuning.xlsx",
    index=False,
)

prediction_df = val_df.copy()
prediction_df["Pred_RF"] = best_pred

prediction_df.to_excel(
    output_dir / "11_1_RF_predictions.xlsx",
    index=False,
)

joblib.dump(
    best_model,
    output_dir / "11_1_RF.joblib",
)