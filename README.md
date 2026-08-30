# 论文复现

本项目用于复现论文中基于 Supporting Information 数据的材料寿命（`Lifetime`）预测流程。代码覆盖原始数据提取与审计、异常值处理、归一化、Vanilla GAN 数据增强、多种回归模型比较，以及 SHAP 特征解释和 Tb3/CGW 泛化分析。运行结果统一保存在 `results/`。

## 项目结构

```text
data/
  supporting_information.docx   # 论文补充材料（输入）
code/                            # 按阶段编号的 Python 脚本
results/                         # 中间数据、模型、表格和图片（输出）
```

主要输入特征为 `a`、`b`、`c`、`V`、`concentration`、`Layer`、`valence_electron` 和 `IQE`，预测目标为 `Lifetime`。

## 环境安装

建议使用 Python 3.10+ 的虚拟环境：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib openpyxl python-docx \
            torch joblib xgboost shap optuna
```

## 运行流程

在项目根目录执行，脚本中的相对路径均以根目录为基准：

```bash
python code/00_extract_and_audit.py
python code/01_outlier_diagnosis.py
python code/02_normalization_and_heatmap.py   # 可选：全量数据 baseline
python code/04_clean_and_normalize.py
python code/07_gan_optimize.py                # 推荐的 GAN 调参与增强流程
python code/08_split_dataset_recovered.py
```

`code/05_gan_final.py` 是另一套经典 Vanilla GAN 实现，可作为 `07` 的替代方案，不需要与 `07` 同时运行。GAN 训练耗时较长；调试时可设置 `GAN_QUICK=1`（仅 `07` 支持）。

完成数据划分后，可按需运行模型脚本：

```bash
python code/09_mlr.py       # 线性/正则化/GAM：09_*.py
python code/10_lstm.py      # BP/CNN/GRU/LSTM：10_*.py
python code/11_rf.py        # DT/RF/XGBoost/LSBoost：11_*.py
python code/12_gkr.py      # GKR/SVM/GPR/GRNN：12_*.py
python code/13_top5.py
python code/14_shap.py
python code/15_rf_revised.py
python code/16.py
python code/17_tb3.py
python code/18.py
```

各模型脚本会在 `results/09_1_MLR`、`results/10_1_GRU` 等对应目录中保存模型、预测值、评价指标和图像。评价指标通常包括 MAE、MSE、RMSE、MAPE 和 R²。

## 复现结果

仓库中已有的 `results/` 文件包含部分预生成的 Excel、PNG、Joblib 和 PyTorch 模型文件，可直接用于检查结果或跳过相应计算阶段。由于随机种子、PyTorch 版本和硬件差异，重新训练时数值结果可能存在小幅变化。
