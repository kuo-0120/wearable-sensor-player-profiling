# AI CUP 2025 穿戴式感測器球員特徵分類

由加速度計與陀螺儀序列預測球員的性別、持拍手、球齡與程度。專案重點是手工訊號特徵、自行實作 FFT，以及使用 player-level group split 避免同一球員同時出現在訓練與驗證資料造成 leakage。

## 技術重點

- 時域統計量、RMS、zero crossing、crest／waveform factor
- jerk、cross-axis interaction、smoothness
- 自行實作 FFT 與頻帶能量
- StratifiedGroupKFold／player-aware split
- class weight、SMOTE、Optuna
- XGBoost + LightGBM probability ensemble

## 執行

```powershell
$env:WEARABLE_DATA_ROOT = "D:\path\to\competition-data"
python -m venv .venv
pip install -r requirements.txt
python src/feature_pipeline_o3.py
```

資料根目錄預期包含 `39_Training_Dataset/`、`39_Test_Dataset/`、`tabular_data_train/` 與 `tabular_data_test/`。程式不會在執行時自動呼叫 pip；相依套件統一由 `requirements.txt` 管理。

## 檔案

- `src/feature_pipeline_o3.py`：含不平衡處理與 LightGBM ensemble 的擴充流程
- `src/train_ensemble.py`：FFT 特徵、Optuna 與四任務訓練流程

原始版本為研究型單檔 pipeline，為保留可追溯性暫不進行大規模行為改寫。下一步可再拆成 `features/`、`models/`、`train.py` 與 `predict.py`。

## 評估限制

目前資料夾沒有可獨立驗證的最終 AUC、排行榜截圖或正式排名紀錄，因此 README 不宣稱最終分數。重新執行後應保存每一任務 AUC、平均 AUC、split seed 與 leaderboard 證據。競賽原始資料、模型檔與 submission 均未收錄。
