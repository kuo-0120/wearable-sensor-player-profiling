from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv
import os

DATA_ROOT = Path(os.getenv("WEARABLE_DATA_ROOT", "data"))
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import optuna
from xgboost import XGBClassifier
from scipy import stats
from sklearn.utils.class_weight import compute_sample_weight

# ==== 新增參數設定 ====
# 資料不平衡處理設定
USE_SAMPLE_WEIGHTING = True    # 是否使用樣本權重平衡類別
OVERSAMPLE_METHOD = 'SMOTE'    # 選擇過抽樣方法（例如 'SMOTE' 或 None 不使用）
# 模型集成設定
USE_LIGHTGBM = True            # 是否使用 LightGBM 作為集成模型的一部分

# ==== 嘗試載入 SMOTE 過抽樣方法 ====
SMOTE = None
if OVERSAMPLE_METHOD is not None:
    try:
        from imblearn.over_sampling import SMOTE as ImblearnSMOTE
        SMOTE = ImblearnSMOTE
    except ImportError:
        print("Warning: install requirements.txt to enable SMOTE; continuing without it.")
        SMOTE = None
        OVERSAMPLE_METHOD = None

# ==== 嘗試載入 LightGBM ====
LGBMClassifier = None
if USE_LIGHTGBM:
    try:
        import lightgbm as lgb
        LGBMClassifier = lgb.LGBMClassifier
    except ImportError:
        print("Warning: install requirements.txt to enable LightGBM; using XGBoost only.")
        LGBMClassifier = None
        USE_LIGHTGBM = False

# 全域常數：原始特徵列表（擴充新版）
ORIGINAL_HEADER_LIST = [
    'ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean',
    'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var',
    'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms',
    'jerk_ax_mean', 'jerk_ay_mean', 'jerk_az_mean',
    'jerk_ax_var', 'jerk_ay_var', 'jerk_az_var',
    'jerk_gx_mean', 'jerk_gy_mean', 'jerk_gz_mean',
    'jerk_gx_var', 'jerk_gy_var', 'jerk_gz_var',
    'zero_crossing_ax', 'zero_crossing_ay', 'zero_crossing_az',
    'waveform_factor_ax', 'waveform_factor_ay', 'waveform_factor_az',
    'crest_factor_ax', 'crest_factor_ay', 'crest_factor_az',
    'ax_band_energy_1', 'ax_band_energy_2', 'ax_band_energy_3',
    'ay_band_energy_1', 'ay_band_energy_2', 'ay_band_energy_3',
    'az_band_energy_1', 'az_band_energy_2', 'az_band_energy_3',
    'gx_band_energy_1', 'gx_band_energy_2', 'gx_band_energy_3',
    'gy_band_energy_1', 'gy_band_energy_2', 'gy_band_energy_3',
    'gz_band_energy_1', 'gz_band_energy_2', 'gz_band_energy_3',
    'cross_ax_ay_mean', 'cross_ax_az_mean', 'cross_ay_az_mean',
    'cross_gx_gy_mean', 'cross_gx_gz_mean', 'cross_gy_gz_mean',
    'cross_ax_gx_mean', 'cross_ay_gy_mean', 'cross_az_gz_mean',
    'accel_mag_mean', 'accel_mag_var', 'accel_mag_rms',
    'gyro_mag_mean', 'gyro_mag_var', 'gyro_mag_rms',
    'smoothness_ax', 'smoothness_ay', 'smoothness_az',
    'smoothness_gx', 'smoothness_gy', 'smoothness_gz',
    'ax_skew', 'ay_skew', 'az_skew', 'gx_skew', 'gy_skew', 'gz_skew',
    'ax_kurtosis', 'ay_kurtosis', 'az_kurtosis', 'gx_kurtosis', 'gy_kurtosis', 'gz_kurtosis'
]
NUM_ORIGINAL_FEATURES = len(ORIGINAL_HEADER_LIST)

# --- FFT 函數（維持原實現） ---
def FFT(xreal, ximag):
    n = 2
    while n * 2 <= len(xreal):
        n *= 2
    if n == 0 and len(xreal) > 0:
        n = 1
    elif n == 0 and len(xreal) == 0:
        return 0, [], []
    xreal_padded = list(xreal)
    ximag_padded = list(ximag)
    if len(xreal_padded) < n:
        xreal_padded.extend([0.0] * (n - len(xreal_padded)))
        ximag_padded.extend([0.0] * (n - len(ximag_padded)))
    else:
        xreal_padded = xreal_padded[:n]
        ximag_padded = ximag_padded[:n]
    p = 0
    if n > 1:
        p = int(math.log(n, 2))
    # 位逆序置換
    for i in range(n):
        a = i; b = 0
        for _ in range(p):
            b = (b << 1) | (a & 1)
            a >>= 1
        if b > i:
            xreal_padded[i], xreal_padded[b] = xreal_padded[b], xreal_padded[i]
            ximag_padded[i], ximag_padded[b] = ximag_padded[b], ximag_padded[i]
    # 蝴蝶操作
    if n > 1:
        for stage in range(1, p + 1):
            m = 1 << stage
            half_m = m >> 1
            angle_step = -2.0 * math.pi / m
            for k in range(0, n, m):
                w_real = 1.0; w_imag = 0.0
                cos_step = math.cos(angle_step); sin_step = math.sin(angle_step)
                for j in range(half_m):
                    idx1 = k + j
                    idx2 = idx1 + half_m
                    t_real = w_real * xreal_padded[idx2] - w_imag * ximag_padded[idx2]
                    t_imag = w_real * ximag_padded[idx2] + w_imag * xreal_padded[idx2]
                    u_real = xreal_padded[idx1]; u_imag = ximag_padded[idx1]
                    xreal_padded[idx1] = u_real + t_real
                    ximag_padded[idx1] = u_imag + t_imag
                    xreal_padded[idx2] = u_real - t_real
                    ximag_padded[idx2] = u_imag - t_imag
                    if j < half_m - 1:
                        next_w_real = w_real * cos_step - w_imag * sin_step
                        next_w_imag = w_real * sin_step + w_imag * cos_step
                        w_real = next_w_real; w_imag = next_w_imag
    return n, xreal_padded, ximag_padded

# --- Feature 提取函數（加入新特徵計算） ---
def feature(input_data):
    if not input_data or len(input_data) == 0:
        return [0.0] * NUM_ORIGINAL_FEATURES

    # 原始資料列表
    mean_vals, var_vals, rms_vals = [], [], []
    ax, ay, az, gx, gy, gz = [], [], [], [], [], []
    jerk_ax, jerk_ay, jerk_az = [], [], []
    jerk_gx, jerk_gy, jerk_gz = [], [], []  # 新增：陀螺儀軸差分
    cross_ax_ay, cross_ax_az, cross_ay_az = [], [], []  # 新增：各軸交叉項
    cross_gx_gy, cross_gx_gz, cross_gy_gz = [], [], []
    cross_ax_gx, cross_ay_gy, cross_az_gz = [], [], []

    # 迭代處理每行 segment 資料
    for item in input_data:
        ax.append(item[0]); ay.append(item[1]); az.append(item[2])
        gx.append(item[3]); gy.append(item[4]); gz.append(item[5])
        if len(item) >= 2:
            cross_ax_ay.append(item[0] * item[1])
        if len(item) >= 3:
            cross_ax_az.append(item[0] * item[2])
            cross_ay_az.append(item[1] * item[2])
        if len(item) >= 5:
            cross_gx_gy.append(item[3] * item[4])
        if len(item) >= 6:
            cross_gx_gz.append(item[3] * item[5])
            cross_gy_gz.append(item[4] * item[5])
            cross_ax_gx.append(item[0] * item[3])
            cross_ay_gy.append(item[1] * item[4])
            cross_az_gz.append(item[2] * item[5])

    # 計算加速度與陀螺儀的jerk（一階差分）
    if len(ax) > 1:
        for i in range(1, len(ax)):
            jerk_ax.append(ax[i] - ax[i-1])
            jerk_ay.append(ay[i] - ay[i-1])
            jerk_az.append(az[i] - az[i-1])
            jerk_gx.append(gx[i] - gx[i-1])
            jerk_gy.append(gy[i] - gy[i-1])
            jerk_gz.append(gz[i] - gz[i-1])

    # 基本統計特徵（均值、變異數、均方根）
    mean_vals = [
        np.mean(ax) if ax else 0.0, np.mean(ay) if ay else 0.0, np.mean(az) if az else 0.0,
        np.mean(gx) if gx else 0.0, np.mean(gy) if gy else 0.0, np.mean(gz) if gz else 0.0
    ]
    var_vals = [
        np.var(ax) if ax else 0.0, np.var(ay) if ay else 0.0, np.var(az) if az else 0.0,
        np.var(gx) if gx else 0.0, np.var(gy) if gy else 0.0, np.var(gz) if gz else 0.0
    ]
    rms_vals = [
        np.sqrt(np.mean(np.square(ax))) if ax else 0.0,
        np.sqrt(np.mean(np.square(ay))) if ay else 0.0,
        np.sqrt(np.mean(np.square(az))) if az else 0.0,
        np.sqrt(np.mean(np.square(gx))) if gx else 0.0,
        np.sqrt(np.mean(np.square(gy))) if gy else 0.0,
        np.sqrt(np.mean(np.square(gz))) if gz else 0.0
    ]

    # jerk 特徵（平均值與變異數）
    jerk_mean = [
        np.mean(jerk_ax) if jerk_ax else 0.0,
        np.mean(jerk_ay) if jerk_ay else 0.0,
        np.mean(jerk_az) if jerk_az else 0.0
    ]
    jerk_var = [
        np.var(jerk_ax) if jerk_ax else 0.0,
        np.var(jerk_ay) if jerk_ay else 0.0,
        np.var(jerk_az) if jerk_az else 0.0
    ]
    jerk_mean_gyro = [
        np.mean(jerk_gx) if jerk_gx else 0.0,
        np.mean(jerk_gy) if jerk_gy else 0.0,
        np.mean(jerk_gz) if jerk_gz else 0.0
    ]
    jerk_var_gyro = [
        np.var(jerk_gx) if jerk_gx else 0.0,
        np.var(jerk_gy) if jerk_gy else 0.0,
        np.var(jerk_gz) if jerk_gz else 0.0
    ]

    # 零交叉率
    def zero_crossing_rate(data_list):
        if not data_list or len(data_list) < 2:
            return 0.0
        signs = np.sign(data_list)
        return len(np.where(np.diff(signs))[0]) / (len(data_list) - 1)

    zero_crossing_ax = zero_crossing_rate(ax)
    zero_crossing_ay = zero_crossing_rate(ay)
    zero_crossing_az = zero_crossing_rate(az)

    # 波形因子與峰值因子
    def waveform_factor(data_list):
        if not data_list or len(data_list) == 0:
            return 0.0
        data_arr = np.array(data_list)
        rms_val = np.sqrt(np.mean(np.square(data_arr)))
        abs_mean_val = np.mean(np.abs(data_arr))
        return rms_val / abs_mean_val if abs_mean_val != 0 else 0.0

    def crest_factor(data_list):
        if not data_list or len(data_list) == 0:
            return 0.0
        data_arr = np.array(data_list)
        abs_max_val = np.max(np.abs(data_arr)) if data_arr.size > 0 else 0.0
        rms_val = np.sqrt(np.mean(np.square(data_arr)))
        return abs_max_val / rms_val if rms_val != 0 else 0.0

    waveform_factor_ax = waveform_factor(ax)
    waveform_factor_ay = waveform_factor(ay)
    waveform_factor_az = waveform_factor(az)
    crest_factor_ax = crest_factor(ax)
    crest_factor_ay = crest_factor(ay)
    crest_factor_az = crest_factor(az)

    # 頻帶能量特徵（FFT 能量分布）
    def frequency_band_energy_segment(signal_data, num_bands=3):
        if not signal_data:
            return [0.0] * num_bands
        signal_imag = [0.0] * len(signal_data)
        n_fft, fft_real, fft_imag = FFT(list(signal_data), signal_imag)
        if n_fft == 0 or not fft_real or not fft_imag:
            return [0.0] * num_bands
        actual_len = len(fft_real)
        if actual_len == 0:
            return [0.0] * num_bands
        # 直接使用整段 FFT 結果計算 PSD
        psd = [math.pow(fft_real[i], 2) + math.pow(fft_imag[i], 2) for i in range(actual_len)]
        total_psd_sum = sum(psd)
        if total_psd_sum == 0:
            return [0.0] * num_bands
        band_energy = []
        meaningful_fft_len = n_fft // 2 if n_fft > 0 else 0
        if meaningful_fft_len == 0:
            if actual_len > 0:
                band_size_approx = actual_len // num_bands
                if band_size_approx == 0:
                    energies = []
                    for i in range(num_bands):
                        if i < actual_len:
                            energies.append(psd[i] / total_psd_sum)
                        else:
                            energies.append(0.0)
                    return energies
                else:
                    meaningful_fft_len = actual_len
        band_size = meaningful_fft_len // num_bands
        if band_size == 0 and meaningful_fft_len > 0:
            for i in range(num_bands):
                if i < meaningful_fft_len:
                    band_energy.append(psd[i] / total_psd_sum)
                else:
                    band_energy.append(0.0)
            return band_energy
        elif band_size == 0 and meaningful_fft_len == 0:
            return [0.0] * num_bands
        for i in range(num_bands):
            start_index = i * band_size
            end_index = (i + 1) * band_size
            if i == num_bands - 1:
                end_index = meaningful_fft_len
            current_band_psd_sum = sum(psd[min(start_index, meaningful_fft_len-1) : min(end_index, meaningful_fft_len)])
            band_energy.append(current_band_psd_sum / total_psd_sum)
            if start_index >= meaningful_fft_len and len(band_energy) <= i:
                band_energy.append(0.0)
        while len(band_energy) < num_bands:
            band_energy.append(0.0)
        return band_energy[:num_bands]

    ax_band_energy = frequency_band_energy_segment(ax, num_bands=3)
    ay_band_energy = frequency_band_energy_segment(ay, num_bands=3)
    az_band_energy = frequency_band_energy_segment(az, num_bands=3)
    gx_band_energy = frequency_band_energy_segment(gx, num_bands=3)
    gy_band_energy = frequency_band_energy_segment(gy, num_bands=3)
    gz_band_energy = frequency_band_energy_segment(gz, num_bands=3)

    # 差分信號平滑度（jerk 的 RMS）
    smoothness = [
        np.sqrt(np.mean(np.square(jerk_ax))) if jerk_ax else 0.0,
        np.sqrt(np.mean(np.square(jerk_ay))) if jerk_ay else 0.0,
        np.sqrt(np.mean(np.square(jerk_az))) if jerk_az else 0.0,
        np.sqrt(np.mean(np.square(jerk_gx))) if jerk_gx else 0.0,
        np.sqrt(np.mean(np.square(jerk_gy))) if jerk_gy else 0.0,
        np.sqrt(np.mean(np.square(jerk_gz))) if jerk_gz else 0.0
    ]

    # 偏度與峰度（為避免極端情況，用安全函數）
    def is_valid_for_moment(data):
        if len(data) < 3:
            return False
        if np.std(data) < 1e-10:
            return False
        return True

    def safe_skew(data):
        return stats.skew(data) if is_valid_for_moment(data) else 0.0

    def safe_kurtosis(data):
        return stats.kurtosis(data, fisher=True) if is_valid_for_moment(data) else 0.0

    skew_vals = [
        safe_skew(ax), safe_skew(ay), safe_skew(az),
        safe_skew(gx), safe_skew(gy), safe_skew(gz)
    ]
    kurtosis_vals = [
        safe_kurtosis(ax), safe_kurtosis(ay), safe_kurtosis(az),
        safe_kurtosis(gx), safe_kurtosis(gy), safe_kurtosis(gz)
    ]

    # 交叉特徵平均值（捕捉軸間非線性交互）
    cross_means = [
        np.mean(cross_ax_ay) if cross_ax_ay else 0.0,
        np.mean(cross_ax_az) if cross_ax_az else 0.0,
        np.mean(cross_ay_az) if cross_ay_az else 0.0,
        np.mean(cross_gx_gy) if cross_gx_gy else 0.0,
        np.mean(cross_gx_gz) if cross_gx_gz else 0.0,
        np.mean(cross_gy_gz) if cross_gy_gz else 0.0,
        np.mean(cross_ax_gx) if cross_ax_gx else 0.0,
        np.mean(cross_ay_gy) if cross_ay_gy else 0.0,
        np.mean(cross_az_gz) if cross_az_gz else 0.0
    ]

    # 加速度與陀螺儀的向量幅值特徵
    if ax:
        accel_mag = np.sqrt(np.array(ax)**2 + np.array(ay)**2 + np.array(az)**2)
        accel_mag_mean = np.mean(accel_mag); accel_mag_var = np.var(accel_mag)
        accel_mag_rms = np.sqrt(np.mean(np.square(accel_mag)))
    else:
        accel_mag_mean = accel_mag_var = accel_mag_rms = 0.0
    if gx:
        gyro_mag = np.sqrt(np.array(gx)**2 + np.array(gy)**2 + np.array(gz)**2)
        gyro_mag_mean = np.mean(gyro_mag); gyro_mag_var = np.var(gyro_mag)
        gyro_mag_rms = np.sqrt(np.mean(np.square(gyro_mag)))
    else:
        gyro_mag_mean = gyro_mag_var = gyro_mag_rms = 0.0
    mag_vals = [accel_mag_mean, accel_mag_var, accel_mag_rms, gyro_mag_mean, gyro_mag_var, gyro_mag_rms]

    # 組合所有特徵輸出
    output = (
        mean_vals + var_vals + rms_vals +
        jerk_mean + jerk_var + jerk_mean_gyro + jerk_var_gyro +
        [zero_crossing_ax, zero_crossing_ay, zero_crossing_az] +
        [waveform_factor_ax, waveform_factor_ay, waveform_factor_az] +
        [crest_factor_ax, crest_factor_ay, crest_factor_az] +
        ax_band_energy + ay_band_energy + az_band_energy + gx_band_energy + gy_band_energy + gz_band_energy +
        cross_means + mag_vals + smoothness + skew_vals + kurtosis_vals
    )
    return list(output)

# --- XGBoost 多分類模型（years） ---
def years_model_multiary(X_tr, y_tr_enc, X_te, y_te_enc, params, sample_weights=None):
    num_classes_train = len(np.unique(y_tr_enc))
    clf = XGBClassifier(**params, random_state=42, eval_metric='mlogloss')
    if sample_weights is not None:
        clf.fit(X_tr, y_tr_enc, sample_weight=sample_weights, eval_set=[(X_te, y_te_enc)], verbose=False)
    else:
        clf.fit(X_tr, y_tr_enc, eval_set=[(X_te, y_te_enc)], verbose=False)
    predicted_proba = clf.predict_proba(X_te)
    try:
        if len(np.unique(y_te_enc)) < 2:
            return 0.5, clf
        num_classes_true_val = len(np.unique(y_te_enc))
        num_classes_pred_val = predicted_proba.shape[1]
        if num_classes_pred_val < num_classes_true_val and num_classes_true_val > 1:
            auc_score = 0.5
        else:
            auc_score = roc_auc_score(y_te_enc, predicted_proba, average='weighted', multi_class='ovr')
    except ValueError:
        auc_score = 0.5
    return auc_score, clf

# --- XGBoost 二元分類模型（gender / hold） ---
def gender_model_binary(X_tr, y_tr_enc, X_te, y_te_enc, params, sample_weights=None):
    # GPU 加速：若已安裝 GPU 版 XGBoost，建議加上 tree_method='gpu_hist'
    clf = XGBClassifier(
        **params,
        random_state=42,
        eval_metric='auc',
        tree_method='hist',      # 若尚未安裝 GPU 版，可先拿掉
        device='cuda'   # 同上
    )
    if sample_weights is not None:
        clf.fit(X_tr, y_tr_enc, sample_weight=sample_weights,
                eval_set=[(X_te, y_te_enc)], verbose=False)
    else:
        clf.fit(X_tr, y_tr_enc, eval_set=[(X_te, y_te_enc)], verbose=False)

    proba = clf.predict_proba(X_te)[:, 1]
    # 若驗證集只有單一類別，roc_auc_score 會 ValueError
    auc = 0.5
    if len(np.unique(y_te_enc)) > 1:
        auc = roc_auc_score(y_te_enc, proba)
    return auc, clf


def hold_model_binary(X_tr, y_tr_enc, X_te, y_te_enc, params, sample_weights=None):
    clf = XGBClassifier(
        **params,
        random_state=42,
        eval_metric='auc',
        tree_method='hist',
        device='cuda',
        predictor='gpu_predictor'
    )
    if sample_weights is not None:
        clf.fit(X_tr, y_tr_enc, sample_weight=sample_weights,
                eval_set=[(X_te, y_te_enc)], verbose=False)
    else:
        clf.fit(X_tr, y_tr_enc, eval_set=[(X_te, y_te_enc)], verbose=False)

    proba = clf.predict_proba(X_te)[:, 1]
    auc = 0.5
    if len(np.unique(y_te_enc)) > 1:
        auc = roc_auc_score(y_te_enc, proba)
    return auc, clf


# --- XGBoost 多分類模型（level） ---
def level_model_multiary(X_tr, y_tr_enc, X_te, y_te_enc, params, sample_weights=None):
    num_classes_train = len(np.unique(y_tr_enc))
    clf = XGBClassifier(tree_method='hist', device='cuda',**params, random_state=42, eval_metric='mlogloss')
    if sample_weights is not None:
        clf.fit(X_tr, y_tr_enc, sample_weight=sample_weights, eval_set=[(X_te, y_te_enc)], verbose=False)
    else:
        clf.fit(X_tr, y_tr_enc, eval_set=[(X_te, y_te_enc)], verbose=False)
    predicted_proba = clf.predict_proba(X_te)
    try:
        if len(np.unique(y_te_enc)) < 2:
            return 0.5, clf
        num_classes_true_val = len(np.unique(y_te_enc))
        num_classes_pred_val = predicted_proba.shape[1]
        if num_classes_pred_val < num_classes_true_val and num_classes_true_val > 1:
            auc_score = 0.5
        else:
            auc_score = roc_auc_score(y_te_enc, predicted_proba, average='weighted', multi_class='ovr')
    except ValueError:
        auc_score = 0.5
    return auc_score, clf

# --- LightGBM 多分類模型（years） ---
def years_model_multiary_lgb(X_tr, y_tr_enc, X_te, y_te_enc, params, sample_weights=None):
    # 移除 XGBoost 特有的參數
    params_lgb = params.copy()
    for key in ['gamma', 'min_child_weight']:
        params_lgb.pop(key, None)
    # 訓練 LightGBM 模型
    clf = LGBMClassifier(tree_method='hist', device='cuda',**params_lgb, random_state=42) if LGBMClassifier else None
    if clf is None:
        # 若未能載入 LightGBM，返回0.5
        return 0.5, None
    if sample_weights is not None:
        clf.fit(X_tr, y_tr_enc, sample_weight=sample_weights, eval_set=[(X_te, y_te_enc)], verbose=False)
    else:
        clf.fit(X_tr, y_tr_enc, eval_set=[(X_te, y_te_enc)], verbose=False)
    predicted_proba = clf.predict_proba(X_te)
    try:
        if len(np.unique(y_te_enc)) < 2:
            return 0.5, clf
        # LightGBM 輸出維度應為全類別數
        auc_score = roc_auc_score(y_te_enc, predicted_proba, average='weighted', multi_class='ovr')
    except ValueError:
        auc_score = 0.5
    return auc_score, clf

# --- LightGBM 多分類模型（level） ---
def level_model_multiary_lgb(X_tr, y_tr_enc, X_te, y_te_enc, params, sample_weights=None):
    params_lgb = params.copy()
    for key in ['gamma', 'min_child_weight']:
        params_lgb.pop(key, None)
    clf = LGBMClassifier(**params_lgb, random_state=42) if LGBMClassifier else None
    if clf is None:
        return 0.5, None
    if sample_weights is not None:
        clf.fit(X_tr, y_tr_enc, sample_weight=sample_weights, eval_set=[(X_te, y_te_enc)], verbose=False)
    else:
        clf.fit(X_tr, y_tr_enc, eval_set=[(X_te, y_te_enc)], verbose=False)
    predicted_proba = clf.predict_proba(X_te)
    try:
        if len(np.unique(y_te_enc)) < 2:
            return 0.5, clf
        auc_score = roc_auc_score(y_te_enc, predicted_proba, average='weighted', multi_class='ovr')
    except ValueError:
        auc_score = 0.5
    return auc_score, clf

# --- Optuna 目標函數（支援 StratifiedGroupKFold 與資料不平衡處理） ---
def get_optuna_objective_function(model_fn, X_train, y_train, *extra_sets, groups=None, n_splits=5,
                                  use_sample_weighting=True, oversample_method=None):
    from sklearn.model_selection import StratifiedKFold
    def _slice(arr, idx):
        try:
            return arr.iloc[idx]
        except AttributeError:
            return arr[idx]
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 2.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 500, 4000)
        }
        # 使用分層（群組）K折驗證
        if groups is not None:
            splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
            splits = splitter.split(X_train, y_train, groups)
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = splitter.split(X_train, y_train)
        aucs = []
        for tr_idx, val_idx in splits:
            X_tr = _slice(X_train, tr_idx); X_val = _slice(X_train, val_idx)
            y_tr = _slice(y_train, tr_idx); y_val = _slice(y_train, val_idx)
            # 資料不平衡處理：SMOTE 過抽樣或樣本權重
            if oversample_method is not None and SMOTE is not None:
                try:
                    X_tr_res, y_tr_res = SMOTE().fit_resample(np.array(X_tr), np.array(y_tr))
                except Exception as e:
                    X_tr_res, y_tr_res = X_tr, y_tr  # 無法過抽樣則使用原資料
                sample_w = None
            else:
                X_tr_res, y_tr_res = X_tr, y_tr
                sample_w = compute_sample_weight(class_weight='balanced', y=y_tr_res) if use_sample_weighting else None
            auc, _ = model_fn(X_tr_res, y_tr_res, X_val, y_val, params, sample_weights=sample_w)
            aucs.append(auc)
        return float(np.mean(aucs))
    return objective

# --- 主流程 ---
def main():
    # 生成訓練與測試的聚合特徵資料
    print("Generating training features...")
    generate_aggregated_features(DATA_ROOT / "39_Training_Dataset" / "train_data",
                                 DATA_ROOT / "tabular_data_train", is_test=False)
    print("Generating test features...")
    generate_aggregated_features(DATA_ROOT / "39_Test_Dataset" / "test_data",
                                 DATA_ROOT / "tabular_data_test", is_test=True)
    print("Feature generation complete.")

    # 讀取訓練資料資訊與標籤
    info_df = pd.read_csv(DATA_ROOT / "39_Training_Dataset" / "train_info.csv")
    info_df['play years'] = info_df['play years'].astype(str)
    info_df['level'] = info_df['level'].astype(str)
    # 更新聚合後特徵欄位名稱
    agg_stats_options = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'cv']
    aggregated_feature_names = [f"{col}_{stat}" for col in ORIGINAL_HEADER_LIST for stat in agg_stats_options]

    # 匯入全部訓練特徵與標籤
    all_features_list = []
    all_labels_list = []
    player_ids_for_split = []
    for _, row in info_df.iterrows():
        unique_id = row['unique_id']; player_id = row['player_id']
        feature_file_path = (DATA_ROOT / "tabular_data_train" / f"{unique_id}.csv")
        if feature_file_path.exists() and feature_file_path.stat().st_size > 0:
            data = pd.read_csv(feature_file_path)
            if not data.empty:
                if list(data.columns) == aggregated_feature_names:
                    all_features_list.append(data.iloc[0].values)
                    labels = row[['gender', 'hold racket handed', 'play years', 'level']].values
                    all_labels_list.append(labels)
                    player_ids_for_split.append(player_id)
    if not all_features_list:
        print("No training data loaded. Exiting.")
        return
    X_all = pd.DataFrame(all_features_list, columns=aggregated_feature_names)
    y_all_df = pd.DataFrame(all_labels_list, columns=['gender', 'hold racket handed', 'play years', 'level'])
    unique_player_ids_in_data = sorted(set(player_ids_for_split))
    if len(unique_player_ids_in_data) < 2:
        print("Not enough unique players in data for splitting. Exiting.")
        return

    # 拆分部分資料供 Optuna 調參（按選手分組）
    train_pids, val_pids = train_test_split(unique_player_ids_in_data, test_size=0.2, random_state=42)
    train_indices = [i for i, pid_val in enumerate(player_ids_for_split) if pid_val in train_pids]
    val_indices = [i for i, pid_val in enumerate(player_ids_for_split) if pid_val in val_pids]
    if not train_indices or not val_indices:
        print("Train/validation split failed. Exiting.")
        return
    x_train_df_optuna = X_all.iloc[train_indices]
    y_train_df_optuna = y_all_df.iloc[train_indices]
    x_val_df_optuna = X_all.iloc[val_indices]
    y_val_df_optuna = y_all_df.iloc[val_indices]

    # 標準化特徵
    scaler = MinMaxScaler()
    X_train_scaled_optuna = scaler.fit_transform(x_train_df_optuna)
    X_val_scaled_optuna = scaler.transform(x_val_df_optuna)
    # 標籤編碼（分別對訓練部分資料）
    le_gender = LabelEncoder()
    y_train_le_gender_optuna = le_gender.fit_transform(y_train_df_optuna['gender'])
    y_val_le_gender_optuna = le_gender.transform(y_val_df_optuna['gender'])
    le_hold = LabelEncoder()
    y_train_le_hold_optuna = le_hold.fit_transform(y_train_df_optuna['hold racket handed'])
    y_val_le_hold_optuna = le_hold.transform(y_val_df_optuna['hold racket handed'])
    le_years = LabelEncoder()
    y_train_le_years_optuna = le_years.fit_transform(y_train_df_optuna['play years'])
    y_val_le_years_optuna = le_years.transform(y_val_df_optuna['play years'])
    print(f"Play Years classes after LabelEncoding: {le_years.classes_}")
    le_level = LabelEncoder()
    y_train_le_level_optuna = le_level.fit_transform(y_train_df_optuna['level'])
    y_val_le_level_optuna = le_level.transform(y_val_df_optuna['level'])
    print(f"Level classes after LabelEncoding: {le_level.classes_}")

    # Optuna 參數搜尋
    optuna_trials = 70
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    USE_SAMPLE_WEIGHTING_OPTUNA = True

    # 使用 Optuna 調整各任務模型超參數
    print("Optimizing hyperparameters for gender...")
    study_gender = optuna.create_study(direction='maximize')
    study_gender.optimize(get_optuna_objective_function(gender_model_binary, X_train_scaled_optuna, # type: ignore
                                                        y_train_le_gender_optuna, groups=[player_ids_for_split[i] for i in train_indices],
                                                        use_sample_weighting=USE_SAMPLE_WEIGHTING_OPTUNA,
                                                        oversample_method=OVERSAMPLE_METHOD),
                          n_trials=optuna_trials, n_jobs=-1)
    best_params_gender = study_gender.best_params
    final_auc_gender, model_gender = gender_model_binary(X_train_scaled_optuna, y_train_le_gender_optuna, # type: ignore
                                                        X_val_scaled_optuna, y_val_le_gender_optuna,
                                                        best_params_gender,
                                                        sample_weights=compute_sample_weight('balanced', y=y_train_le_gender_optuna) if USE_SAMPLE_WEIGHTING_OPTUNA else None)
    print(f"Best params for gender: {best_params_gender}, Best Optuna Val AUC: {study_gender.best_value:.4f}, Final Val AUC: {final_auc_gender:.4f}")

    print("Optimizing hyperparameters for hold...")
    study_hold = optuna.create_study(direction='maximize')
    study_hold.optimize(get_optuna_objective_function(hold_model_binary, X_train_scaled_optuna, # type: ignore
                                                      y_train_le_hold_optuna, groups=[player_ids_for_split[i] for i in train_indices],
                                                      use_sample_weighting=USE_SAMPLE_WEIGHTING_OPTUNA,
                                                      oversample_method=OVERSAMPLE_METHOD),
                        n_trials=optuna_trials, n_jobs=-1)
    best_params_hold = study_hold.best_params
    final_auc_hold, model_hold = hold_model_binary(X_train_scaled_optuna, y_train_le_hold_optuna, # type: ignore
                                                  X_val_scaled_optuna, y_val_le_hold_optuna,
                                                  best_params_hold,
                                                  sample_weights=compute_sample_weight('balanced', y=y_train_le_hold_optuna) if USE_SAMPLE_WEIGHTING_OPTUNA else None)
    print(f"Best params for hold: {best_params_hold}, Best Optuna Val AUC: {study_hold.best_value:.4f}, Final Val AUC: {final_auc_hold:.4f}")

    print("Optimizing hyperparameters for years (XGBoost)...")
    study_years = optuna.create_study(direction='maximize')
    study_years.optimize(get_optuna_objective_function(years_model_multiary, X_train_scaled_optuna,
                                                       y_train_le_years_optuna, groups=[player_ids_for_split[i] for i in train_indices],
                                                       use_sample_weighting=USE_SAMPLE_WEIGHTING_OPTUNA,
                                                       oversample_method=OVERSAMPLE_METHOD),
                         n_trials=optuna_trials, n_jobs=-1)
    best_params_year = study_years.best_params
    final_auc_years, model_years_xgb = years_model_multiary(X_train_scaled_optuna, y_train_le_years_optuna,
                                                            X_val_scaled_optuna, y_val_le_years_optuna,
                                                            best_params_year,
                                                            sample_weights=compute_sample_weight('balanced', y=y_train_le_years_optuna) if USE_SAMPLE_WEIGHTING_OPTUNA else None)
    print(f"Best params for years (XGB): {best_params_year}, Best Optuna Val AUC: {study_years.best_value:.4f}, Final Val AUC: {final_auc_years:.4f}")

    model_years_lgb = None; final_auc_years_lgb = 0.0
    if USE_LIGHTGBM and LGBMClassifier is not None:
        print("Optimizing hyperparameters for years (LightGBM)...")
        study_years_lgb = optuna.create_study(direction='maximize')
        study_years_lgb.optimize(get_optuna_objective_function(years_model_multiary_lgb, X_train_scaled_optuna,
                                                               y_train_le_years_optuna, groups=[player_ids_for_split[i] for i in train_indices],
                                                               use_sample_weighting=USE_SAMPLE_WEIGHTING_OPTUNA,
                                                               oversample_method=OVERSAMPLE_METHOD),
                                 n_trials=int(optuna_trials/2), n_jobs=-1)
        best_params_year_lgb = study_years_lgb.best_params
        final_auc_years_lgb, model_years_lgb = years_model_multiary_lgb(X_train_scaled_optuna, y_train_le_years_optuna,
                                                                        X_val_scaled_optuna, y_val_le_years_optuna,
                                                                        best_params_year_lgb,
                                                                        sample_weights=compute_sample_weight('balanced', y=y_train_le_years_optuna) if USE_SAMPLE_WEIGHTING_OPTUNA else None)
        print(f"Best params for years (LGB): {best_params_year_lgb}, Best Optuna Val AUC: {study_years_lgb.best_value:.4f}, Final Val AUC: {final_auc_years_lgb:.4f}")

    print("Optimizing hyperparameters for level (XGBoost)...")
    study_level = optuna.create_study(direction='maximize')
    study_level.optimize(get_optuna_objective_function(level_model_multiary, X_train_scaled_optuna,
                                                       y_train_le_level_optuna, groups=[player_ids_for_split[i] for i in train_indices],
                                                       use_sample_weighting=USE_SAMPLE_WEIGHTING_OPTUNA,
                                                       oversample_method=OVERSAMPLE_METHOD),
                         n_trials=optuna_trials, n_jobs=-1)
    best_params_level = study_level.best_params
    final_auc_level, model_level_xgb = level_model_multiary(X_train_scaled_optuna, y_train_le_level_optuna,
                                                            X_val_scaled_optuna, y_val_le_level_optuna,
                                                            best_params_level,
                                                            sample_weights=compute_sample_weight('balanced', y=y_train_le_level_optuna) if USE_SAMPLE_WEIGHTING_OPTUNA else None)
    print(f"Best params for level (XGB): {best_params_level}, Best Optuna Val AUC: {study_level.best_value:.4f}, Final Val AUC: {final_auc_level:.4f}")

    model_level_lgb = None; final_auc_level_lgb = 0.0
    if USE_LIGHTGBM and LGBMClassifier is not None:
        print("Optimizing hyperparameters for level (LightGBM)...")
        study_level_lgb = optuna.create_study(direction='maximize')
        study_level_lgb.optimize(get_optuna_objective_function(level_model_multiary_lgb, X_train_scaled_optuna,
                                                               y_train_le_level_optuna, groups=[player_ids_for_split[i] for i in train_indices],
                                                               use_sample_weighting=USE_SAMPLE_WEIGHTING_OPTUNA,
                                                               oversample_method=OVERSAMPLE_METHOD),
                                 n_trials=int(optuna_trials/2), n_jobs=-1)
        best_params_level_lgb = study_level_lgb.best_params
        final_auc_level_lgb, model_level_lgb = level_model_multiary_lgb(X_train_scaled_optuna, y_train_le_level_optuna,
                                                                        X_val_scaled_optuna, y_val_le_level_optuna,
                                                                        best_params_level_lgb,
                                                                        sample_weights=compute_sample_weight('balanced', y=y_train_le_level_optuna) if USE_SAMPLE_WEIGHTING_OPTUNA else None)
        print(f"Best params for level (LGB): {best_params_level_lgb}, Best Optuna Val AUC: {study_level_lgb.best_value:.4f}, Final Val AUC: {final_auc_level_lgb:.4f}")

    # 集成模型對 years 和 level 驗證集的最終 AUC
    if USE_LIGHTGBM and model_years_lgb is not None:
        # 平均 XGBoost 與 LightGBM 概率
        years_proba_xgb = model_years_xgb.predict_proba(X_val_scaled_optuna)
        years_proba_lgb = model_years_lgb.predict_proba(X_val_scaled_optuna)
        years_proba_avg = (years_proba_xgb + years_proba_lgb) / 2
        ensemble_auc_years = roc_auc_score(y_val_le_years_optuna, years_proba_avg, average='weighted', multi_class='ovr')
        final_auc_years = ensemble_auc_years  # 用集成結果作為最終 AUC
        level_proba_xgb = model_level_xgb.predict_proba(X_val_scaled_optuna)
        level_proba_lgb = model_level_lgb.predict_proba(X_val_scaled_optuna)
        level_proba_avg = (level_proba_xgb + level_proba_lgb) / 2
        ensemble_auc_level = roc_auc_score(y_val_le_level_optuna, level_proba_avg, average='weighted', multi_class='ovr')
        final_auc_level = ensemble_auc_level
        print(f"Ensemble final Val AUC - years: {ensemble_auc_years:.4f}, level: {ensemble_auc_level:.4f}")

    final_avg_auc = (final_auc_gender + final_auc_hold + final_auc_years + final_auc_level) / 4
    print(f"Final Average Validation AUC: {final_avg_auc:.4f}")

    # ================= 交叉驗證 (Stratified K-fold) 評估模型性能 =================
    print("\nPerforming stratified K-fold cross-validation...")
    # 準備全域 LabelEncoder（使用整個訓練資料）
    le_gender_all = LabelEncoder().fit(y_all_df['gender'])
    le_hold_all = LabelEncoder().fit(y_all_df['hold racket handed'])
    le_years_all = LabelEncoder().fit(y_all_df['play years'])
    le_level_all = LabelEncoder().fit(y_all_df['level'])
    # 轉換所有樣本標籤為數值
    y_gender_all_enc = le_gender_all.transform(y_all_df['gender'])
    y_hold_all_enc = le_hold_all.transform(y_all_df['hold racket handed'])
    y_years_all_enc = le_years_all.transform(y_all_df['play years'])
    y_level_all_enc = le_level_all.transform(y_all_df['level'])
    X_all_values = X_all.values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 逐一任務計算 K-fold AUC
    # Gender
    y_true_all = []; y_score_all = []
    for train_idx, test_idx in sgkf.split(X_all_values, y_gender_all_enc, groups=player_ids_for_split):
        X_tr = X_all_values[train_idx]; X_te = X_all_values[test_idx]
        y_tr = y_gender_all_enc[train_idx]; y_te = y_gender_all_enc[test_idx]
        if len(np.unique(y_tr)) < 2:
            probs = np.full((len(y_te), 2), 0.5)  # 單一類別訓練則預測0.5機率
        else:
            scaler_cv = MinMaxScaler().fit(X_tr)
            X_tr_scaled = scaler_cv.transform(X_tr); X_te_scaled = scaler_cv.transform(X_te)
            if OVERSAMPLE_METHOD is not None and SMOTE is not None:
                try:
                    X_tr_res, y_tr_res = SMOTE().fit_resample(X_tr_scaled, y_tr)
                except Exception:
                    X_tr_res, y_tr_res = X_tr_scaled, y_tr
                sample_w_cv = None
            else:
                X_tr_res, y_tr_res = X_tr_scaled, y_tr
                sample_w_cv = compute_sample_weight('balanced', y_tr_res) if USE_SAMPLE_WEIGHTING else None
            clf_cv = XGBClassifier(**best_params_gender, random_state=42, eval_metric='mlogloss')
            if sample_w_cv is not None:
                clf_cv.fit(X_tr_res, y_tr_res, sample_weight=sample_w_cv)
            else:
                clf_cv.fit(X_tr_res, y_tr_res)
            probs = clf_cv.predict_proba(X_te_scaled)
        y_true_all.append(y_te); y_score_all.append(probs)
    y_true_all = np.concatenate(y_true_all); y_score_all = np.vstack(y_score_all)
    class_auc = roc_auc_score(y_true_all, y_score_all, average=None, multi_class='ovr')
    macro_auc = roc_auc_score(y_true_all, y_score_all, average='macro', multi_class='ovr')
    micro_auc = roc_auc_score(y_true_all, y_score_all, average='micro', multi_class='ovr')
    print(f"Gender - Macro AUC: {macro_auc:.3f}, Micro AUC: {micro_auc:.3f}, "
          f"Class {le_gender_all.classes_[0]} AUC: {class_auc[0]:.3f}, Class {le_gender_all.classes_[1]} AUC: {class_auc[1]:.3f}")

    # Hold racket handed
    y_true_all = []; y_score_all = []
    for train_idx, test_idx in sgkf.split(X_all_values, y_hold_all_enc, groups=player_ids_for_split):
        X_tr = X_all_values[train_idx]; X_te = X_all_values[test_idx]
        y_tr = y_hold_all_enc[train_idx]; y_te = y_hold_all_enc[test_idx]
        if len(np.unique(y_tr)) < 2:
            probs = np.full((len(y_te), 2), 0.5)
        else:
            scaler_cv = MinMaxScaler().fit(X_tr)
            X_tr_scaled = scaler_cv.transform(X_tr); X_te_scaled = scaler_cv.transform(X_te)
            if OVERSAMPLE_METHOD is not None and SMOTE is not None:
                try:
                    X_tr_res, y_tr_res = SMOTE().fit_resample(X_tr_scaled, y_tr)
                except Exception:
                    X_tr_res, y_tr_res = X_tr_scaled, y_tr
                sample_w_cv = None
            else:
                X_tr_res, y_tr_res = X_tr_scaled, y_tr
                sample_w_cv = compute_sample_weight('balanced', y_tr_res) if USE_SAMPLE_WEIGHTING else None
            clf_cv = XGBClassifier(**best_params_hold, random_state=42, eval_metric='mlogloss')
            if sample_w_cv is not None:
                clf_cv.fit(X_tr_res, y_tr_res, sample_weight=sample_w_cv)
            else:
                clf_cv.fit(X_tr_res, y_tr_res)
            probs = clf_cv.predict_proba(X_te_scaled)
        y_true_all.append(y_te); y_score_all.append(probs)
    y_true_all = np.concatenate(y_true_all); y_score_all = np.vstack(y_score_all)
    class_auc = roc_auc_score(y_true_all, y_score_all, average=None, multi_class='ovr')
    macro_auc = roc_auc_score(y_true_all, y_score_all, average='macro', multi_class='ovr')
    micro_auc = roc_auc_score(y_true_all, y_score_all, average='micro', multi_class='ovr')
    print(f"Hold-Hand - Macro AUC: {macro_auc:.3f}, Micro AUC: {micro_auc:.3f}, "
          f"Class {le_hold_all.classes_[0]} AUC: {class_auc[0]:.3f}, Class {le_hold_all.classes_[1]} AUC: {class_auc[1]:.3f}")

    # Play years (使用 XGBoost + LightGBM 集成預測)
    y_true_all = []; y_score_all = []
    for train_idx, test_idx in sgkf.split(X_all_values, y_years_all_enc, groups=player_ids_for_split):
        X_tr = X_all_values[train_idx]; X_te = X_all_values[test_idx]
        y_tr = y_years_all_enc[train_idx]; y_te = y_years_all_enc[test_idx]
        if len(np.unique(y_tr)) < len(le_years_all.classes_):
            # 若某折缺少某個年數類別，用隨機預測
            probs = np.full((len(y_te), len(le_years_all.classes_)), 1.0/len(le_years_all.classes_))
        else:
            scaler_cv = MinMaxScaler().fit(X_tr)
            X_tr_scaled = scaler_cv.transform(X_tr); X_te_scaled = scaler_cv.transform(X_te)
            # 過抽樣
            if OVERSAMPLE_METHOD is not None and SMOTE is not None:
                try:
                    X_tr_res, y_tr_res = SMOTE().fit_resample(X_tr_scaled, y_tr)
                except Exception:
                    X_tr_res, y_tr_res = X_tr_scaled, y_tr
                sample_w_cv = None
            else:
                X_tr_res, y_tr_res = X_tr_scaled, y_tr
                sample_w_cv = compute_sample_weight('balanced', y_tr_res) if USE_SAMPLE_WEIGHTING else None
            # 訓練 XGBoost 模型
            clf_xgb = XGBClassifier(**best_params_year, random_state=42, eval_metric='mlogloss')
            if sample_w_cv is not None:
                clf_xgb.fit(X_tr_res, y_tr_res, sample_weight=sample_w_cv)
            else:
                clf_xgb.fit(X_tr_res, y_tr_res)
            # 訓練 LightGBM 模型（若可用）
            clf_lgb = None
            if USE_LIGHTGBM and LGBMClassifier is not None:
                params_lgb = best_params_year.copy()
                for key in ['gamma', 'min_child_weight']:
                    params_lgb.pop(key, None)
                clf_lgb = LGBMClassifier(**params_lgb, random_state=42)
                if sample_w_cv is not None:
                    clf_lgb.fit(X_tr_res, y_tr_res, sample_weight=sample_w_cv)
                else:
                    clf_lgb.fit(X_tr_res, y_tr_res)
            # 集成預測
            proba_xgb = clf_xgb.predict_proba(X_te_scaled)
            if clf_lgb is not None:
                proba_lgb = clf_lgb.predict_proba(X_te_scaled)
                probs = (proba_xgb + proba_lgb) / 2
            else:
                probs = proba_xgb
        y_true_all.append(y_te); y_score_all.append(probs)
    y_true_all = np.concatenate(y_true_all); y_score_all = np.vstack(y_score_all)
    class_auc = roc_auc_score(y_true_all, y_score_all, average=None, multi_class='ovr')
    macro_auc = roc_auc_score(y_true_all, y_score_all, average='macro', multi_class='ovr')
    micro_auc = roc_auc_score(y_true_all, y_score_all, average='micro', multi_class='ovr')
    # 輸出各類 AUC，類別名稱用原標籤值
    year_labels = le_years_all.classes_
    print(f"PlayYears - Macro AUC: {macro_auc:.3f}, Micro AUC: {micro_auc:.3f}, "
          + ", ".join([f"Class {lbl} AUC: {auc:.3f}" for lbl, auc in zip(year_labels, class_auc)]))

    # Level (使用 XGBoost + LightGBM 集成預測)
    y_true_all = []; y_score_all = []
    for train_idx, test_idx in sgkf.split(X_all_values, y_level_all_enc, groups=player_ids_for_split):
        X_tr = X_all_values[train_idx]; X_te = X_all_values[test_idx]
        y_tr = y_level_all_enc[train_idx]; y_te = y_level_all_enc[test_idx]
        if len(np.unique(y_tr)) < len(le_level_all.classes_):
            probs = np.full((len(y_te), len(le_level_all.classes_)), 1.0/len(le_level_all.classes_))
        else:
            scaler_cv = MinMaxScaler().fit(X_tr)
            X_tr_scaled = scaler_cv.transform(X_tr); X_te_scaled = scaler_cv.transform(X_te)
            if OVERSAMPLE_METHOD is not None and SMOTE is not None:
                try:
                    X_tr_res, y_tr_res = SMOTE().fit_resample(X_tr_scaled, y_tr)
                except Exception:
                    X_tr_res, y_tr_res = X_tr_scaled, y_tr
                sample_w_cv = None
            else:
                X_tr_res, y_tr_res = X_tr_scaled, y_tr
                sample_w_cv = compute_sample_weight('balanced', y_tr_res) if USE_SAMPLE_WEIGHTING else None
            clf_xgb = XGBClassifier(**best_params_level, random_state=42, eval_metric='mlogloss')
            if sample_w_cv is not None:
                clf_xgb.fit(X_tr_res, y_tr_res, sample_weight=sample_w_cv)
            else:
                clf_xgb.fit(X_tr_res, y_tr_res)
            clf_lgb = None
            if USE_LIGHTGBM and LGBMClassifier is not None:
                params_lgb = best_params_level.copy()
                for key in ['gamma', 'min_child_weight']:
                    params_lgb.pop(key, None)
                clf_lgb = LGBMClassifier(**params_lgb, random_state=42)
                if sample_w_cv is not None:
                    clf_lgb.fit(X_tr_res, y_tr_res, sample_weight=sample_w_cv)
                else:
                    clf_lgb.fit(X_tr_res, y_tr_res)
            proba_xgb = clf_xgb.predict_proba(X_te_scaled)
            if clf_lgb is not None:
                proba_lgb = clf_lgb.predict_proba(X_te_scaled)
                probs = (proba_xgb + proba_lgb) / 2
            else:
                probs = proba_xgb
        y_true_all.append(y_te); y_score_all.append(probs)
    y_true_all = np.concatenate(y_true_all); y_score_all = np.vstack(y_score_all)
    class_auc = roc_auc_score(y_true_all, y_score_all, average=None, multi_class='ovr')
    macro_auc = roc_auc_score(y_true_all, y_score_all, average='macro', multi_class='ovr')
    micro_auc = roc_auc_score(y_true_all, y_score_all, average='micro', multi_class='ovr')
    level_labels = le_level_all.classes_
    print(f"Level - Macro AUC: {macro_auc:.3f}, Micro AUC: {micro_auc:.3f}, "
          + ", ".join([f"Class {lbl} AUC: {auc:.3f}" for lbl, auc in zip(level_labels, class_auc)]))
    print("Cross-validation evaluation completed.\n")

    # ================= 使用全部資料重新訓練最終模型並生成提交預測 =================
    # 使用所有訓練資料重新標準化
    scaler_all = MinMaxScaler().fit(X_all)
    X_all_scaled = scaler_all.transform(X_all)
    # 重新以所有資料貼標 LabelEncoder
    le_gender = LabelEncoder().fit(y_all_df['gender'])
    le_hold = LabelEncoder().fit(y_all_df['hold racket handed'])
    le_years = LabelEncoder().fit(y_all_df['play years'])
    le_level = LabelEncoder().fit(y_all_df['level'])
    y_gender_all_enc = le_gender.transform(y_all_df['gender'])
    y_hold_all_enc = le_hold.transform(y_all_df['hold racket handed'])
    y_years_all_enc = le_years.transform(y_all_df['play years'])
    y_level_all_enc = le_level.transform(y_all_df['level'])
    # 最終模型訓練（使用最佳參數）
    # Gender最終模型
    if OVERSAMPLE_METHOD is not None and SMOTE is not None:
        try:
            X_gender_res, y_gender_res = SMOTE().fit_resample(X_all_scaled, y_gender_all_enc)
        except Exception:
            X_gender_res, y_gender_res = X_all_scaled, y_gender_all_enc
        sample_w_gender = None
    else:
        X_gender_res, y_gender_res = X_all_scaled, y_gender_all_enc
        sample_w_gender = compute_sample_weight('balanced', y=y_gender_res) if USE_SAMPLE_WEIGHTING else None
    model_gender_full = XGBClassifier(**best_params_gender, random_state=42, eval_metric='mlogloss')
    if sample_w_gender is not None:
        model_gender_full.fit(X_gender_res, y_gender_res, sample_weight=sample_w_gender)
    else:
        model_gender_full.fit(X_gender_res, y_gender_res)

    # Hold最終模型
    if OVERSAMPLE_METHOD is not None and SMOTE is not None:
        try:
            X_hold_res, y_hold_res = SMOTE().fit_resample(X_all_scaled, y_hold_all_enc)
        except Exception:
            X_hold_res, y_hold_res = X_all_scaled, y_hold_all_enc
        sample_w_hold = None
    else:
        X_hold_res, y_hold_res = X_all_scaled, y_hold_all_enc
        sample_w_hold = compute_sample_weight('balanced', y=y_hold_res) if USE_SAMPLE_WEIGHTING else None
    model_hold_full = XGBClassifier(**best_params_hold, random_state=42, eval_metric='mlogloss')
    if sample_w_hold is not None:
        model_hold_full.fit(X_hold_res, y_hold_res, sample_weight=sample_w_hold)
    else:
        model_hold_full.fit(X_hold_res, y_hold_res)

    # Years最終模型（XGB + LGB ensemble）
    if OVERSAMPLE_METHOD is not None and SMOTE is not None:
        try:
            X_years_res, y_years_res = SMOTE().fit_resample(X_all_scaled, y_years_all_enc)
        except Exception:
            X_years_res, y_years_res = X_all_scaled, y_years_all_enc
        sample_w_years = None
    else:
        X_years_res, y_years_res = X_all_scaled, y_years_all_enc
        sample_w_years = compute_sample_weight('balanced', y=y_years_res) if USE_SAMPLE_WEIGHTING else None
    model_years_xgb_full = XGBClassifier(**best_params_year, random_state=42, eval_metric='mlogloss')
    if sample_w_years is not None:
        model_years_xgb_full.fit(X_years_res, y_years_res, sample_weight=sample_w_years)
    else:
        model_years_xgb_full.fit(X_years_res, y_years_res)
    model_years_lgb_full = None
    if USE_LIGHTGBM and LGBMClassifier is not None:
        params_year_lgb = best_params_year.copy()
        for key in ['gamma', 'min_child_weight']:
            params_year_lgb.pop(key, None)
        model_years_lgb_full = LGBMClassifier(**params_year_lgb, random_state=42)
        if sample_w_years is not None:
            model_years_lgb_full.fit(X_years_res, y_years_res, sample_weight=sample_w_years)
        else:
            model_years_lgb_full.fit(X_years_res, y_years_res)
    # Level最終模型（XGB + LGB ensemble）
    if OVERSAMPLE_METHOD is not None and SMOTE is not None:
        try:
            X_level_res, y_level_res = SMOTE().fit_resample(X_all_scaled, y_level_all_enc)
        except Exception:
            X_level_res, y_level_res = X_all_scaled, y_level_all_enc
        sample_w_level = None
    else:
        X_level_res, y_level_res = X_all_scaled, y_level_all_enc
        sample_w_level = compute_sample_weight('balanced', y=y_level_res) if USE_SAMPLE_WEIGHTING else None
    model_level_xgb_full = XGBClassifier(**best_params_level, random_state=42, eval_metric='mlogloss')
    if sample_w_level is not None:
        model_level_xgb_full.fit(X_level_res, y_level_res, sample_weight=sample_w_level)
    else:
        model_level_xgb_full.fit(X_level_res, y_level_res)
    model_level_lgb_full = None
    if USE_LIGHTGBM and LGBMClassifier is not None:
        params_level_lgb = best_params_level.copy()
        for key in ['gamma', 'min_child_weight']:
            params_level_lgb.pop(key, None)
        model_level_lgb_full = LGBMClassifier(**params_level_lgb, random_state=42)
        if sample_w_level is not None:
            model_level_lgb_full.fit(X_level_res, y_level_res, sample_weight=sample_w_level)
        else:
            model_level_lgb_full.fit(X_level_res, y_level_res)

    # 準備傳入 predict_test 的模型（集成模型用list傳入）
    model_gender = model_gender_full
    model_hold = model_hold_full
    model_years = [model_years_xgb_full, model_years_lgb_full] if (USE_LIGHTGBM and model_years_lgb_full is not None) else model_years_xgb_full
    model_level = [model_level_xgb_full, model_level_lgb_full] if (USE_LIGHTGBM and model_level_lgb_full is not None) else model_level_xgb_full

    # 預測測試集結果
    print("Predicting on test set...")
    results = predict_test(model_gender, model_hold, model_years, model_level,
                           scaler_all, le_gender, le_hold, le_years, le_level, aggregated_feature_names)
    save_submission(results, filename='submission_boosted.csv')
    print("Submission file 'submission_boosted.csv' created.")

# 儲存提交檔案的函數
def save_submission(results, filename='submission.csv'):
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for row in results:
                writer.writerow(row)
    except Exception as e:
        print(f"Error saving submission file {filename}: {e}")

# ------------------------------------------------------------
#  重新補上：聚合特徵產生函式  generate_aggregated_features(...)
# ------------------------------------------------------------
def generate_aggregated_features(datapath: str,
                                 tar_dir: str,
                                 is_test: bool = False,
                                 num_segments_target: int = 27):
    """
    讀取『一位選手一次揮拍』的 txt 原始六軸感測資料，
    每個檔案切成 num_segments_target 段，
    對每段呼叫 feature(...) 取特徵，再做 segment-level 聚合
    （mean / std / min / max / median / q25 / q75 / cv）。

    產出：每位選手一個 .csv (只有一列)，
          欄位即 aggregated_feature_names。
    """
    pathlist_txt = Path(datapath).glob('**/*.txt')
    os.makedirs(tar_dir, exist_ok=True)

    agg_stats_options = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'cv']
    new_headerList = [f"{col}_{stat}" for col in ORIGINAL_HEADER_LIST for stat in agg_stats_options]

    for file in pathlist_txt:
        all_data = []
        with open(file, 'r', encoding='utf-8') as f:
            for ln, line in enumerate(f):
                if ln == 0 or line.strip() == '':
                    continue                         # 跳 header / 空行
                nums = line.split(' ')
                if len(nums) >= 6:
                    try:
                        all_data.append([float(x) for x in nums[:6]])
                    except ValueError:
                        pass
        if len(all_data) < 2:
            continue                                # 資料不足，跳過

        # ---- 依規定等距切 27 段 ----
        idx_pts = np.linspace(0, len(all_data), num_segments_target+1, dtype=int)
        idx_pts = np.unique(idx_pts)
        if len(idx_pts) < 2:
            continue

        seg_feats = []
        for i in range(len(idx_pts)-1):
            seg = all_data[idx_pts[i]: idx_pts[i+1]]
            seg_feats.append(feature(seg))         # 用新版 feature()

        if not seg_feats:
            continue
        df_seg = pd.DataFrame(seg_feats, columns=ORIGINAL_HEADER_LIST)

        # ---- segment-level 聚合 ----
        agg_row = []
        for col in ORIGINAL_HEADER_LIST:
            col_vals = df_seg[col].dropna()
            if col_vals.empty:
                agg_row.extend([0.0] * len(agg_stats_options))
                continue
            m = col_vals.mean();  sd = col_vals.std();  mn = col_vals.min()
            mx = col_vals.max();  med = col_vals.median()
            q25 = col_vals.quantile(0.25);  q75 = col_vals.quantile(0.75)
            cv  = (sd / m) if m != 0 else 0.0
            agg_row.extend([m, sd, mn, mx, med, q25, q75, cv])

        if len(agg_row) != len(new_headerList):
            # 理論不會發生；若發生就補 0
            agg_row = (agg_row + [0.0]*len(new_headerList))[:len(new_headerList)]

        # ---- 輸出單列 csv ----
        out_csv = Path(tar_dir) / f"{file.stem}.csv"
        with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(new_headerList)
            writer.writerow(agg_row)


# --- 預測測試集資料 ---
def predict_test(model_gender, model_hand, model_years, model_level, scaler,
                 le_gender, le_hold, le_years, le_level, feature_names_passed):
    test_data_dir = './tabular_data_test'
    test_files = list(Path(test_data_dir).glob('*.csv'))
    results = []
    # 標題列
    submission_header = ["unique_id", "gender", "hold racket handed",
                         "play years_0", "play years_1", "play years_2",
                         "level_2", "level_3", "level_4", "level_5"]
    results.append(submission_header)
    # 預設機率（無法預測時使用均勻分佈）
    default_gender_prob = 0.5
    default_hand_prob = 0.5
    num_years_classes = len(le_years.classes_) if hasattr(le_years, 'classes_') else 3
    default_years_probs = [1.0/num_years_classes] * num_years_classes
    num_level_classes = len(le_level.classes_) if hasattr(le_level, 'classes_') else 4
    default_level_probs = [1.0/num_level_classes] * num_level_classes

    for file_path in test_files:
        df = pd.read_csv(file_path)
        if df.empty:
            # 空資料則輸出預設值
            uid = file_path.stem
            results.append([uid, default_gender_prob, default_hand_prob] + 
                           [round(p, 6) for p in default_years_probs] + 
                           [round(p, 6) for p in default_level_probs])
            continue
        X = df.values
        # 使用與訓練相同的標準化器轉換特徵
        X_scaled_player = scaler.transform(X)
        # 性別預測（取機率為 "男" 類別的機率）
        gender_pred_probs = model_gender.predict_proba(X_scaled_player)[0]
        try:
            gender_final_prob = gender_pred_probs[le_gender.transform(['M'])[0]]
        except Exception:
            gender_final_prob = default_gender_prob
        # 持拍手預測（取機率為 "右手" 類別的機率）
        hand_pred_probs = model_hand.predict_proba(X_scaled_player)[0]
        try:
            hand_final_prob = hand_pred_probs[le_hold.transform(['R'])[0]]
        except Exception:
            hand_final_prob = default_hand_prob
        # 遊玩年數預測（多類別，可能使用集成模型）
        if isinstance(model_years, list):
            # 集成：平均多個模型的機率
            years_probs_list = [m.predict_proba(X_scaled_player)[0] for m in model_years]
            years_pred_probs_raw = np.mean(years_probs_list, axis=0)
        else:
            years_pred_probs_raw = model_years.predict_proba(X_scaled_player)[0]
        final_years_probs_submission = []
        for year_label in ["0", "1", "2"]:
            try:
                enc_val = le_years.transform([year_label])[0]
                prob_val = years_pred_probs_raw[enc_val] if enc_val < len(years_pred_probs_raw) else 0.0
                final_years_probs_submission.append(prob_val)
            except Exception:
                final_years_probs_submission.append(0.0)
        # 正規化概率
        sum_years = sum(final_years_probs_submission)
        if sum_years > 1e-6:
            final_years_probs_submission = [p/sum_years for p in final_years_probs_submission]
        else:
            final_years_probs_submission = [1/3.0] * 3
        # 水平（等級）預測
        if isinstance(model_level, list):
            level_probs_list = [m.predict_proba(X_scaled_player)[0] for m in model_level]
            level_pred_probs_raw = np.mean(level_probs_list, axis=0)
        else:
            level_pred_probs_raw = model_level.predict_proba(X_scaled_player)[0]
        final_level_probs_submission = []
        for level_label in ["2", "3", "4", "5"]:
            try:
                enc_val = le_level.transform([level_label])[0]
                prob_val = level_pred_probs_raw[enc_val] if enc_val < len(level_pred_probs_raw) else 0.0
                final_level_probs_submission.append(prob_val)
            except Exception:
                final_level_probs_submission.append(0.0)
        sum_level = sum(final_level_probs_submission)
        if sum_level > 1e-6:
            final_level_probs_submission = [p/sum_level for p in final_level_probs_submission]
        else:
            final_level_probs_submission = [1/4.0] * 4
        # 構建輸出行
        unique_id = file_path.stem
        current_row = [unique_id, round(gender_final_prob, 6), round(hand_final_prob, 6)]
        current_row.extend([round(p, 6) for p in final_years_probs_submission])
        current_row.extend([round(p, 6) for p in final_level_probs_submission])
        results.append(current_row)
    return results

if __name__ == "__main__":
    main()
