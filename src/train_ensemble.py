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
from sklearn.utils.class_weight import compute_sample_weight # 新增，為了後續可能的樣本權重

# 全域常數：原始特徵列表
# 注意：頻帶能量特徵的計算邏輯會改變，但名稱保留
ORIGINAL_HEADER_LIST = [
    'ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean',
    'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var',
    'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms',
    'jerk_ax_mean', 'jerk_ay_mean', 'jerk_az_mean',
    'jerk_ax_var', 'jerk_ay_var', 'jerk_az_var',
    'zero_crossing_ax', 'zero_crossing_ay', 'zero_crossing_az',
    'waveform_factor_ax', 'waveform_factor_ay', 'waveform_factor_az',
    'crest_factor_ax', 'crest_factor_ay', 'crest_factor_az',
    'ax_band_energy_1', 'ax_band_energy_2', 'ax_band_energy_3', # 這些將由 segment 內部 FFT 計算
    'ay_band_energy_1', 'ay_band_energy_2', 'ay_band_energy_3', # 這些將由 segment 內部 FFT 計算
    'az_band_energy_1', 'az_band_energy_2', 'az_band_energy_3', # 這些將由 segment 內部 FFT 計算
    'cross_ax_ay_mean', 'cross_gx_gy_mean',
    'smoothness_ax', 'smoothness_ay', 'smoothness_az',
    'gx_skew', 'gy_skew', 'gz_skew',
    'gx_kurtosis', 'gy_kurtosis', 'gz_kurtosis'
]
NUM_ORIGINAL_FEATURES = len(ORIGINAL_HEADER_LIST)

# --- FFT 相關函數 ---
# 保持您原來的 FFT 函數，因為 feature 函數內部會用到它對 segment 數據進行FFT
def FFT(xreal, ximag):
    n = 2
    while n * 2 <= len(xreal):
        n *= 2
    if n == 0 and len(xreal) > 0:
        n = 1
    elif n == 0 and len(xreal) == 0:
        return 0, [], []

    xreal_padded = list(xreal) # 創建副本以避免修改原始列表
    ximag_padded = list(ximag)

    if len(xreal_padded) < n:
        xreal_padded.extend([0.0] * (n - len(xreal_padded)))
        ximag_padded.extend([0.0] * (n - len(ximag_padded)))
    else: # 如果原始長度大於n，截斷到n
        xreal_padded = xreal_padded[:n]
        ximag_padded = ximag_padded[:n]


    p = 0
    if n > 1:
        p = int(math.log(n, 2))

    # Bit-reversal permutation
    for i in range(n):
        a = i
        b = 0
        for _ in range(p):
            b = (b << 1) | (a & 1)
            a >>= 1
        if b > i:
            xreal_padded[i], xreal_padded[b] = xreal_padded[b], xreal_padded[i]
            ximag_padded[i], ximag_padded[b] = ximag_padded[b], ximag_padded[i]

    # Danielson-Lanczos section (Butterfly operations)
    if n > 1:
        for stage in range(1, p + 1):
            m = 1 << stage
            half_m = m >> 1
            angle_step = -2.0 * math.pi / m
            for k in range(0, n, m):
                w_real = 1.0
                w_imag = 0.0
                cos_step = math.cos(angle_step)
                sin_step = math.sin(angle_step)
                for j in range(half_m):
                    idx1 = k + j
                    idx2 = idx1 + half_m
                    t_real = w_real * xreal_padded[idx2] - w_imag * ximag_padded[idx2]
                    t_imag = w_real * ximag_padded[idx2] + w_imag * xreal_padded[idx2]
                    u_real = xreal_padded[idx1]
                    u_imag = ximag_padded[idx1]
                    xreal_padded[idx1] = u_real + t_real
                    ximag_padded[idx1] = u_imag + t_imag
                    xreal_padded[idx2] = u_real - t_real
                    ximag_padded[idx2] = u_imag - t_imag
                    if j < half_m - 1: # Avoid recalculating for the last iteration
                        next_w_real = w_real * cos_step - w_imag * sin_step
                        next_w_imag = w_real * sin_step + w_imag * cos_step
                        w_real = next_w_real
                        w_imag = next_w_imag
    return n, xreal_padded, ximag_padded

# --- FFT_data 函數被移除 ---
# 因為 FFT 現在在 feature 函數中針對每個 segment 執行

# --- feature 函數修改 ---
def feature(input_data): # 移除了 FFT 相關參數，因為現在內部計算
    if not input_data or len(input_data) == 0: # 確保 input_data 不是 None 也不是空列表
        return [0.0] * NUM_ORIGINAL_FEATURES

    mean_vals, var_vals, rms_vals = [], [], []
    ax, ay, az, gx, gy, gz = [], [], [], [], [], []
    jerk_ax, jerk_ay, jerk_az = [], [], []
    cross_ax_ay, cross_gx_gy = [], []

    for item in input_data:
        ax.append(item[0])
        ay.append(item[1])
        az.append(item[2])
        gx.append(item[3])
        gy.append(item[4])
        gz.append(item[5])
        if len(item) >= 2: # 應該是 item[0] 和 item[1] 存在
             cross_ax_ay.append(item[0] * item[1])
        if len(item) >= 5: # 應該是 item[3] 和 item[4] 存在
             cross_gx_gy.append(item[3] * item[4])


    if len(ax) > 1:
        for i in range(1, len(ax)):
            jerk_ax.append(ax[i] - ax[i-1])
            jerk_ay.append(ay[i] - ay[i-1])
            jerk_az.append(az[i] - az[i-1])

    mean_vals = [
        np.mean(ax) if ax else 0.0, np.mean(ay) if ay else 0.0, np.mean(az) if az else 0.0,
        np.mean(gx) if gx else 0.0, np.mean(gy) if gy else 0.0, np.mean(gz) if gz else 0.0
    ]
    var_vals = [
        np.var(ax) if ax else 0.0, np.var(ay) if ay else 0.0, np.var(az) if az else 0.0,
        np.var(gx) if gx else 0.0, np.var(gy) if gy else 0.0, np.var(gz) if gz else 0.0
    ]
    rms_vals = [
        np.sqrt(np.mean(np.square(ax))) if ax else 0.0, np.sqrt(np.mean(np.square(ay))) if ay else 0.0,
        np.sqrt(np.mean(np.square(az))) if az else 0.0, np.sqrt(np.mean(np.square(gx))) if gx else 0.0,
        np.sqrt(np.mean(np.square(gy))) if gy else 0.0, np.sqrt(np.mean(np.square(gz))) if gz else 0.0
    ]

    jerk_mean = [np.mean(jerk_ax) if jerk_ax else 0.0, np.mean(jerk_ay) if jerk_ay else 0.0, np.mean(jerk_az) if jerk_az else 0.0]
    jerk_var = [np.var(jerk_ax) if jerk_ax else 0.0, np.var(jerk_ay) if jerk_ay else 0.0, np.var(jerk_az) if jerk_az else 0.0]

    def zero_crossing_rate(data_list):
        if not data_list or len(data_list) < 2:
            return 0.0
        # data_arr = np.array(data_list) # 移到下面檢查 data_arr.size 之前
        # if data_arr.size == 0: # 這個檢查可以移除，因為上面已經有 len(data_list) < 2
        #     return 0.0
        signs = np.sign(data_list) # 直接對 list 操作 sign
        return len(np.where(np.diff(signs))[0]) / (len(data_list) - 1)

    zero_crossing_ax = zero_crossing_rate(ax)
    zero_crossing_ay = zero_crossing_rate(ay)
    zero_crossing_az = zero_crossing_rate(az)

    def waveform_factor(data_list):
        if not data_list:
            return 0.0
        data_arr = np.array(data_list)
        rms_val = np.sqrt(np.mean(np.square(data_arr)))
        abs_mean_val = np.mean(np.abs(data_arr))
        return rms_val / abs_mean_val if abs_mean_val != 0 else 0.0

    waveform_factor_ax = waveform_factor(ax)
    waveform_factor_ay = waveform_factor(ay)
    waveform_factor_az = waveform_factor(az)

    def crest_factor(data_list):
        if not data_list:
            return 0.0
        data_arr = np.array(data_list)
        abs_max_val = np.max(np.abs(data_arr)) if data_arr.size > 0 else 0.0
        rms_val = np.sqrt(np.mean(np.square(data_arr)))
        return abs_max_val / rms_val if rms_val != 0 else 0.0

    crest_factor_ax = crest_factor(ax)
    crest_factor_ay = crest_factor(ay)
    crest_factor_az = crest_factor(az)

    # --- 新的頻帶能量計算邏輯 ---
    def frequency_band_energy_segment(signal_data, num_bands=3):
        if not signal_data:
            return [0.0] * num_bands
        
        signal_imag = [0.0] * len(signal_data)
        n_fft, fft_real, fft_imag = FFT(list(signal_data), signal_imag) # 對當前 segment 的信號做 FFT

        if n_fft == 0 or not fft_real or not fft_imag:
            return [0.0] * num_bands

        actual_len = len(fft_real) # n_fft 是 padded length, actual_len 也是 padded length
        if actual_len == 0:
            return [0.0] * num_bands

        # 只取 FFT 結果的前半部分 (去除對稱部分)，通常到 n_fft // 2
        # 但由於您的 FFT 實現返回的是完整長度，且能量計算時會平方，我們這裡用完整長度
        # 不過，標準做法是只用非冗餘部分。如果您的 FFT 實現已經處理了，則無需切片。
        # 這裡假設您的 FFT 輸出可以直接用於 PSD 計算。
        
        psd = [math.pow(fft_real[i], 2) + math.pow(fft_imag[i], 2) for i in range(actual_len)]
        total_psd_sum = sum(psd)
        if total_psd_sum == 0:
            return [0.0] * num_bands

        band_energy = []
        # 我們只關心到 Nyquist 頻率，所以用 actual_len // 2。
        # 但為了與您原始的 `frequency_band_energy` 的 band_size 計算方式保持一定相似性，
        # 且您的 FFT 輸出是 n (padded_length)，我們將基於 n_fft 計算頻帶。
        # 一般來說，PSD 會在 n_fft/2 之後對稱 (對於實數輸入)。
        # 為了簡單起見，這裡我們先用整個 n_fft 長度來劃分頻帶，
        # 如果需要更精確的物理意義，通常只分析到 n_fft/2。
        meaningful_fft_len = n_fft // 2 if n_fft > 0 else 0 # 通常分析到奈奎斯特頻率
        if meaningful_fft_len == 0: # 如果 FFT 長度太短
             if actual_len > 0: # 如果原始 fft_real 有數據
                band_size_approx = actual_len // num_bands
                if band_size_approx == 0: # 如果整體數據點比 band 數還少
                    energies = []
                    for i in range(num_bands):
                        if i < actual_len:
                            energies.append(psd[i] / total_psd_sum)
                        else:
                            energies.append(0.0)
                    return energies
                else: # 均分
                    meaningful_fft_len = actual_len # 退回到用整個長度

        band_size = meaningful_fft_len // num_bands
        
        if band_size == 0 and meaningful_fft_len > 0: # 如果頻帶太窄，無法均分
            # 給每個band至少一個點，直到用完 meaningful_fft_len
            for i in range(num_bands):
                if i < meaningful_fft_len:
                    band_energy.append(psd[i] / total_psd_sum) # 只取單個點的能量
                else:
                    band_energy.append(0.0)
            return band_energy
        elif band_size == 0 and meaningful_fft_len == 0:
            return [0.0] * num_bands


        for i in range(num_bands):
            start_index = i * band_size
            end_index = (i + 1) * band_size
            if i == num_bands - 1: # 最後一個頻帶包含所有剩餘的點到 meaningful_fft_len
                end_index = meaningful_fft_len
            
            # 確保索引不越界
            current_band_psd_sum = sum(psd[min(start_index, meaningful_fft_len-1) : min(end_index, meaningful_fft_len)])
            band_energy.append(current_band_psd_sum / total_psd_sum)
            if start_index >= meaningful_fft_len and len(band_energy) <= i : # 如果起始索引已經超出，後面補0
                band_energy.append(0.0)


        # 確保返回 num_bands 個值
        while len(band_energy) < num_bands:
            band_energy.append(0.0)
        return band_energy[:num_bands]


    ax_band_energy = frequency_band_energy_segment(ax, num_bands=3)
    ay_band_energy = frequency_band_energy_segment(ay, num_bands=3)
    az_band_energy = frequency_band_energy_segment(az, num_bands=3)
    # gx, gy, gz 的頻帶能量也可以考慮加入 ORIGINAL_HEADER_LIST，如果需要的話
    # 目前 ORIGINAL_HEADER_LIST 只有 ax, ay, az 的頻帶能量

    smoothness = [
        np.sqrt(np.mean(np.square(jerk_ax))) if jerk_ax else 0.0,
        np.sqrt(np.mean(np.square(jerk_ay))) if jerk_ay else 0.0,
        np.sqrt(np.mean(np.square(jerk_az))) if jerk_az else 0.0
    ]

    def is_valid_for_moment(data):
        if len(data) < 3:
            return False
        if np.std(data) < 1e-10: # 避免除以零或極小標準差
            return False
        return True

    def safe_skew(data):
        return stats.skew(data) if is_valid_for_moment(data) else 0.0

    def safe_kurtosis(data):
        # scipy.stats.kurtosis 默認計算 Fisher (excess) kurtosis，即減3
        return stats.kurtosis(data, fisher=True) if is_valid_for_moment(data) else 0.0

    skew_vals = [
        safe_skew(gx), safe_skew(gy), safe_skew(gz)
    ]
    kurtosis_vals = [
        safe_kurtosis(gx), safe_kurtosis(gy), safe_kurtosis(gz)
    ]

    output = (
        mean_vals + var_vals + rms_vals + jerk_mean + jerk_var +
        [zero_crossing_ax, zero_crossing_ay, zero_crossing_az] +
        [waveform_factor_ax, waveform_factor_ay, waveform_factor_az] +
        [crest_factor_ax, crest_factor_ay, crest_factor_az] +
        ax_band_energy + ay_band_energy + az_band_energy + # 這裡的值來自新的計算方式
        [np.mean(cross_ax_ay) if cross_ax_ay else 0.0, np.mean(cross_gx_gy) if cross_gx_gy else 0.0] +
        smoothness + skew_vals + kurtosis_vals
    )

    return list(output)

# --- generate_aggregated_features 函數修改 ---
def generate_aggregated_features(datapath, tar_dir, is_test=False):
    pathlist_txt = Path(datapath).glob('**/*.txt')
    os.makedirs(tar_dir, exist_ok=True)

    # --- 新增聚合統計選項 ---
    agg_stats_options = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'cv']
    new_headerList = [f"{col}_{stat}" for col in ORIGINAL_HEADER_LIST for stat in agg_stats_options]

    for file in pathlist_txt:
        All_data = []
        try:
            with open(file, 'r', encoding='utf-8') as f:
                count = 0
                for line in f.readlines():
                    if line.strip() == '' or count == 0: # 忽略空行和第一行（假設是表頭）
                        count += 1
                        continue
                    num_str = line.split(' ')
                    if len(num_str) >= 6: # 確保至少有6個數據點
                        try:
                            # 假設數據是浮點數或可以轉換為浮點數
                            tmp_list = [float(s) for s in num_str[:6]]
                            All_data.append(tmp_list)
                        except ValueError:
                            # print(f"Warning: ValueError converting data in {file.name}, line: {line.strip()}")
                            pass # 跳過無法轉換的行
                    count += 1
        except Exception as e:
            print(f"Error reading file {file.name}: {e}")
            continue

        if not All_data:
            # print(f"Warning: No data loaded from {file.name}")
            continue
        
        if len(All_data) < 2: # 需要至少兩個數據點來形成一個 segment
            # print(f"Warning: Not enough data in {file.name} for segmentation (len: {len(All_data)})")
            continue


        # --- 分段邏輯不變 ---
        # 創建28個點，形成27個segment
        # 如果 All_data 長度小於28，linspace會處理，但可能導致 segment 過短或重複
        num_segments_target = 27
        # 確保 swing_index 的長度是 num_segments_target + 1
        # 並且 swing_index 的值不超過 All_data 的長度
        swing_index = np.linspace(0, len(All_data), num_segments_target + 1, dtype=int)
        # 修正linspace可能產生的重複索引問題，特別是當 len(All_data) < num_segments_target + 1
        swing_index = np.unique(swing_index)
        # 如果去重後點太少，無法形成足夠的segment，則跳過此文件
        if len(swing_index) < 2:
            # print(f"Warning: Not enough unique segment points in {file.name} after np.unique on swing_index.")
            continue

        segment_features_collection = []
        
        # --- FFT_data 函數調用被移除 ---
        # n_fft_val, ax_fft_res, etc. 也不再需要全局計算

        num_segments_actual = len(swing_index) - 1
        if num_segments_actual <= 0:
            # print(f"Warning: num_segments_actual is {num_segments_actual} for {file.name}. Skipping.")
            continue

        for i in range(num_segments_actual):
            segment_data = All_data[swing_index[i]:swing_index[i+1]]
            if not segment_data: # 如果 segment 為空則跳過
                # print(f"Warning: Empty segment {i} for file {file.name}. Skipping segment.")
                # 可以選擇填充0，或者跳過。如果跳過，後續聚合時該 segment 不計入。
                # 為了保持 segment_features_collection 的長度與 num_segments_actual 一致（如果需要固定長度），
                # 這裡可以添加一個全零的特徵向量。但如果允許變長，則可以直接continue。
                # 假設我們期望每個文件都有 num_segments_actual 個特徵集用於聚合
                # 如果 segment_data 為空，feature 函數會返回全零向量，所以直接調用即可。
                pass

            # --- feature 函數調用修改 ---
            current_segment_features = feature(segment_data)
            segment_features_collection.append(current_segment_features)

        if not segment_features_collection:
            # print(f"Warning: No segment features collected for {file.name}")
            continue

        df_segment_features = pd.DataFrame(segment_features_collection, columns=ORIGINAL_HEADER_LIST)

        aggregated_row = []
        for col_name in ORIGINAL_HEADER_LIST:
            feature_values_over_segments = df_segment_features[col_name].dropna() # Drop NaN before aggregation

            if feature_values_over_segments.empty:
                # 如果移除 NaN 後為空，則所有聚合統計量都設為0
                aggregated_row.extend([0.0] * len(agg_stats_options))
                continue

            mean_val = feature_values_over_segments.mean()
            std_val = feature_values_over_segments.std()
            min_val = feature_values_over_segments.min()
            max_val = feature_values_over_segments.max()
            median_val = feature_values_over_segments.median()
            q25_val = feature_values_over_segments.quantile(0.25)
            q75_val = feature_values_over_segments.quantile(0.75)
            
            # 計算 CV，處理 mean_val 為 0 的情況
            cv_val = (std_val / mean_val) if mean_val != 0 else 0.0
            # 如果 std_val 也是 0 (所有值相同)，cv_val 也應該是 0
            if std_val == 0 and mean_val !=0 : # 所有值相同且不為0
                cv_val = 0.0
            if mean_val == 0 and std_val == 0: # 所有值都是0
                 cv_val = 0.0


            aggregated_row.extend([
                mean_val, std_val, min_val, max_val, median_val,
                q25_val, q75_val, cv_val
            ])

        aggregated_row_cleaned = [0 if pd.isna(x) else x for x in aggregated_row]
        # 確保長度正確
        if len(aggregated_row_cleaned) != len(new_headerList):
            # print(f"Warning: Mismatch in feature length for {file.name}. Expected {len(new_headerList)}, got {len(aggregated_row_cleaned)}. Padding with zeros.")
            # 這種情況理論上不應發生，如果發生，說明 aggregated_row 的構建有問題
            # 但作為保險，可以填充或截斷
            if len(aggregated_row_cleaned) < len(new_headerList):
                 aggregated_row_cleaned.extend([0.0] * (len(new_headerList) - len(aggregated_row_cleaned)))
            else:
                 aggregated_row_cleaned = aggregated_row_cleaned[:len(new_headerList)]


        output_csv_path = Path(tar_dir) / f"{file.stem}.csv"
        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(new_headerList) # 使用新的 header
                writer.writerow(aggregated_row_cleaned)
        except Exception as e:
            print(f"Error writing CSV for {file.name}: {e}")

# --- data_generate 和 data_generate_test 不變 ---
def data_generate():
    print("Generating training features...")
    generate_aggregated_features(DATA_ROOT / "39_Training_Dataset" / "train_data", DATA_ROOT / "tabular_data_train", is_test=False)
    print("Training feature generation complete.")

def data_generate_test():
    print("Generating test features...")
    generate_aggregated_features(DATA_ROOT / "39_Test_Dataset" / "test_data", DATA_ROOT / "tabular_data_test", is_test=True)
    print("Test feature generation complete.")

# --- 模型訓練和預測相關函數 (gender_model_binary, hold_model_binary, etc.) ---
# --- 保持不變，但它們會接收到不同維度和內容的特徵 ---
 
# --- XGBoost 二元分類模型（gender / hold） ---
def gender_model_binary(X_tr, y_tr_enc, X_te, y_te_enc, params, sample_weights=None):
    # GPU 加速：若已安裝 GPU 版 XGBoost，建議加上 tree_method='gpu_hist'
    clf = XGBClassifier(
        **params,
        random_state=42,
        eval_metric='auc',
        tree_method='gpu_hist',      # 若尚未安裝 GPU 版，可先拿掉
        predictor='gpu_predictor'    # 同上
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
        tree_method='gpu_hist',
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


def years_model_multiary(X_tr, y_tr_enc, X_te, y_te_enc, params, sample_weights=None):
    num_classes_train = len(np.unique(y_tr_enc))
    clf = XGBClassifier(**params, random_state=42, eval_metric='mlogloss')
    # XGBoost 版本問題：如果 fit 方法不接受 early_stopping_rounds，請移除它
    if sample_weights is not None:
         clf.fit(X_tr, y_tr_enc, sample_weight=sample_weights, eval_set=[(X_te, y_te_enc)], verbose=False) # 移除了 early_stopping_rounds
    else:
        clf.fit(X_tr, y_tr_enc, eval_set=[(X_te, y_te_enc)], verbose=False) # 移除了 early_stopping_rounds

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
    except ValueError as e:
        auc_score = 0.5
    return auc_score, clf

def level_model_multiary(X_tr, y_tr_enc, X_te, y_te_enc, params, sample_weights=None):
    num_classes_train = len(np.unique(y_tr_enc))
    clf = XGBClassifier(**params, random_state=42, eval_metric='mlogloss')
    # XGBoost 版本問題：如果 fit 方法不接受 early_stopping_rounds，請移除它
    if sample_weights is not None:
        clf.fit(X_tr, y_tr_enc, sample_weight=sample_weights, eval_set=[(X_te, y_te_enc)], verbose=False) # 移除了 early_stopping_rounds
    else:
        clf.fit(X_tr, y_tr_enc, eval_set=[(X_te, y_te_enc)], verbose=False) # 移除了 early_stopping_rounds

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
    except ValueError as e:
        auc_score = 0.5
    return auc_score, clf

# --- Optuna 目標函數修改 (可選：傳入 y_tr_full 以計算樣本權重) ---

def get_optuna_objective_function(model_fn, X_train, y_train, *extra_sets, groups=None, n_splits=5, use_sample_weighting=True):
    """Create an Optuna objective using stratified (group) k‑fold.
    The *extra_sets parameter is ignored; it exists only for backward compatibility."""
    from sklearn.model_selection import StratifiedKFold
    def _slice(arr, idx):
        # Works for pandas as well as numpy
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
            'n_estimators': trial.suggest_int('n_estimators', 500, 4000),
        }
        if groups is not None:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = splitter.split(X_train, y_train, groups)
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = splitter.split(X_train, y_train)
        aucs = []
        for tr_idx, val_idx in splits:
            X_tr, X_val = _slice(X_train, tr_idx), _slice(X_train, val_idx)
            y_tr, y_val = _slice(y_train, tr_idx), _slice(y_train, val_idx)
            sample_w = compute_sample_weight(class_weight='balanced', y=y_tr) if use_sample_weighting else None
            auc, _ = model_fn(X_tr, y_tr, X_val, y_val, params, sample_weights=sample_w)
            aucs.append(auc)
        return float(np.mean(aucs))
    return objective



# --- predict_test 函數基本不變，但接收的 feature_names_passed 會是新的 ---
def predict_test(model_gender, model_hand, model_years, model_level, scaler,
                 le_gender, le_hold, le_years, le_level, feature_names_passed): # feature_names_passed 會是新的表頭
    test_data_dir = './tabular_data_test'
    test_files = list(Path(test_data_dir).glob('*.csv'))

    results = []
    # --- 更新提交文件的表頭以匹配預期格式 ---
    # 假設 'play years' 標籤編碼後是 0, 1, 2
    # 假設 'level' 標籤編碼後是 0, 1, 2, 3 (對應實際值可能是 '2', '3', '4', '5')
    # 這裡需要非常小心 LabelEncoder 的 classes_ 和提交格式的對應關係
    
    # 獲取 LabelEncoder 的映射
    # year_classes_sorted = sorted(le_years.classes_, key=lambda x: int(x) if str(x).isdigit() else str(x)) # 假設年份是數字或可排序
    # level_classes_sorted = sorted(le_level.classes_, key=lambda x: int(x) if str(x).isdigit() else str(x)) # 假設水平是數字或可排序
    
    # 直接使用 LabelEncoder 內部排序的類別
    # train_info.csv 中的 play years: 0, 1, 2
    # train_info.csv 中的 level: 2, 3, 4, 5
    
    # 預期提交格式的列
    # 'unique_id', 'gender', 'hold racket handed',
    # 'play years_0', 'play years_1', 'play years_2', (對應 le_years.transform([0,1,2]) 的概率)
    # 'level_2', 'level_3', 'level_4', 'level_5' (對應 le_level.transform([2,3,4,5]) 的概率)

    # 創建映射：提交列名 -> LabelEncoder 編碼後的索引
    year_col_to_idx = {}
    if hasattr(le_years, 'classes_'):
        for i, cls_val in enumerate(le_years.classes_): # le_years.classes_ 是訓練時的原始標籤的有序集合
            year_col_to_idx[f"play years_{cls_val}"] = i # cls_val 應該是 0, 1, 2

    level_col_to_idx = {}
    if hasattr(le_level, 'classes_'):
        for i, cls_val in enumerate(le_level.classes_): # cls_val 應該是 2, 3, 4, 5
            level_col_to_idx[f"level_{cls_val}"] = i


    submission_header = ["unique_id", "gender", "hold racket handed",
                         "play years_0", "play years_1", "play years_2",
                         "level_2", "level_3", "level_4", "level_5"]
    results.append(submission_header)

    default_gender_prob = 0.5
    default_hand_prob = 0.5
    
    # 根據 LabelEncoder 的類別數來確定預期概率列表長度
    num_years_classes_actual = len(le_years.classes_) if hasattr(le_years, 'classes_') and le_years.classes_ is not None else 3
    default_years_probs = [round(1.0/num_years_classes_actual, 6)] * num_years_classes_actual if num_years_classes_actual > 0 else [0.0,0.0,0.0] # 保底3個
    
    num_level_classes_actual = len(le_level.classes_) if hasattr(le_level, 'classes_') and le_level.classes_ is not None else 4
    default_level_probs = [round(1.0/num_level_classes_actual, 6)] * num_level_classes_actual if num_level_classes_actual > 0 else [0.0,0.0,0.0,0.0] # 保底4個


    for file_path in test_files:
        unique_id_str = file_path.stem
        try:
            unique_id = int(unique_id_str) # 確保 unique_id 是整數
        except ValueError:
            # print(f"Skipping file with non-integer name: {file_path.name}")
            continue # 跳過文件名不是純數字的文件

        current_row = [unique_id]
        
        # 預填充默認概率，以處理讀取或預測失敗的情況
        # 順序要和 submission_header 一致
        # gender (1), hand (1), years (num_years_classes_actual), level (num_level_classes_actual)
        
        # 確保 default_years_probs 和 default_level_probs 的長度與 submission_header 中對應的列數一致
        # submission_header: play years (3), level (4)
        final_default_years = [0.0] * 3
        for i in range(min(len(default_years_probs), 3)):
            final_default_years[i] = default_years_probs[i]
        if sum(final_default_years) == 0 and len(final_default_years) > 0 : final_default_years = [1.0/len(final_default_years)] * len(final_default_years)


        final_default_levels = [0.0] * 4
        for i in range(min(len(default_level_probs), 4)):
            final_default_levels[i] = default_level_probs[i]
        if sum(final_default_levels) == 0 and len(final_default_levels) > 0 : final_default_levels = [1.0/len(final_default_levels)] * len(final_default_levels)

        
        full_default_row = [default_gender_prob, default_hand_prob] + \
                           [round(p, 6) for p in final_default_years] + \
                           [round(p, 6) for p in final_default_levels]


        try:
            df_test_player = pd.read_csv(file_path)
            if df_test_player.empty:
                results.append([unique_id] + full_default_row)
                continue
            # 確保列名完全匹配，如果 tabular_data_test 中的列名與 aggregated_feature_names 不完全一致會出錯
            # X_player_features = df_test_player[feature_names_passed].iloc[[0]] # 使用傳入的特徵名
            
            # 為了安全，只選取存在於 df_test_player.columns 中的 feature_names_passed
            # 並確保順序與 feature_names_passed 一致
            available_features = [col for col in feature_names_passed if col in df_test_player.columns]
            missing_features = [col for col in feature_names_passed if col not in df_test_player.columns]
            if missing_features:
                # print(f"Warning: File {file_path.name} is missing features: {missing_features}. Prediction might be affected.")
                # 可以選擇填充缺失值，或報錯，或使用默認值
                # 這裡選擇繼續，XGBoost 可能能處理部分缺失（如果訓練時也遇到並學習了）
                # 或者 scaler 會因為列數不匹配而報錯
                # 更穩妥的做法是，如果缺失嚴重，則使用默認概率
                results.append([unique_id] + full_default_row) # 如果特徵嚴重缺失，用默認值
                continue

            X_player_features = df_test_player[available_features].iloc[[0]]
            # 如果 available_features 和 feature_names_passed 長度不同，scaler 會報錯
            # 需要確保 scaler 是基於 feature_names_passed 訓練的，並且這裡傳給 transform 的列也對應
            # 如果 X_player_features 的列數不等於 scaler.n_features_in_，會報錯
            if X_player_features.shape[1] != scaler.n_features_in_:
                # print(f"Error: Feature mismatch for scaling in {file_path.name}. Expected {scaler.n_features_in_}, got {X_player_features.shape[1]}. Using defaults.")
                results.append([unique_id] + full_default_row)
                continue


        except pd.errors.EmptyDataError:
            # print(f"EmptyDataError for {file_path.name}. Using defaults.")
            results.append([unique_id] + full_default_row)
            continue
        except KeyError as e: # 如果 feature_names_passed 中的列在文件中不存在
            # print(f"KeyError reading features for {file_path.name}: {e}. Using defaults.")
            results.append([unique_id] + full_default_row)
            continue
        except Exception as e: # 其他潛在錯誤
            # print(f"Generic error reading features for {file_path.name}: {e}. Using defaults.")
            results.append([unique_id] + full_default_row)
            continue

        X_scaled_player = scaler.transform(X_player_features)

        # Gender
        # predict_proba 返回 [[prob_class_0, prob_class_1]]
        # 假設 le_gender.transform(['F', 'M']) -> [0, 1] (或相反)
        # 提交格式的 'gender' 列需要 P(gender=F) 或 P(gender=M)？假設是 le_gender.classes_[0] 的概率
        gender_pred_all_classes = model_gender.predict_proba(X_scaled_player)[0]
        # 通常第一個類別的概率 (index 0) 用於二分類提交
        gender_final_prob = gender_pred_all_classes[np.where(model_gender.classes_ == le_gender.transform([le_gender.classes_[0]]))[0][0]] if len(le_gender.classes_) > 0 else default_gender_prob
        if len(le_gender.classes_) == 2: # 確保是二分類
             # 假設提交需要第一個 encode 後的類別的概率
             # 比如 le_gender.classes_ = ['F', 'M'], transform 後 F->0, M->1. model.classes_ = [0,1]
             # P(class=0) 是 gender_pred_all_classes[0]
             # 如果提交格式要求的是'F'的概率，且'F'被編碼為0，則取 gender_pred_all_classes[0]
             # 為了安全，我們取 le_gender.classes_[0] (例如 'F') 對應的概率
             try:
                 target_class_encoded = le_gender.transform([le_gender.classes_[0]])[0]
                 idx_in_model_classes = np.where(model_gender.classes_ == target_class_encoded)[0][0]
                 gender_final_prob = gender_pred_all_classes[idx_in_model_classes]
             except: # 如果 le_gender.classes_[0] 不在 model_gender.classes_ 中（不太可能）
                 gender_final_prob = default_gender_prob
        else: # 如果不是標準二分類（例如訓練數據只有一個性別）
            gender_final_prob = default_gender_prob


        # Hand
        hand_pred_all_classes = model_hand.predict_proba(X_scaled_player)[0]
        # 類似 gender，假設提交需要 le_hold.classes_[0] 的概率
        hand_final_prob = hand_pred_all_classes[np.where(model_hand.classes_ == le_hold.transform([le_hold.classes_[0]]))[0][0]] if len(le_hold.classes_) > 0 else default_hand_prob
        if len(le_hold.classes_) == 2:
            try:
                target_class_encoded = le_hold.transform([le_hold.classes_[0]])[0]
                idx_in_model_classes = np.where(model_hand.classes_ == target_class_encoded)[0][0]
                hand_final_prob = hand_pred_all_classes[idx_in_model_classes]
            except:
                hand_final_prob = default_hand_prob
        else:
            hand_final_prob = default_hand_prob
            

        # Play Years - 輸出應為3個概率值
        years_pred_probs_raw = model_years.predict_proba(X_scaled_player)[0] # shape (num_classes_in_model_years,)
        # model_years.classes_ 告訴我們 years_pred_probs_raw 中每個概率對應的 LabelEncoded 類別
        # le_years.classes_ 告訴我們原始標籤 ('0', '1', '2')
        # submission_header: "play years_0", "play years_1", "play years_2"
        
        final_years_probs_submission = [0.0] * 3 # 對應 "play years_0", "_1", "_2"
        if hasattr(le_years, 'classes_') and hasattr(model_years, 'classes_'):
            for i_submit, year_val_submit_str in enumerate(["0", "1", "2"]): # 遍歷提交格式的年份值
                try:
                    # 1. 將提交格式的年份值 (str) 轉換為 LabelEncoder 編碼後的值
                    year_val_le_encoded = le_years.transform([year_val_submit_str])[0]
                    # 2. 找到這個編碼後的值在模型輸出類別中的索引
                    idx_in_model_output = np.where(model_years.classes_ == year_val_le_encoded)[0]
                    if len(idx_in_model_output) > 0:
                        final_years_probs_submission[i_submit] = years_pred_probs_raw[idx_in_model_output[0]]
                    else: # 如果模型沒有預測這個類別（可能因為訓練數據中缺失）
                        final_years_probs_submission[i_submit] = 0.0 # 或均分剩餘概率
                except ValueError: # 如果 le_years.transform 不能處理 year_val_submit_str (例如 '0' 不在訓練標籤中)
                    final_years_probs_submission[i_submit] = 0.0
        else: # 如果 LabelEncoder 或模型沒有 classes_ 屬性，使用默認值
            final_years_probs_submission = final_default_years
        # 歸一化 (可選，如果上面填充0導致總和不為1)
        s = sum(final_years_probs_submission)
        if s > 1e-6 : final_years_probs_submission = [p/s for p in final_years_probs_submission]
        else: final_years_probs_submission = [1/3.0]*3


        # Level - 輸出應為4個概率值
        level_pred_probs_raw = model_level.predict_proba(X_scaled_player)[0]
        # submission_header: "level_2", "level_3", "level_4", "level_5"
        final_level_probs_submission = [0.0] * 4 # 對應 "level_2" 到 "level_5"
        if hasattr(le_level, 'classes_') and hasattr(model_level, 'classes_'):
            for i_submit, level_val_submit_str in enumerate(["2", "3", "4", "5"]):
                try:
                    level_val_le_encoded = le_level.transform([level_val_submit_str])[0]
                    idx_in_model_output = np.where(model_level.classes_ == level_val_le_encoded)[0]
                    if len(idx_in_model_output) > 0:
                        final_level_probs_submission[i_submit] = level_pred_probs_raw[idx_in_model_output[0]]
                    else:
                        final_level_probs_submission[i_submit] = 0.0
                except ValueError:
                     final_level_probs_submission[i_submit] = 0.0
        else:
            final_level_probs_submission = final_default_levels
        s = sum(final_level_probs_submission)
        if s > 1e-6 : final_level_probs_submission = [p/s for p in final_level_probs_submission]
        else: final_level_probs_submission = [1/4.0]*4


        current_row.extend([
            round(gender_final_prob, 6),
            round(hand_final_prob, 6)
        ])
        current_row.extend([round(p, 6) for p in final_years_probs_submission])
        current_row.extend([round(p, 6) for p in final_level_probs_submission])
        results.append(current_row)

    return results


# --- save_submission 函數不變 ---
def save_submission(results, filename='submission.csv'):
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in results:
                writer.writerow([
                    f"{val:.6f}" if isinstance(val, (float, np.floating)) and not isinstance(val, int) else val
                    for val in row
                ])
        print(f"檔案已儲存：{filename}")
    except PermissionError as e:
        print(f"無法寫入檔案 {filename}：{e}")
        alt_path = Path('./output') / Path(filename).name
        os.makedirs(alt_path.parent, exist_ok=True)
        try:
            with open(alt_path, 'w', newline='', encoding='utf-8') as f_alt:
                writer_alt = csv.writer(f_alt)
                for row_alt in results:
                    writer_alt.writerow([
                        f"{val_alt:.6f}" if isinstance(val_alt, (float, np.floating)) and not isinstance(val_alt, int) else val_alt
                        for val_alt in row_alt
                    ])
            print(f"檔案已儲存到替代路徑：{alt_path}")
        except Exception as e2:
            print(f"無法儲存到替代路徑 {alt_path}：{e2}")

# --- main 函數修改 ---
def main():
    data_generate()
    data_generate_test()

    info_df = pd.read_csv(DATA_ROOT / "39_Training_Dataset" / "train_info.csv")
    info_df['play years'] = info_df['play years'].astype(str) # 確保年份是字符串，以便 LabelEncoder 正確處理
    info_df['level'] = info_df['level'].astype(str)       # 確保水平是字符串

    # --- 更新 aggregated_feature_names ---
    agg_stats_options = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'cv']
    aggregated_feature_names = [f"{col}_{stat}" for col in ORIGINAL_HEADER_LIST for stat in agg_stats_options]

    all_features_list = []
    all_labels_list = []
    player_ids_for_split = []

    for _, row in info_df.iterrows():
        unique_id = row['unique_id']
        player_id = row['player_id']

        feature_file_path = (DATA_ROOT / "tabular_data_train" / f"{unique_id}.csv")
        if feature_file_path.exists() and feature_file_path.stat().st_size > 0:
            try:
                data = pd.read_csv(feature_file_path)
                if not data.empty:
                    # 檢查列名是否匹配新的 aggregated_feature_names
                    # 由於 generate_aggregated_features 已更新，這裡的列名應該是匹配的
                    # 但為了安全，可以做一個簡單的列數檢查或列名子集檢查
                    if list(data.columns) == aggregated_feature_names: # 嚴格檢查
                        all_features_list.append(data.iloc[0].values)
                        labels = row[['gender', 'hold racket handed', 'play years', 'level']].values
                        all_labels_list.append(labels)
                        player_ids_for_split.append(player_id)
                    # else:
                        # print(f"Warning: Column mismatch for {feature_file_path.name}. Expected {len(aggregated_feature_names)} cols, got {len(data.columns)}. File cols: {list(data.columns)[:5]}")

            except pd.errors.EmptyDataError:
                # print(f"EmptyDataError for aggregated feature file {feature_file_path.name}")
                pass
            except Exception as e:
                print(f"Error reading aggregated feature file {feature_file_path.name}: {e}")
        # else:
            # print(f"Feature file not found or empty: {feature_file_path.name}")


    if not all_features_list:
        print("No training data loaded after filtering. Exiting.")
        return

    X_all = pd.DataFrame(all_features_list, columns=aggregated_feature_names)
    y_all_df = pd.DataFrame(all_labels_list, columns=['gender', 'hold racket handed', 'play years', 'level'])

    unique_player_ids_in_data = sorted(list(set(player_ids_for_split)))

    if not unique_player_ids_in_data or len(unique_player_ids_in_data) < 2:
        print("Not enough unique players in loaded data for train/test split. Exiting.")
        return

    # --- train_test_split 和後續的數據準備不變 ---
    # 使用 player_id 進行分割，以確保 Optuna 的驗證集與訓練集來自不同選手
    train_pids, val_pids = train_test_split(unique_player_ids_in_data, test_size=0.2, random_state=42)

    train_indices = [i for i, pid_val in enumerate(player_ids_for_split) if pid_val in train_pids]
    val_indices = [i for i, pid_val in enumerate(player_ids_for_split) if pid_val in val_pids]


    if not train_indices or not val_indices:
        print("Train or validation split resulted in empty indices. Check player_id distribution and data loading.")
        # print(f"Total unique players: {len(unique_player_ids_in_data)}, Train PIDs: {len(train_pids)}, Val PIDs: {len(val_pids)}")
        # print(f"Player IDs in data: {player_ids_for_split}")
        return

    x_train_df_optuna = X_all.iloc[train_indices]
    y_train_df_optuna = y_all_df.iloc[train_indices]
    x_val_df_optuna = X_all.iloc[val_indices]
    y_val_df_optuna = y_all_df.iloc[val_indices]

    if x_train_df_optuna.empty or x_val_df_optuna.empty:
        print("Train or validation DataFrame for Optuna is empty after iloc. Issue with indices or data.")
        return
        
    print(f"Optuna x_train shape: {x_train_df_optuna.shape}, Optuna x_val shape: {x_val_df_optuna.shape}")

    scaler = MinMaxScaler()
    # 在完整的 Optuna 訓練集上 fit scaler
    X_train_scaled_optuna = scaler.fit_transform(x_train_df_optuna)
    X_val_scaled_optuna = scaler.transform(x_val_df_optuna) # 用於 Optuna 驗證

    # --- Label Encoders ---
    le_gender = LabelEncoder()
    y_train_le_gender_optuna = le_gender.fit_transform(y_train_df_optuna['gender'])
    y_val_le_gender_optuna = le_gender.transform(y_val_df_optuna['gender'])

    le_hold = LabelEncoder()
    y_train_le_hold_optuna = le_hold.fit_transform(y_train_df_optuna['hold racket handed'])
    y_val_le_hold_optuna = le_hold.transform(y_val_df_optuna['hold racket handed'])

    le_years = LabelEncoder()
    # 確保 fit 和 transform 的數據類型一致 (str)
    y_train_le_years_optuna = le_years.fit_transform(y_train_df_optuna['play years'].astype(str))
    y_val_le_years_optuna = le_years.transform(y_val_df_optuna['play years'].astype(str))
    print(f"Play Years classes after LabelEncoding: {le_years.classes_}")


    le_level = LabelEncoder()
    y_train_le_level_optuna = le_level.fit_transform(y_train_df_optuna['level'].astype(str))
    y_val_le_level_optuna = le_level.transform(y_val_df_optuna['level'].astype(str))
    print(f"Level classes after LabelEncoding: {le_level.classes_}")


    optuna_trials = 70 # 可以增加 trials
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    USE_SAMPLE_WEIGHTING_OPTUNA = True # 改為 True 來啟用樣本權重

    # 訓練最終模型時，我們將在 *所有* 非驗證集數據上訓練 (即 X_all[train_indices])
    # Optuna 用於找超參數，它使用 X_train_scaled_optuna 和 X_val_scaled_optuna

    print("Optimizing hyperparameters for gender...")
    study_gender = optuna.create_study(direction='maximize')
    study_gender.optimize(get_optuna_objective_function(gender_model_binary,
                                                        X_train_scaled_optuna, y_train_le_gender_optuna,
                                                        X_val_scaled_optuna, y_val_le_gender_optuna,
                                                        use_sample_weighting=USE_SAMPLE_WEIGHTING_OPTUNA),
                          n_trials=optuna_trials, n_jobs=-1)
    best_params_gender = study_gender.best_params
    # 用找到的最佳參數在 Optuna 的訓練集+驗證集 (即原始的 train_indices 對應的數據) 上重新訓練最終模型
    # 或者，為了更穩健，只在 Optuna 訓練集上訓練，然後在獨立的測試集上評估 (如果有的話)
    # 這裡我們假設 Optuna 的驗證集 (val_indices) 是我們的最終測試集
    # 所以，我們用 Optuna 的訓練數據訓練，用 Optuna 的驗證數據評估最終 AUC
    final_auc_gender, model_gender = gender_model_binary(X_train_scaled_optuna, y_train_le_gender_optuna,
                                                       X_val_scaled_optuna, y_val_le_gender_optuna,
                                                       best_params_gender,
                                                       sample_weights=compute_sample_weight('balanced', y=y_train_le_gender_optuna) if USE_SAMPLE_WEIGHTING_OPTUNA else None)
    print(f"Best parameters for gender: {best_params_gender}, Best Optuna Val AUC: {study_gender.best_value:.4f}, Final Val AUC with best params: {final_auc_gender:.4f}")


    print("Optimizing hyperparameters for hold...")
    study_hold = optuna.create_study(direction='maximize')
    study_hold.optimize(get_optuna_objective_function(hold_model_binary,
                                                      X_train_scaled_optuna, y_train_le_hold_optuna,
                                                      X_val_scaled_optuna, y_val_le_hold_optuna,
                                                      use_sample_weighting=USE_SAMPLE_WEIGHTING_OPTUNA),
                        n_trials=optuna_trials, n_jobs=-1)
    best_params_hold = study_hold.best_params
    final_auc_hold, model_hold = hold_model_binary(X_train_scaled_optuna, y_train_le_hold_optuna,
                                                 X_val_scaled_optuna, y_val_le_hold_optuna,
                                                 best_params_hold,
                                                 sample_weights=compute_sample_weight('balanced', y=y_train_le_hold_optuna) if USE_SAMPLE_WEIGHTING_OPTUNA else None)
    print(f"Best parameters for hold: {best_params_hold}, Best Optuna Val AUC: {study_hold.best_value:.4f}, Final Val AUC with best params: {final_auc_hold:.4f}")


    print("Optimizing hyperparameters for years...")
    study_years = optuna.create_study(direction='maximize')
    study_years.optimize(get_optuna_objective_function(years_model_multiary,
                                                       X_train_scaled_optuna, y_train_le_years_optuna,
                                                       X_val_scaled_optuna, y_val_le_years_optuna,
                                                       use_sample_weighting=USE_SAMPLE_WEIGHTING_OPTUNA),
                         n_trials=optuna_trials, n_jobs=-1)
    best_params_years = study_years.best_params
    final_auc_years, model_years = years_model_multiary(X_train_scaled_optuna, y_train_le_years_optuna,
                                                      X_val_scaled_optuna, y_val_le_years_optuna,
                                                      best_params_years,
                                                      sample_weights=compute_sample_weight('balanced', y=y_train_le_years_optuna) if USE_SAMPLE_WEIGHTING_OPTUNA else None)
    print(f"Best parameters for years: {best_params_years}, Best Optuna Val AUC: {study_years.best_value:.4f}, Final Val AUC with best params: {final_auc_years:.4f}")


    print("Optimizing hyperparameters for level...")
    study_level = optuna.create_study(direction='maximize')
    study_level.optimize(get_optuna_objective_function(level_model_multiary,
                                                       X_train_scaled_optuna, y_train_le_level_optuna,
                                                       X_val_scaled_optuna, y_val_le_level_optuna,
                                                       use_sample_weighting=USE_SAMPLE_WEIGHTING_OPTUNA),
                         n_trials=optuna_trials, n_jobs=-1)
    best_params_level = study_level.best_params
    final_auc_level, model_level = level_model_multiary(X_train_scaled_optuna, y_train_le_level_optuna,
                                                      X_val_scaled_optuna, y_val_le_level_optuna,
                                                      best_params_level,
                                                      sample_weights=compute_sample_weight('balanced', y=y_train_le_level_optuna) if USE_SAMPLE_WEIGHTING_OPTUNA else None)
    print(f"Best parameters for level: {best_params_level}, Best Optuna Val AUC: {study_level.best_value:.4f}, Final Val AUC with best params: {final_auc_level:.4f}")

    final_avg_auc = (final_auc_gender + final_auc_hold + final_auc_years + final_auc_level) / 4
    print(f'Final Average Validation AUC Score: {final_avg_auc:.4f}')

    print("Predicting on test set...")
    # 使用在 Optuna 訓練集上訓練得到的 scaler 和 LabelEncoders
    # 以及使用最佳參數在 Optuna 訓練集上訓練的模型
    results = predict_test(model_gender, model_hold, model_years, model_level, scaler,
                           le_gender, le_hold, le_years, le_level, aggregated_feature_names)
    save_submission(results, filename='submission_new_features.csv') # 改個名
    print("Submission file created.")

#!/usr/bin/env python

if __name__ == "__main__":
    main()