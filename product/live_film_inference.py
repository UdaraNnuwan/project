import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import time

# =========================================================
# PATH SETUP
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.model import FiLMAutoencoder  # noqa: E402

# =========================================================
# CONFIGURATION - ඔබගේ සැබෑ දත්ත වලට අනුව වෙනස් කරන ලදී
# =========================================================
CSV_PATH = os.path.join(CURRENT_DIR, "../prometheus/prometheus_timeseries_metrics.csv")

MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "ae_model.pt")
X_SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
C_SCALER_PATH = os.path.join(MODEL_DIR, "ctx_scaler.joblib")
DETECTOR_META_PATH = os.path.join(MODEL_DIR, "detector_meta.joblib")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- මෙන්න මෙතැන ඔබේ දත්ත වලට අනුව නිවැරදි කර ඇත ---
TARGET_NAMESPACE = "default"
TARGET_POD = "chcekone"   # "chcekone" යන කොටස අඩංගු Pods සොයයි
TARGET_CONTAINER = "nginx"
# --------------------------------------------------

FEATURE_COLS = [
    "cpu_util", "mem_util", "net_in", "net_out", 
    "disk_read", "disk_write", "mem_rss", "mem_cache"
]

def load_assets():
    """පුහුණු කළ ආකෘතිය සහ Scalers පූරණය කිරීම."""
    print("[INFO] Loading scalers and metadata...")
    x_scaler = joblib.load(X_SCALER_PATH)
    c_scaler = joblib.load(C_SCALER_PATH)
    det_meta = joblib.load(DETECTOR_META_PATH)
    
    threshold = det_meta['threshold']
    window_size = det_meta['window_size']
    
    # Model Loading Fix: මුළු checkpoint එකම load කර 'model_state_dict' පරීක්ෂා කිරීම
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    model = FiLMAutoencoder(
        window_size=window_size,
        n_features=len(FEATURE_COLS), 
        context_dim=c_scaler.n_features_in_
    ).to(DEVICE)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return x_scaler, c_scaler, threshold, window_size, model

# def main():
#     x_scaler, c_scaler, threshold, window_size, model = load_assets()

#     if not os.path.exists(CSV_PATH):
#         print(f"[ERROR] CSV not found at {CSV_PATH}")
#         return

#     print(f"[INFO] Reading live data from {CSV_PATH}...")
#     df = pd.read_csv(CSV_PATH)

#     # Filter data - මෙහිදී case-insensitive සහ partial match භාවිතා කරයි
#     mask = (df['namespace'] == TARGET_NAMESPACE) & \
#            (df['pod'].str.contains(TARGET_POD, na=False)) & \
#            (df['container'] == TARGET_CONTAINER)
    
#     target_df = df[mask].copy()

#     if len(target_df) < window_size:
#         print(f"[WARN] Not enough data for {TARGET_CONTAINER}. Need {window_size} rows, but found {len(target_df)}")
#         print(f"[DEBUG] CSV contains these pods: {df['pod'].unique()[:5]}")
#         return

#     # අවසාන rows (window_size ප්‍රමාණය) ලබා ගැනීම
#     target_df = target_df.tail(window_size)
#     x_raw = target_df[FEATURE_COLS].values
    
#     # Scaling & Transformation
#     x_log = np.log1p(np.clip(x_raw, a_min=0, a_max=None))
#     x_scaled = x_scaler.transform(x_log) 

#     # Context handling (Fallback to zero if scaling fails)
#     try:
#         # Context scaler එක පුහුණු කළ දත්ත වලට ගැලපිය යුතුය
#         c_raw = target_df[['namespace', 'container']].iloc[-1:]
#         c_scaled = c_scaler.transform(c_raw)
#     except:
#         c_scaled = np.zeros((1, c_scaler.n_features_in_))

#     # Inference
#     x_tensor = torch.FloatTensor(x_scaled).unsqueeze(0).to(DEVICE)
#     c_tensor = torch.FloatTensor(c_scaled).to(DEVICE)

#     with torch.no_grad():
#         reconstructed = model(x_tensor, c_tensor)
        
#     x_pred_scaled = reconstructed.cpu().numpy().squeeze()
    
#     # Anomaly Calculation
#     mse_per_feature = np.mean((x_scaled - x_pred_scaled)**2, axis=0)
#     total_score = np.mean(mse_per_feature)

#     print("\n" + "="*40)
#     print(f"POD       : {target_df['pod'].iloc[-1]}")
#     print(f"CONTAINER : {TARGET_CONTAINER}")
#     print(f"SCORE     : {total_score:.6f}")
#     print(f"THRESHOLD : {threshold:.6f}")
#     print(f"STATUS    : {'🔴 ANOMALY' if total_score > threshold else '🟢 NORMAL'}")
#     print("="*40)

#     if total_score > threshold:
#         top_idx = np.argmax(mse_per_feature)
#         print(f"Suspected Feature: {FEATURE_COLS[top_idx]}")
def main():
    x_scaler, c_scaler, threshold, window_size, model = load_assets()

    print(f"\n[INFO] Starting Real-Time Monitoring for {TARGET_CONTAINER}...")
    
    while True:
        try:
            if os.path.exists(CSV_PATH):
                df = pd.read_csv(CSV_PATH)
                mask = (df['namespace'] == TARGET_NAMESPACE) & \
                       (df['pod'].str.contains(TARGET_POD, na=False)) & \
                       (df['container'] == TARGET_CONTAINER)
                
                target_df = df[mask].tail(window_size)

                if len(target_df) >= window_size:
                    x_raw = target_df[FEATURE_COLS].values
                    x_log = np.log1p(np.clip(x_raw, a_min=0, a_max=None))
                    x_scaled = x_scaler.transform(x_log) 

                    x_tensor = torch.FloatTensor(x_scaled).unsqueeze(0).to(DEVICE)
                    c_scaled = np.zeros((1, c_scaler.n_features_in_))
                    c_tensor = torch.FloatTensor(c_scaled).to(DEVICE)

                    with torch.no_grad():
                        reconstructed = model(x_tensor, c_tensor)
                    
                    x_pred_scaled = reconstructed.cpu().numpy().squeeze()
                    
                    # එක් එක් පේළියට සහ එක් එක් feature එකට අදාළ Error එක ගණනය කිරීම
                    # MSE = (Original - Predicted)^2
                    errors = (x_scaled - x_pred_scaled)**2
                    mse_per_feature = np.mean(errors, axis=0)
                    total_score = np.mean(mse_per_feature)

                    # Output මුද්‍රණය කිරීම
                    print("\n" + "="*50)
                    print(f"POD       : {target_df['pod'].iloc[-1]}")
                    print(f"CONTAINER : {TARGET_CONTAINER}")
                    print(f"SCORE     : {total_score:.6f}")
                    print(f"THRESHOLD : {threshold:.6f}")
                    
                    if total_score > threshold:
                        print(f"STATUS    : 🔴 ANOMALY DETECTED!")
                        print("-" * 50)
                        print("REASON (TOP CONTRIBUTING METRICS):")
                        
                        # වැඩිම බලපෑමක් කළ metrics 3 සොයා ගැනීම
                        top_indices = np.argsort(mse_per_feature)[::-1][:3]
                        for idx in top_indices:
                            f_name = FEATURE_COLS[idx]
                            f_error = mse_per_feature[idx]
                            # අදාළ metric එකේ දැනට පවතින සැබෑ අගය (Original Value)
                            actual_val = x_raw[-1, idx] 
                            print(f" -> {f_name.upper()}: Error={f_error:.4f} (Actual Val={actual_val:.2f})")
                        
                        print("-" * 50)
                        print(f"LATEST LOG DATA (Snapshot):")
                        print(target_df[FEATURE_COLS].tail(1).to_string(index=False))
                    else:
                        print(f"STATUS    : 🟢 NORMAL")
                    
                    print("="*50)

        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(5)
if __name__ == "__main__":
    main()