import os
import time
import requests
import logging
import sys
import torch
import csv
import json
import numpy as np
from collections import defaultdict, deque
from sklearn.preprocessing import MinMaxScaler

try:
    from dotenv import load_dotenv
    # Explicitly load from the project root .env file, resolving reliably across directory structures
    root_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    load_dotenv(dotenv_path=root_env_path)
except ImportError:
    pass  # python-dotenv not installed, fallback to manual env vars

# Append project root to sys.path if running locally so alibaba_trace can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from alibaba_trace.model_architecture import load_dual_head_model
except ImportError:
    # Fallback if scripts are copied into the same directory in Docker
    from model_architecture import load_dual_head_model

# Configure production-ready logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BiLSTM-FiLM-Inference")

# Constants
QUERY_INTERVAL = 10 # seconds
TIMEOUT = 10 # seconds
DEFAULT_PROM_URL = "http://35.206.92.147:9090/api/v1/query"

# Global buffers and thresholds
buffers = defaultdict(lambda: deque(maxlen=50))
loss_histories = defaultdict(lambda: deque(maxlen=100))
threshold_multiplier = float(os.environ.get("THRESHOLD_MULTIPLIER", 2.0))
warm_up_cycles = int(os.environ.get("WARM_UP_CYCLES", 10))
cycle_count = 0

# Modified this to properly match your cAdvisor metrics layout.
# Your setup doesn't have a 'container' label, it uses 'name'. Also stripping 'pause' images to filter out sandbox POD networks.
PROMQL_QUERY = 'rate(container_cpu_usage_seconds_total{image!="", image!~".*pause.*", name!=""}[5m])'

# Import Common Notification Interfaces
try:
    from src.notifications import send_telegram_alert, analyze_with_gpt
except ImportError:
    # Handle scenario where scripts are isolated (e.g. Dockerfile mapping rules)
    from notifications import send_telegram_alert, analyze_with_gpt

def log_results_to_csv(results, filepath="metrics_log.csv"):
    """Append batch metric results reliably to a local CSV file."""
    file_exists = os.path.isfile(filepath)
    try:
        with open(filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'Container_Name', 'Metric_Value'])
                
            for result in results:
                labels = result.get('metric', {})
                container_name = labels.get('container_label_io_kubernetes_pod_name', labels.get('name', labels.get('id', 'unknown')))
                val_data = result.get('value', [])
                if len(val_data) == 2:
                    ts, val = val_data
                    writer.writerow([ts, container_name, val])
    except Exception as e:
        logger.error(f"Failed to write metrics to CSV: {e}")

# Modified this to properly match your cAdvisor metrics layout.
PROMQL_QUERY = 'rate(container_cpu_usage_seconds_total{image!="", image!~".*pause.*", name!=""}[5m])'

def load_model():
    """
    Load the DualHeadBiLSTMFiLM model and scaler here.
    """
    logger.info("Loading Dual-Head BiLSTM-FiLM model...")
    
    # Check for model path in env vars (e.g. from volume mount or copied file)
    # Defaulting to local path if not running in cluster yet
    # We resolve the absolute path relative to this script's directory (the k8s folder)
    default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "alibaba_trace", "outputs", "dual_head_model.pt")
    model_path = os.environ.get("MODEL_PATH", default_path)
    
    if not os.path.exists(model_path):
        # Fallback for Docker container if copied to root /app
        model_path = "dual_head_model.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found. Checked absolute path: {default_path} and Docker path: dual_head_model.pt")
            
    # Load scaler
    scaler_path = os.path.join(os.path.dirname(model_path), "scaler_params.json")
    if not os.path.exists(scaler_path):
        scaler_path = "scaler_params.json"
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler params not found. Checked {scaler_path}")
    
    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)
    
    scaler = MinMaxScaler()
    scaler.min_ = np.array(scaler_params['min_'])
    scaler.scale_ = 1.0 / (np.array(scaler_params['max_']) - np.array(scaler_params['min_']))
    scaler.data_min_ = scaler.min_
    scaler.data_max_ = np.array(scaler_params['max_'])
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.n_features_in_ = len(scaler.min_)
    
    # Model parameters as defined during training
    WINDOW_SIZE = 50
    N_TS_FEAT = 7
    N_META_FEAT = 2
    LATENT_DIM = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model to device: {device} from {model_path}")
    
    model = load_dual_head_model(
        checkpoint_path=model_path,
        window_size=WINDOW_SIZE,
        n_ts_features=N_TS_FEAT,
        n_meta_features=N_META_FEAT,
        latent_dim=LATENT_DIM,
        device=device
    )
    return model, scaler, device

def fetch_prometheus_data(prom_url):
    """
    Robust fetch function that calls the Prometheus HTTP API endpoint (/api/v1/query).
    """
    try:
        response = requests.get(
            prom_url, 
            params={'query': PROMQL_QUERY}, 
            timeout=TIMEOUT
        )
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as errh:
        logger.error(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        logger.error(f"Connection Error (is Prometheus down?): {errc}")
    except requests.exceptions.Timeout as errt:
        logger.error(f"Timeout Error querying Prometheus: {errt}")
    except requests.exceptions.RequestException as err:
        logger.error(f"General Request Exception: {err}")
    
    return None

def process_and_infer(model, scaler, device, raw_data):
    """
    Preprocess data and run anomaly detection inference using the trained model.
    """
    global cycle_count
    
    # Process each container's metric stream from the current batch
    for result in raw_data:
        # Extract the metric dictionary (labels/metadata) to be fed into the FiLM layer
        labels = result.get('metric', {})
        container_name = labels.get('container_label_io_kubernetes_pod_name', labels.get('name', labels.get('id', 'unknown')))
        
        # Extract the value[1] (the actual float metric) to be fed into the BiLSTM
        val_data = result.get('value', [])
        if len(val_data) < 2:
            continue
            
        try:
            cpu_val = float(val_data[1])
        except ValueError:
            logger.error(f"Failed to cast value to float: {val_data[1]}")
            continue
            
        # Basic filter: log an exact 0.0 value as a warning and skip feeding it to the model        
        if cpu_val == 0.0 or np.isnan(cpu_val):
            logger.warning(f"⚠️ Skipping container {container_name}. CPU value is {cpu_val}.")
            continue
        
        # Append to buffer
        buffers[container_name].append(cpu_val)
        
        # Only run inference when window is full
        if len(buffers[container_name]) < 50:
            continue
        
        # Prepare time-series tensor: replicate CPU to 7 features
        sequence = list(buffers[container_name])
        ts_features = np.array([[val] * 7 for val in sequence])
        
        # Normalize using scaler
        ts_features = scaler.transform(ts_features.reshape(-1, 7)).reshape(50, 7)
        
        ts_tensor = torch.tensor(ts_features).float().unsqueeze(0).to(device)
        
        # Prepare metadata tensor
        meta1 = hash(container_name) % 1000 / 1000.0  # Normalized hash
        meta2 = 1.0 if 'image' in labels and labels['image'] else 0.0
        meta_tensor = torch.tensor([[meta1, meta2]]).float().to(device)
        
        # Run model inference
        with torch.no_grad():
            recon_output, forecast_output = model(ts_tensor, meta_tensor)
        
        # Compute MSE loss using reconstruction output
        mse_loss = torch.mean((recon_output - ts_tensor) ** 2).item()
        
        # Append to loss history
        loss_histories[container_name].append(mse_loss)
        
        # Compute dynamic threshold if sufficient history
        if len(loss_histories[container_name]) >= 10:
            losses = list(loss_histories[container_name])
            mean_loss = sum(losses) / len(losses)
            std_loss = (sum((x - mean_loss) ** 2 for x in losses) / len(losses)) ** 0.5
            threshold = mean_loss + threshold_multiplier * std_loss
        else:
            threshold = float('inf')  # No threshold yet
        
        # Determine status
        if cycle_count > warm_up_cycles and len(loss_histories[container_name]) >= 10:
            status = "Anomaly" if mse_loss > threshold else "Normal"
        else:
            status = "Normal"  # Warm-up or insufficient history
        
        # Log the status
        if status == "Anomaly":
            logger.warning(f"🚨 STATUS: {status} | Container: {container_name} | MSE: {mse_loss:.4f} > Threshold: {threshold:.4f}")
            send_telegram_alert(f"🚨 Anomaly Detected!\nContainer: {container_name}\nMSE: {mse_loss:.4f}\nThreshold: {threshold:.4f}")
        else:
            logger.info(f"✅ STATUS: {status} | Container: {container_name} | MSE: {mse_loss:.4f} <= Threshold: {threshold:.4f}")

    logger.info("Successfully processed batch and ran inference.")

def main():
    logger.info("Initializing Live Inference Service...")
    
    # Send test message to group at initialization
    send_telegram_alert("🚀 BiLSTM-FiLM Anomaly Inference Service Started! Connecting to metrics stream...")
    
    # Environment variable check with fallback
    prom_url = os.environ.get("PROM_URL", DEFAULT_PROM_URL)
    logger.info(f"Configured Prometheus Endpoint: {prom_url}")
    
    # Load model once at startup to prevent memory leaks and overhead
    try:
        model, scaler, device = load_model()
    except Exception as e:
        logger.critical(f"Failed to load model. Exiting pod. Error: {e}")
        sys.exit(1)
        
    logger.info("Entering continuous inference loop...")
    
    while True:
        start_time = time.time()
        global cycle_count
        cycle_count += 1
        
        logger.info(f"Querying Prometheus API with query: {PROMQL_QUERY}")
        data = fetch_prometheus_data(prom_url)
        
        if data and data.get('status') == 'success':
            results = data.get('data', {}).get('result', [])
            if not results:
                logger.warning("Query returned successful status but no metrics were found.")
            else:
                logger.info(f"Fetched {len(results)} metric streams from Prometheus.")
                
                # Log each individual query result (to stdout and to CSV)
                log_results_to_csv(results)
                
                for i, result in enumerate(results):
                    metric_labels = result.get('metric', {})
                    # A small trick to cleanly log important labels (like name or image) and the value
                    container_name = metric_labels.get('container_label_io_kubernetes_pod_name', metric_labels.get('name', metric_labels.get('id', 'unknown')))
                    value = result.get('value', [])
                    logger.info(f"  [{i+1}/{len(results)}] Container: {container_name} | Labels: {metric_labels} | Data: {value}")
                
                try:
                    process_and_infer(model, scaler, device, results)
                except Exception as e:
                    # Catch ML-related exceptions so the pod doesn't crash on bad data
                    logger.error(f"Error during inference execution: {e}", exc_info=True)
                    gpt_analysis = analyze_with_gpt(str(e))
                    send_telegram_alert(f"🚨 Inference Error Alert!\n\nError: {e}\n\n🤖 GPT Analysis:\n{gpt_analysis}")
        else:
            logger.warning("Failed to retrieve valid data from Prometheus in this cycle. Skipping inference.")
            
        # Ensure we sleep for the remainder of the 60 seconds interval
        elapsed = time.time() - start_time
        sleep_time = max(0, QUERY_INTERVAL - elapsed)
        
        logger.info(f"Cycle completed in {elapsed:.2f}s. Sleeping for {sleep_time:.2f}s before next query...")
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
