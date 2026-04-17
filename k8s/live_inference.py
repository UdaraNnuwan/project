import os
import time
import requests
import logging
import sys
import torch
import csv
import json

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

# Example PromQL query: Modify this to extract the relevant metrics for your model
PROMQL_QUERY = 'rate(container_cpu_usage_seconds_total{image!=""}[5m])'

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

# Example PromQL query: Modify this to extract the relevant metrics for your model
PROMQL_QUERY = 'rate(container_cpu_usage_seconds_total{image!=""}[5m])'

def load_model():
    """
    Load the DualHeadBiLSTMFiLM model here.
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
    return model

def fetch_metrics(prom_url):
    """
    Fetches metrics from the Prometheus API.
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

def process_and_infer(model, raw_data):
    """
    Preprocess data and run anomaly detection inference.
    """
    # Placeholder for running the actual predictions
    # 1. Parse 'raw_data' from Prometheus
    # 2. Window and scale the time-series arrays
    # 3. Call `predictions = model.predict(processed_data)`
    # 4. Compute anomaly thresholds/loss and log anomalies
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
        model = load_model()
    except Exception as e:
        logger.critical(f"Failed to load model. Exiting pod. Error: {e}")
        sys.exit(1)
        
    logger.info("Entering continuous inference loop...")
    
    while True:
        start_time = time.time()
        
        logger.info(f"Querying Prometheus API with query: {PROMQL_QUERY}")
        data = fetch_metrics(prom_url)
        
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
                    process_and_infer(model, results)
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
