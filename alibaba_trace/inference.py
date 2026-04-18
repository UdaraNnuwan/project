import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from alibaba_trace.data_pipeline import (
    PipelineConfig, build_data_schema, compile_split_plan,
    fit_streaming_scalers, build_streaming_dataloader
)
from alibaba_trace.model import BiLSTMFiLMAutoencoder

def run_inference_and_log():
    # 1. Setup Configuration 
    raw_dir = Path("../data").resolve() 
    if not (raw_dir / "container_usage.tar.gz").exists():
        raw_dir = Path("dataset/data").resolve() 
        
    config = PipelineConfig(
        raw_data_dir=raw_dir,
        chunksize=50000,
        batch_size=128,
        window_size=30,
        container_limit=50 # Using 50 containers for the demo
    )

    print("Phase 1: Preparing Vocabularies and Scalers...")
    allowed_entities, row_counts, vocabulary = build_data_schema(config)
    split_plan = compile_split_plan(config, row_counts)
    x_scaler, c_scaler = fit_streaming_scalers(config, split_plan, vocabulary, allowed_entities)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = len(config.feature_columns)
    context_dim = len(config.context_columns)

    model = BiLSTMFiLMAutoencoder(
        window_size=config.window_size,
        n_features=n_features,
        context_dim=context_dim,
        hidden_size=64,
        num_layers=1
    ).to(device)
    
    # Normally you would load your trained weights here:
    # model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()

    print("Phase 2: Initiating Test Inference Stream...")
    test_loader = build_streaming_dataloader(
        config=config,
        split="test",  # We run inference on the TEST split
        plan=split_plan,
        vocab=vocabulary,
        x_scaler=x_scaler,
        c_scaler=c_scaler,
        allowed=allowed_entities,
        device=device
    )

    # We will use a dummy threshold for demonstration. 
    # In practice, you calculate this on your validation set.
    ANOMALY_THRESHOLD = 0.5 

    anomaly_log = []
    
    print(" Running inference and determining Normal vs. Anomaly...")
    with torch.no_grad():
        for batch_idx, (x_batch, c_batch) in enumerate(test_loader):
            
            # Reconstruct the window
            reconstructed = model(x_batch, c_batch)
            
            # Calculate MSE per window (batch_size, sequence_length, features)
            # We average the error over the sequence and features
            mse_per_window = torch.mean((reconstructed - x_batch) ** 2, dim=(1, 2))
            
            mse_scores = mse_per_window.cpu().numpy()
            
            for idx, score in enumerate(mse_scores):
                # Is it normal or not?
                status = "Anomaly" if score > ANOMALY_THRESHOLD else "Normal"
                
                anomaly_log.append({
                    "batch": batch_idx,
                    "window_index": idx,
                    "mse_score": round(float(score), 4),
                    "threshold": ANOMALY_THRESHOLD,
                    "status": status  # This logs "Normal" or "Anomaly" (normalda nadda)
                })

    # Save the log to a file
    output_df = pd.DataFrame(anomaly_log)
    output_path = Path("anomaly_predictions_log.csv")
    output_df.to_csv(output_path, index=False)
    
    print(f"\\n Inference complete! Logged {len(output_df)} windows.")
    print(f" Results saved to: {output_path.resolve()}")
    
    # Preview top 10 results
    print("\\nPreview of Log:")
    print(output_df.head(10).to_string(index=False))

if __name__ == "__main__":
    run_inference_and_log()
