import os
import time
import requests
import pandas as pd
from functools import reduce
from datetime import datetime, timezone

# =========================================================
# CONFIGURATION
# =========================================================
PROM_URL = "http://35.206.92.147:9090/api/v1/query"
OUTPUT_FILE = "prometheus_timeseries_metrics.csv"
COLLECT_INTERVAL_SECONDS = 5  # කාල පරාසය තත්පර 15 දක්වා වැඩි කරන ලදී

# True නම්, සියලුම අගයන් බින්දුව (zero) වන පේළි ඉවත් කරයි
SKIP_ALL_ZERO_ROWS = True

# Prometheus queries (ලක්ෂණ 8ක් වන සේ සකස් කර ඇත)
QUERIES = {
    "cpu_util": """
        sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name) 
        (rate(container_cpu_usage_seconds_total{image!="", container_label_io_kubernetes_container_name!=""}[5m]))
    """,
    "mem_util": """
        sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name) 
        (container_memory_working_set_bytes{image!="", container_label_io_kubernetes_container_name!=""})
    """,
    "net_in": """
        sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name) 
        (rate(container_network_receive_bytes_total{image!="", interface="eth0"}[5m]))
    """,
    "net_out": """
        sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name) 
        (rate(container_network_transmit_bytes_total{image!="", interface="eth0"}[5m]))
    """,
    "disk_read": """
        sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name) 
        (rate(container_fs_reads_bytes_total{image!="", container_label_io_kubernetes_container_name!=""}[5m]))
    """,
    "disk_write": """
        sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name) 
        (rate(container_fs_writes_bytes_total{image!="", container_label_io_kubernetes_container_name!=""}[5m]))
    """,
    "mem_rss": """
        sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name) 
        (container_memory_rss{image!="", container_label_io_kubernetes_container_name!=""})
    """,
    "mem_cache": """
        sum by (container_label_io_kubernetes_pod_namespace, container_label_io_kubernetes_pod_name, container_label_io_kubernetes_container_name) 
        (container_memory_cache{image!="", container_label_io_kubernetes_container_name!=""})
    """
}

# CSV ගොනුවේ අඩංගු විය යුතු තීරු (Columns) අනුපිළිවෙල
FINAL_COLUMNS = [
    "timestamp",
    "namespace",
    "pod",
    "container",
    "cpu_util",
    "mem_util",
    "net_in",
    "net_out",
    "disk_read",
    "disk_write",
    "mem_rss",
    "mem_cache"
]

def query_prometheus(query: str):
    """Prometheus වෙතින් දත්ත ලබා ගැනීම."""
    try:
        response = requests.get(PROM_URL, params={"query": query}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success":
            return []
        return payload["data"]["result"]
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        return []

def normalize_labels(metric: dict) -> dict:
    """Labels එකම ආකෘතියකට සැකසීම."""
    return {
        "namespace": metric.get("container_label_io_kubernetes_pod_namespace", ""),
        "pod": metric.get("container_label_io_kubernetes_pod_name", ""),
        "container": metric.get("container_label_io_kubernetes_container_name", "")
    }

def collect_all_metrics_once() -> pd.DataFrame:
    """සියලුම metrics එක් snapshot එකක් ලෙස රැස් කිරීම."""
    metric_dfs = {}

    for name, query in QUERIES.items():
        results = query_prometheus(query)
        rows = []
        for item in results:
            labels = normalize_labels(item["metric"])
            # Network metrics වලට container label එක නැති නිසා __pod__ ලෙස නම් කරයි
            if name in ["net_in", "net_out"] and not labels["container"]:
                labels["container"] = "__pod__"
            
            rows.append({
                "namespace": labels["namespace"],
                "pod": labels["pod"],
                "container": labels["container"],
                name: float(item["value"][1])
            })
        metric_dfs[name] = pd.DataFrame(rows)

    # Container index එකක් සෑදීම (Network metrics බෙදා හැරීමට)
    container_parts = [df for k, df in metric_dfs.items() if k not in ["net_in", "net_out"] and not df.empty]
    container_index = pd.concat(container_parts, ignore_index=True)[["namespace", "pod", "container"]].drop_duplicates() if container_parts else pd.DataFrame()

    # Network metrics container මට්ටමට ව්‍යාප්ත කිරීම
    for net_key in ["net_in", "net_out"]:
        df = metric_dfs[net_key]
        if not df.empty and not container_index.empty:
            pod_level = df[df["container"] == "__pod__"]
            merged = pod_level.merge(container_index, on=["namespace", "pod"], suffixes=("_old", ""))
            metric_dfs[net_key] = merged[["namespace", "pod", "container", net_key]]

    # සියලුම DataFrames එකතු කිරීම (Merge)
    dfs = [df for df in metric_dfs.values() if not df.empty]
    if not dfs: return pd.DataFrame()

    final_df = reduce(lambda left, right: pd.merge(left, right, on=["namespace", "pod", "container"], how="outer"), dfs)
    final_df = final_df.fillna(0.0)
    final_df["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Column අනුපිළිවෙල සහ හිස් තීරු පිරවීම
    for col in FINAL_COLUMNS:
        if col not in final_df.columns: final_df[col] = 0.0 if col not in ["timestamp", "namespace", "pod", "container"] else ""
    
    final_df = final_df[FINAL_COLUMNS]

    if SKIP_ALL_ZERO_ROWS:
        num_cols = ["cpu_util", "mem_util", "net_in", "net_out", "disk_read", "disk_write", "mem_rss", "mem_cache"]
        final_df = final_df[final_df[num_cols].sum(axis=1) > 0]

    return final_df

def main():
    print(f"[INFO] Collecting 8 features to {OUTPUT_FILE}...")
    while True:
        try:
            df = collect_all_metrics_once()
            if not df.empty:
                file_exists = os.path.exists(OUTPUT_FILE)
                df.to_csv(OUTPUT_FILE, mode="a", header=not file_exists, index=False)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved {len(df)} rows.")
            else:
                print("[WARN] No data from Prometheus.")
        except KeyboardInterrupt: break
        except Exception as e: print(f"[ERROR] Loop failed: {e}")
        time.sleep(COLLECT_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()