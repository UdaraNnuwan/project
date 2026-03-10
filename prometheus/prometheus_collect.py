import requests
import pandas as pd
from functools import reduce

PROM_URL = "http://35.206.92.147:9090/api/v1/query"

# PromQL queries adjusted to cAdvisor label format
QUERIES = {
    "cpu_usage": """
        sum by (
            container_label_io_kubernetes_pod_namespace,
            container_label_io_kubernetes_pod_name,
            container_label_io_kubernetes_container_name
        ) (
            rate(container_cpu_usage_seconds_total{
                job="cadvisor",
                image!="",
                container_label_io_kubernetes_pod_name!="",
                container_label_io_kubernetes_container_name!=""
            }[5m])
        )
    """,
    "memory_usage_bytes": """
        sum by (
            container_label_io_kubernetes_pod_namespace,
            container_label_io_kubernetes_pod_name,
            container_label_io_kubernetes_container_name
        ) (
            container_memory_working_set_bytes{
                job="cadvisor",
                image!="",
                container_label_io_kubernetes_pod_name!="",
                container_label_io_kubernetes_container_name!=""
            }
        )
    """,
    "network_rx_bytes": """
        sum by (
            container_label_io_kubernetes_pod_namespace,
            container_label_io_kubernetes_pod_name
        ) (
            rate(container_network_receive_bytes_total{
                job="cadvisor",
                container_label_io_kubernetes_pod_name!="",
                interface="eth0"
            }[5m])
        )
    """,
    "network_tx_bytes": """
        sum by (
            container_label_io_kubernetes_pod_namespace,
            container_label_io_kubernetes_pod_name
        ) (
            rate(container_network_transmit_bytes_total{
                job="cadvisor",
                container_label_io_kubernetes_pod_name!="",
                interface="eth0"
            }[5m])
        )
    """,
    "restart_count": """
        sum by (namespace, pod, container) (
            kube_pod_container_status_restarts_total{container!=""}
        )
    """
}


def query_prometheus(query: str):
    """Run a single instant query against Prometheus."""
    try:
        response = requests.get(PROM_URL, params={"query": query}, timeout=20)
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") != "success":
            print(f"Query failed:\n{query}")
            return []

        return payload["data"]["result"]

    except requests.RequestException as e:
        print(f"HTTP error while querying Prometheus: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


def normalize_metric_labels(metric: dict) -> dict:
    """
    Normalize metric labels from different exporters into:
    namespace, pod, container
    """
    namespace = (
        metric.get("namespace")
        or metric.get("container_label_io_kubernetes_pod_namespace")
        or ""
    )
    pod = (
        metric.get("pod")
        or metric.get("container_label_io_kubernetes_pod_name")
        or ""
    )
    container = (
        metric.get("container")
        or metric.get("container_label_io_kubernetes_container_name")
        or ""
    )

    return {
        "namespace": namespace,
        "pod": pod,
        "container": container,
    }


def result_to_df(results, metric_name: str) -> pd.DataFrame:
    """Convert Prometheus vector result into a DataFrame."""
    rows = []

    for item in results:
        metric = item.get("metric", {})
        value = item.get("value", [None, None])

        base = normalize_metric_labels(metric)

        # Network metrics are usually pod-level from sandbox/container labels.
        # If container is missing, keep it as '__pod__' so merge works.
        if metric_name in ("network_rx_bytes", "network_tx_bytes") and not base["container"]:
            base["container"] = "__pod__"

        rows.append({
            "namespace": base["namespace"],
            "pod": base["pod"],
            "container": base["container"],
            metric_name: float(value[1]) if value[1] is not None else None
        })

    return pd.DataFrame(rows)


def expand_pod_level_network_to_containers(df_metric: pd.DataFrame, container_index: pd.DataFrame) -> pd.DataFrame:
    """
    Network metrics may come at pod-level without container labels.
    Expand pod-level metrics to all containers in that pod using known container index.
    """
    if df_metric.empty:
        return df_metric

    pod_level = df_metric[df_metric["container"] == "__pod__"].copy()
    normal = df_metric[df_metric["container"] != "__pod__"].copy()

    if pod_level.empty:
        return df_metric

    if container_index.empty:
        return normal

    expanded = pod_level.merge(
        container_index[["namespace", "pod", "container"]].drop_duplicates(),
        on=["namespace", "pod"],
        how="left",
        suffixes=("", "_real")
    )

    expanded["container"] = expanded["container_real"].fillna("__pod__")
    expanded = expanded.drop(columns=["container_real"])

    return pd.concat([normal, expanded], ignore_index=True)


def collect_all_metrics() -> pd.DataFrame:
    """Query all metrics and merge them into one table."""
    metric_dfs = {}

    for metric_name, query in QUERIES.items():
        print(f"Collecting: {metric_name}")
        results = query_prometheus(query)
        df = result_to_df(results, metric_name)
        metric_dfs[metric_name] = df

    # Build container index from metrics that definitely have container labels
    container_index_parts = []
    for key in ["cpu_usage", "memory_usage_bytes", "restart_count"]:
        df = metric_dfs.get(key, pd.DataFrame())
        if not df.empty:
            container_index_parts.append(df[["namespace", "pod", "container"]])

    if container_index_parts:
        container_index = pd.concat(container_index_parts, ignore_index=True).drop_duplicates()
    else:
        container_index = pd.DataFrame(columns=["namespace", "pod", "container"])

    # Expand pod-level network metrics to container-level
    for key in ["network_rx_bytes", "network_tx_bytes"]:
        metric_dfs[key] = expand_pod_level_network_to_containers(metric_dfs[key], container_index)

    dfs = [df for df in metric_dfs.values() if not df.empty]

    if not dfs:
        return pd.DataFrame()

    merged_df = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on=["namespace", "pod", "container"],
            how="outer"
        ),
        dfs
    )

    merged_df = merged_df.fillna(0)

    # Remove synthetic placeholder rows if real container rows exist for same pod
    merged_df = merged_df.sort_values(["namespace", "pod", "container"]).reset_index(drop=True)

    return merged_df


if __name__ == "__main__":
    df = collect_all_metrics()

    if df.empty:
        print("No data returned from Prometheus.")
    else:
        print("\nCollected Metrics:\n")
        print(df.head(50).to_string(index=False))

        output_file = "prometheus_container_metrics.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved to: {output_file}")