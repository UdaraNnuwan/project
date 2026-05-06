from typing import Dict, Tuple

# Main prompt template for the LLM.
PROMPT_TEMPLATE = """\
You are an Expert Site Reliability Engineer (SRE) and DevOps Architect specialising \
in Kubernetes containerised microservices. You have deep expertise in PostgreSQL, Redis, \
Kafka, REST API services, and ML inference platforms.

A BiLSTM-FiLM deep-learning anomaly detection system has flagged a container for \
a potential anomaly. Your task is to act as a second-opinion analyst: verify the \
detection, identify the precise root cause, and prescribe an immediate remediation.

══════════════════ DETECTION CONTEXT ══════════════════
Model          : BiLSTM-FiLM Autoencoder (Alibaba Cloud Trace 2018)
Detection Time : {timestamp_utc}
Severity Tier  : {severity}  (calibrated percentile thresholds: P95/P98/P99.5)
MSE Score      : {mse:.6f}  |  P95 Threshold : {threshold_p95:.6f}
MSE Exceedance : {exceedance:.1f}× above threshold

══════════════════ CONTAINER METADATA (FiLM Context) ══════════════════
Container ID   : {container_id}
Container Type : {container_type}  ({container_display})
Tier           : {tier}
Environment    : {environment}
Pod Name       : {pod_name}
Namespace      : {namespace}
FiLM Vector    : {film_vector}  (conditions the model's reconstruction baseline)
Context Note   : {film_note}

══════════════════ FEATURE-LEVEL RECONSTRUCTION ERRORS ══════════════════
(Per-feature MSE — higher value = model could not reconstruct that metric)
{feature_table}

Primary Anomalous Feature : {primary_metric}  ({primary_pct:.1f}% of total error)
Statistical Diagnosis     : {stat_diagnosis}
Suggested Action (SRE KB) : {stat_action}

══════════════════ RECENT CONTAINER LOGS (tail -20) ══════════════════
```
{recent_logs}
```

══════════════════ INSTRUCTIONS ══════════════════
Based on ALL of the above evidence — the statistical anomaly score, the feature-level \
breakdown, the FiLM container context, and the log entries — \
provide your analysis in EXACTLY the following format:

VERDICT: <GENUINE ANOMALY|FALSE POSITIVE|UNCERTAIN>
ROOT CAUSE: <Exactly two sentences explaining what happened and why, referencing specific metrics and log entries where applicable.>
MITIGATION: <Exactly one concrete sentence describing what the on-call engineer should do RIGHT NOW.>
CONFIDENCE: <Integer percentage 0-100 reflecting your confidence in this analysis>
SEVERITY: <Critical|High|Warning|Normal> (your assessed severity — may override the model's initial tier based on contextual log evidence and metric trajectories)

Do NOT add any other text, preamble, or explanation outside this format.\
"""

# Fallback stories used when no live LLM is available.
DEMO_NARRATIVES: Dict[str, Dict[str, Tuple[str, str]]] = {
    "database": {
        "mem_util_percent": (
            "The PostgreSQL container is experiencing severe buffer pool exhaustion, "
            "with heap memory climbing monotonically to {pct}% "
            "as evidenced by the OOM-killer log entries and the monotonic reconstruction "
            "error spike on the mem_util_percent feature. "
            "The kernel's page reclaimer has been invoked, triggering checkpoint storms "
            "and secondary disk I/O saturation that is further elevating the MSE across "
            "multiple correlated features.",
            "Immediately execute: ALTER SYSTEM SET shared_buffers = '2GB'; "
            "SELECT pg_reload_conf(); and restart any long-running transactions "
            "identified in pg_stat_activity WHERE state = 'idle in transaction'.",
        ),
        "disk_io_percent": (
            "The database is experiencing a WAL checkpoint storm, with disk I/O saturated "
            "by excessive checkpoint writes triggered by a long-running VACUUM or a large "
            "bulk INSERT that has filled the WAL buffer. "
            "Log entries confirm checkpoint completion at >99% dirty page ratio, indicating "
            "the checkpoint_completion_target is misconfigured for current write throughput.",
            "Set checkpoint_completion_target=0.9 and max_wal_size='4GB'; "
            "then identify and terminate the offending bulk operation via "
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE wait_event_type='Lock'.",
        ),
    },
    "api": {
        "cpu_util_percent": (
            "The API gateway is experiencing CPU saturation caused by a sudden traffic surge "
            "or a runaway request handler entering an infinite retry loop against a "
            "slow downstream service, as evidenced by the sharp step-function rise "
            "in cpu_util_percent reconstruction error in the final 20 timesteps. "
            "The correlated net_in elevation indicates the saturation is ingress-driven, "
            "not an internal compute leak.",
            "Immediately scale the API deployment: kubectl scale deployment api-gateway "
            "--replicas=+3, then add a rate-limit annotation to the ingress resource "
            "to shed load while the root cause is investigated.",
        ),
        "net_in": (
            "The API service is receiving an abnormal inbound traffic volume consistent "
            "with a DDoS amplification attack or a misconfigured upstream retry storm, "
            "with net_in reconstruction error dominating at over 60% of total MSE. "
            "The CPU co-elevation confirms the service is actively processing these "
            "requests rather than dropping them at the network layer.",
            "Apply an ingress rate-limit immediately: kubectl annotate ingress api-ingress "
            "nginx.ingress.kubernetes.io/limit-rpm=200 and notify the Security team "
            "to investigate source IPs via kubectl logs ingress-nginx-controller.",
        ),
    },
    "cache": {
        "mem_util_percent": (
            "The Redis cache is approaching its maxmemory limit due to key-space explosion — "
            "likely caused by application code generating unique keys without TTL expiry, "
            "as the gradual monotonic memory ramp indicates accumulation rather than "
            "a sudden load spike. "
            "The elevated eviction rate is causing application-layer cache miss storms "
            "that are indirectly visible in the net_in reconstruction error.",
            "Connect immediately: redis-cli INFO stats | grep evicted_keys; "
            "then run redis-cli --bigkeys to identify the largest key pattern "
            "and add TTL: redis-cli EXPIRE <key-pattern> 3600.",
        ),
    },
    "worker": {
        "disk_io_percent": (
            "The Kafka consumer pod's disk I/O is saturated by a batch job writing "
            "large intermediate result sets directly to the pod's local filesystem "
            "rather than streaming to object storage, as the sustained high-frequency "
            "oscillation in disk_io reconstruction error indicates repeated flush cycles. "
            "The low net_out confirms results are not being shipped externally.",
            "Redirect batch output to S3/GCS by setting OUTPUT_SINK=s3://bucket/path "
            "in the job's ConfigMap, then restart the pod to apply the change.",
        ),
        "net_out": (
            "The worker pod is exhibiting anomalous outbound network traffic consistent "
            "with unintended data exfiltration or a misconfigured streaming sink "
            "sending data to an unexpected endpoint — workers should have near-zero "
            "net_out, making this the clearest possible anomaly signal. "
            "The abrupt onset (final 25 timesteps) rules out gradual misconfiguration "
            "and points to a runtime code path being triggered for the first time.",
            "Immediately isolate the pod: kubectl label pod worker-batch-001 quarantine=true "
            "and capture network traffic: kubectl exec -- ss -tnp to identify the "
            "destination before terminating the connection.",
        ),
    },
    "ml_inference": {
        "mem_util_percent": (
            "The model server (Triton/TorchServe) is failing to release memory between "
            "inference requests, causing progressive heap growth that mirrors a classical "
            "PyTorch CUDA memory leak where tensors are retained on the GPU between batches. "
            "The log-confirmed OOM signals indicate the process will be killed imminently "
            "unless model versions are unloaded.",
            "Immediately unload non-active model versions: "
            "curl -X DELETE http://triton:8000/v2/models/<model-name>/versions/<ver> "
            "and reduce max_batch_size by 50% in the model configuration.",
        ),
    },
}
