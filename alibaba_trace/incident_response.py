"""
incident_response.py — Context-Aware Edition
=============================================
Alerting and Root Cause Analysis (RCA) Engine
BiLSTM-FiLM Container Anomaly Detection System · Alibaba Cloud Trace 2018

DESIGN PHILOSOPHY — The "Common General-Purpose Engine" Concept
---------------------------------------------------------------
The BiLSTM-FiLM autoencoder is a *single* model trained on normal traffic
from ALL container types simultaneously.  Its FiLM (Feature-wise Linear
Modulation) layer injects a metadata vector at inference time so the
model can condition its reconstruction on the container's context — its
type, tier, and operating environment.

This module mirrors that philosophy at the *alerting* tier:

  1.  The same ``IncidentPipeline`` processes every container type.
  2.  The ``ContainerContext`` metadata (fed to FiLM during inference)
      is carried forward into the RCA and alert payload.
  3.  ``ContextAwareRCA`` selects per-type, per-feature diagnoses and
      remediation playbooks from ``CONTAINER_PROFILES``, so an identical
      CPU anomaly triggers:
        – "Scale API replicas"  for an API gateway
        – "Check DB query plans" for a PostgreSQL worker
        – "Inspect batch job"   for a Kafka consumer

Architecture Flow
-----------------

  BiLSTM-FiLM model                 IncidentPipeline
  ─────────────────────────────     ──────────────────────────────────────────
  original window (W, F)       →    SeverityScorer   → severity tier
  + FiLM meta vector (M,)      →    ContextAwareRCA  → diagnosis + action
               ↓                    AlertGenerator   → AM webhook JSON
  reconstruction (W, F)        →    IncidentEvent    → immutable record

Public API
----------
  ContainerContext      : metadata wrapper (semantic + FiLM vector)
  SeverityScorer        : calibrates P95/P98/P99.5; scores MSE → tier
  ContextAwareRCA       : per-type, per-feature root cause attribution
  AlertGenerator        : Prometheus AlertManager v4 JSON builder
  IncidentEvent         : frozen dataclass (one processed anomaly)
  IncidentPipeline      : orchestrates all components
  CONTAINER_PROFILES    : operational knowledge-base for RCA

References
----------
  Prometheus AlertManager webhook:
    https://prometheus.io/docs/alerting/latest/configuration/#webhook_config
  FiLM: "Visual Reasoning with a General Conditioning Layer"
    Perez et al., AAAI 2018. https://arxiv.org/abs/1709.07871
  Alibaba Cluster Trace 2018:
    https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018
"""

from __future__ import annotations

import json
import uuid
import logging
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("incident_response")


# ===========================================================================
# ─── FEATURE INDEX MAP ───────────────────────────────────────────────────────
# Must match the FEATURE_COLS order defined in data_pipeline.py:
#   ["cpu_util_percent", "mem_util_percent", "cpu_request", "mem_request",
#    "net_in", "net_out", "disk_io_percent"]
# ===========================================================================

#: Feature display names, icons, and per-feature severity weights.
#: ``severity_weight`` amplifies features that tend to be more operationally
#: critical (e.g., memory > disk), making the weighted ranking more meaningful.
FEATURE_META: Dict[str, Dict] = {
    "cpu_util_percent": {
        "display_name": "CPU Utilisation (%)",
        "short_name":   "CPU%",
        "icon":         "🔥",
        "severity_weight": 1.30,
    },
    "mem_util_percent": {
        "display_name": "Memory Utilisation (%)",
        "short_name":   "Mem%",
        "icon":         "💾",
        "severity_weight": 1.40,
    },
    "cpu_request": {
        "display_name": "CPU Request (cores)",
        "short_name":   "CPUReq",
        "icon":         "⚙️",
        "severity_weight": 1.00,
    },
    "mem_request": {
        "display_name": "Memory Request (GiB)",
        "short_name":   "MemReq",
        "icon":         "📋",
        "severity_weight": 1.00,
    },
    "net_in": {
        "display_name": "Network Inbound (MB/s)",
        "short_name":   "Net↓",
        "icon":         "📥",
        "severity_weight": 1.20,
    },
    "net_out": {
        "display_name": "Network Outbound (MB/s)",
        "short_name":   "Net↑",
        "icon":         "📤",
        "severity_weight": 1.20,
    },
    "disk_io_percent": {
        "display_name": "Disk I/O Utilisation (%)",
        "short_name":   "Disk%",
        "icon":         "💿",
        "severity_weight": 1.10,
    },
}

_UNKNOWN_META = {
    "display_name":    "Unknown Metric",
    "short_name":      "Unknown",
    "icon":            "❓",
    "severity_weight": 1.0,
}


# ===========================================================================
# ─── CONTAINER PROFILES ──────────────────────────────────────────────────────
#
# This is the operational knowledge-base that makes the engine context-aware.
#
# Structure
# ---------
# CONTAINER_PROFILES[container_type][feature_name] → {
#     anomaly_name   : short label for alerts
#     diagnosis      : root cause explanation tailored to this workload type
#     action         : step-by-step remediation for on-call engineers
#     escalation     : recommended escalation path
# }
#
# Adding a new container type requires only a new top-level key here —
# the pipeline code does not need to change.  This extensibility is a
# direct benefit of the FiLM "Common Engine" design.
# ===========================================================================

CONTAINER_PROFILES: Dict[str, Dict] = {

    # ── API Gateway / REST Services ──────────────────────────────────────────
    "api": {
        "_meta": {
            "display_name":  "REST API / Gateway Service",
            "tier_default":  "frontend",
            "profile_icon":  "🌐",
            "description":   (
                "Stateless HTTP/gRPC request handlers. High net_in signals "
                "traffic surges; CPU spikes indicate handler computation or "
                "downstream blocking calls."
            ),
            "runbook_base":  "https://runbooks.internal/api",
        },
        "cpu_util_percent": {
            "anomaly_name": "API CPU Surge",
            "diagnosis":    (
                "CPU saturation in the API tier — likely caused by a sudden "
                "traffic spike, an expensive synchronous computation in a "
                "request handler, or a missing cache for a hot code path."
            ),
            "action":       (
                "1. kubectl scale deployment <api-svc> --replicas=+2\n"
                "2. Check HPA policy: kubectl describe hpa <api-svc>\n"
                "3. Profile slowest endpoints: kubectl exec -- py-spy top\n"
                "4. Enable connection pooling if downstream DB calls are slow."
            ),
            "escalation":   "SRE → Backend Engineering Team",
        },
        "mem_util_percent": {
            "anomaly_name": "API Memory Leak",
            "diagnosis":    (
                "Memory pressure in the API tier — potential leak in request "
                "handler state, unbounded in-memory caching, or unclosed HTTP "
                "client sessions accumulating across requests."
            ),
            "action":       (
                "1. kubectl exec -- jmap -histo <pid> (JVM) or py-spy dump\n"
                "2. Check for unclosed HTTP sessions / connection pool leaks.\n"
                "3. Set max_requests (gunicorn) to recycle workers periodically.\n"
                "4. Restart pod if OOM imminent: kubectl rollout restart deploy/<svc>"
            ),
            "escalation":   "SRE → Backend Engineering Team",
        },
        "net_in": {
            "anomaly_name": "API Inbound Flood",
            "diagnosis":    (
                "Abnormal inbound network traffic to the API service — possible "
                "DDoS attack, a client retry storm, or a misconfigured upstream "
                "load balancer sending excess traffic."
            ),
            "action":       (
                "1. Apply ingress rate-limit: kubectl annotate ingress <ing> "
                "nginx.ingress.kubernetes.io/limit-rps=100\n"
                "2. Check access logs for anomalous source IPs.\n"
                "3. Enable WAF ruleset for suspicious patterns.\n"
                "4. Alert SecOps if DDoS suspected."
            ),
            "escalation":   "SRE → Network Security Team",
        },
        "net_out": {
            "anomaly_name": "API Outbound Surge",
            "diagnosis":    (
                "API service is generating excessive outbound traffic — often "
                "caused by large response payloads, polling loops against "
                "downstream services, or misconfigured bulk data exports."
            ),
            "action":       (
                "1. Inspect Istio/Linkerd telemetry for top destination services.\n"
                "2. Enable response compression (gzip) if payloads are large.\n"
                "3. Add pagination / streaming for large collection endpoints.\n"
                "4. Alert SecOps if unexpected egress destination is observed."
            ),
            "escalation":   "SRE → Backend Engineering Team",
        },
        "cpu_request": {
            "anomaly_name": "API CPU Request Deviation",
            "diagnosis":    "Actual CPU consumption diverges from declared requests — PodSpec likely under-provisioned for current traffic level.",
            "action":       "1. Run VPA recommendation: kubectl describe vpa <api-svc>\n2. Update resources.requests.cpu to 95th-percentile observed usage.",
            "escalation":   "SRE → Platform Engineering",
        },
        "mem_request": {
            "anomaly_name": "API Memory Request Deviation",
            "diagnosis":    "Memory footprint exceeds declared requests — risk of OOMKill or node memory pressure.",
            "action":       "1. kubectl top pod -n <ns> --containers\n2. Update resources.requests.memory; set limit = 1.5× request.",
            "escalation":   "SRE → Platform Engineering",
        },
        "disk_io_percent": {
            "anomaly_name": "API Disk I/O Anomaly",
            "diagnosis":    "Unexpected disk activity in a typically stateless API service — check for runaway debug logging or unintended local file writes.",
            "action":       "1. Review logging verbosity; set LOG_LEVEL=WARNING.\n2. Check for accidental local file cache writes in handler code.",
            "escalation":   "SRE → Backend Engineering Team",
        },
    },

    # ── Database Services (PostgreSQL, MySQL, etc.) ───────────────────────────
    "database": {
        "_meta": {
            "display_name": "Relational Database Service",
            "tier_default": "data",
            "profile_icon": "🗄️",
            "description":  (
                "Stateful RDBMS workloads. High mem signals buffer pool pressure; "
                "disk_io spikes indicate checkpoint storms or full table scans."
            ),
            "runbook_base": "https://runbooks.internal/database",
        },
        "cpu_util_percent": {
            "anomaly_name": "DB Query CPU Surge",
            "diagnosis":    (
                "Database CPU saturation — likely caused by an unindexed query, "
                "a complex aggregation (GROUP BY / ORDER BY without index), or "
                "a sudden increase in concurrent connections."
            ),
            "action":       (
                "1. Run EXPLAIN ANALYZE on the 5 slowest queries (pg_stat_statements).\n"
                "2. Check for missing indexes: SELECT * FROM pg_stat_user_indexes;\n"
                "3. Increase max_parallel_workers_per_gather if read-heavy.\n"
                "4. Consider read-replica offloading for reporting queries."
            ),
            "escalation":   "DBA Team → Backend Engineering",
        },
        "mem_util_percent": {
            "anomaly_name": "DB Buffer Pool Exhaustion",
            "diagnosis":    (
                "Database memory pressure — the shared buffer pool (PostgreSQL: "
                "shared_buffers) or InnoDB buffer pool is filling up, forcing "
                "eviction of hot pages and causing excessive disk reads."
            ),
            "action":       (
                "1. Check buffer cache hit rate: SELECT blks_hit, blks_read FROM pg_stat_database;\n"
                "2. Increase shared_buffers to 25% of available RAM.\n"
                "3. Review work_mem — per-query memory can accumulate under high concurrency.\n"
                "4. Scale DB pod memory request if node has capacity."
            ),
            "escalation":   "DBA Team → Infrastructure",
        },
        "disk_io_percent": {
            "anomaly_name": "DB Disk I/O Storm",
            "diagnosis":    (
                "Disk I/O saturation — possible WAL checkpoint flood, a long-running "
                "VACUUM, a full table scan on a large relation, or a backup job "
                "competing with OLTP workload."
            ),
            "action":       (
                "1. Check for active VACUUM: SELECT * FROM pg_stat_progress_vacuum;\n"
                "2. Delay checkpoint: checkpoint_completion_target=0.9\n"
                "3. Schedule VACUUM ANALYZE during off-peak hours.\n"
                "4. Migrate to SSD-backed PVC StorageClass if I/O persists."
            ),
            "escalation":   "DBA Team → Infrastructure",
        },
        "net_in": {
            "anomaly_name": "DB Inbound Connection Surge",
            "diagnosis":    "Abnormal inbound connections to the database — connection pool exhaustion or a connection leak in the application tier.",
            "action":       "1. Check pg_stat_activity for idle connections.\n2. Enforce max_connections and deploy PgBouncer connection pooler.\n3. Set idle_in_transaction_session_timeout=10s.",
            "escalation":   "DBA Team → Backend Engineering",
        },
        "net_out": {
            "anomaly_name": "DB Outbound Data Surge",
            "diagnosis":    "Excessive data egress from database — possible large result set being returned without pagination or an rogue analytics query.",
            "action":       "1. Check for queries returning > 10k rows without LIMIT.\n2. Enable statement_timeout for long-running queries.",
            "escalation":   "DBA Team → Backend Engineering",
        },
        "cpu_request": {
            "anomaly_name": "DB CPU Request Deviation",
            "diagnosis":    "DB CPU consumption exceeds declared requests — query workload has grown beyond initial capacity plan.",
            "action":       "1. Profile pg_stat_statements for top CPU-consuming queries.\n2. Update resource requests; evaluate vertical scaling.",
            "escalation":   "DBA Team",
        },
        "mem_request": {
            "anomaly_name": "DB Memory Request Deviation",
            "diagnosis":    "Database memory usage exceeds declared requests — shared_buffers or per-query work_mem tuning required.",
            "action":       "1. Audit PostgreSQL memory parameters.\n2. Increase pod memory limit on next maintenance window.",
            "escalation":   "DBA Team → Infrastructure",
        },
    },

    # ── Cache Services (Redis, Memcached) ─────────────────────────────────────
    "cache": {
        "_meta": {
            "display_name": "In-Memory Cache Service",
            "tier_default": "backend",
            "profile_icon": "⚡",
            "description":  (
                "Redis / Memcached in-memory stores. Net_in spikes signal "
                "client request floods; mem elevation signals eviction pressure "
                "or key space explosion."
            ),
            "runbook_base": "https://runbooks.internal/cache",
        },
        "mem_util_percent": {
            "anomaly_name": "Cache Memory Saturation",
            "diagnosis":    (
                "Cache memory is approaching its maxmemory limit — keys are "
                "being evicted aggressively (volatile-lru / allkeys-lru), "
                "or a key-space explosion has occurred (e.g., unique session keys "
                "not expiring properly)."
            ),
            "action":       (
                "1. Redis CLI: INFO memory — check used_memory vs maxmemory.\n"
                "2. Audit TTL policy: OBJECT FREQ <hotkey> — ensure TTLs are set.\n"
                "3. Scan for large keys: redis-cli --bigkeys\n"
                "4. Scale cache horizontally via Redis Cluster sharding."
            ),
            "escalation":   "SRE → Backend Engineering Team",
        },
        "net_in": {
            "anomaly_name": "Cache Request Flood",
            "diagnosis":    (
                "Abnormally high inbound request rate to the cache — possibly caused "
                "by a wildcard KEYS scan, a SUBSCRIBE storm, or a 'thundering herd' "
                "after a cache flush event."
            ),
            "action":       (
                "1. Run Redis Monitor briefly: redis-cli MONITOR | head -100\n"
                "2. Check for KEYS * usage in app code — replace with SCAN.\n"
                "3. Implement request coalescing (cache stampede prevention).\n"
                "4. Enable Redis client-side caching if supported."
            ),
            "escalation":   "SRE → Backend Engineering Team",
        },
        "cpu_util_percent": {
            "anomaly_name": "Cache CPU Spike",
            "diagnosis":    "CPU saturation on cache instance — often caused by expensive Lua scripts, SORT commands, or blocking operations on large lists.",
            "action":       "1. Identify slow commands: redis-cli SLOWLOG GET 20\n2. Replace SORT with application-side sorting.\n3. Use pipelining to reduce round-trips.",
            "escalation":   "SRE → Backend Engineering Team",
        },
        "net_out": {
            "anomaly_name": "Cache Outbound Data Surge",
            "diagnosis":    "Cache returning abnormally large payloads — possible storage of oversized values or large LRANGE / SMEMBERS scans.",
            "action":       "1. Audit key sizes: redis-cli DEBUG OBJECT <key>\n2. Enforce max value size limits at the client layer.\n3. Use compression for large cached objects.",
            "escalation":   "SRE → Backend Engineering Team",
        },
        "disk_io_percent": {
            "anomaly_name": "Cache Persistence I/O",
            "diagnosis":    "High disk I/O from cache — Redis AOF or RDB snapshot is conflicting with serving traffic. fsync is blocking event loop.",
            "action":       "1. Tune appendfsync: set to everysec instead of always.\n2. Schedule BGSAVE during off-peak.\n3. Increase no-appendfsync-on-rewrite yes.",
            "escalation":   "SRE → Infrastructure",
        },
        "cpu_request": {
            "anomaly_name": "Cache CPU Request Deviation",
            "diagnosis":    "Cache CPU consumption higher than declared — workload growth or new command patterns not accounted for in capacity plan.",
            "action":       "1. Review recent keyspace changes.\n2. Update resource requests based on observed 95th-percentile.",
            "escalation":   "SRE",
        },
        "mem_request": {
            "anomaly_name": "Cache Memory Request Deviation",
            "diagnosis":    "Cache memory footprint exceeds declared requests — maxmemory may be higher than pod memory request, risking OOMKill.",
            "action":       "1. Align maxmemory with pod limit: set maxmemory = 0.85 × pod limit.\n2. Increase pod memory request.",
            "escalation":   "SRE → Infrastructure",
        },
    },

    # ── Background Worker / Message Consumer ─────────────────────────────────
    "worker": {
        "_meta": {
            "display_name": "Background Worker / Message Consumer",
            "tier_default": "infrastructure",
            "profile_icon": "⚙️",
            "description":  (
                "Kafka/RabbitMQ consumers, batch processors, cron jobs. "
                "Disk I/O anomalies indicate batch I/O thrashing; "
                "CPU spikes point to compute-heavy job bursts."
            ),
            "runbook_base": "https://runbooks.internal/worker",
        },
        "disk_io_percent": {
            "anomaly_name": "Worker Batch I/O Thrash",
            "diagnosis":    (
                "Disk I/O saturation on worker pod — likely caused by a large "
                "batch job writing intermediate results to local disk, log "
                "rotation conflict, or uncontrolled output buffering."
            ),
            "action":       (
                "1. Identify top I/O processes: kubectl exec -- iotop -o\n"
                "2. Redirect intermediate results to object storage (S3/GCS).\n"
                "3. Set ulimit -n to prevent file descriptor exhaustion.\n"
                "4. Migrate batch to PVC backed by SSD StorageClass."
            ),
            "escalation":   "SRE → Data Engineering Team",
        },
        "cpu_util_percent": {
            "anomaly_name": "Worker Job CPU Burst",
            "diagnosis":    (
                "CPU saturation in worker — a batch job is compute-intensive "
                "(e.g., ML feature extraction, JSON parsing at scale, or "
                "unoptimised MapReduce step)."
            ),
            "action":       (
                "1. Profile job with py-spy or async-profiler.\n"
                "2. Parallelize work across multiple pods via Kubernetes Job --parallelism.\n"
                "3. Add CPU limit to prevent node-level starvation.\n"
                "4. Consider GPU offloading for ML preprocessing tasks."
            ),
            "escalation":   "SRE → Data Engineering Team",
        },
        "mem_util_percent": {
            "anomaly_name": "Worker Memory Overrun",
            "diagnosis":    "Worker is loading an excessively large dataset into memory — common with unbounded Pandas DataFrames or accumulating Kafka messages.",
            "action":       "1. Switch to chunked processing: pd.read_csv(chunksize=...).\n2. Implement back-pressure on the message consumer.\n3. Increase pod memory limit and set eviction threshold.",
            "escalation":   "SRE → Data Engineering Team",
        },
        "net_in": {
            "anomaly_name": "Worker Message Ingestion Spike",
            "diagnosis":    "Consumer is receiving an unusually large volume of messages — possible backlog catch-up or upstream producer anomaly.",
            "action":       "1. Check consumer group lag: kafka-consumer-groups.sh --describe\n2. Scale consumer group partitions to distribute load.\n3. Implement dead-letter queue for poison messages.",
            "escalation":   "SRE → Data Engineering Team",
        },
        "net_out": {
            "anomaly_name": "Worker Outbound Data Surge",
            "diagnosis":    "Worker producing excessive outbound data — large result set being written to downstream storage or streaming sink.",
            "action":       "1. Implement output batching to reduce small writes.\n2. Check for infinite retry loops writing duplicates.",
            "escalation":   "SRE → Data Engineering Team",
        },
        "cpu_request": {
            "anomaly_name": "Worker CPU Request Deviation",
            "diagnosis":    "Batch job CPU demands exceed declared requests — job sizing or schedule needs review.",
            "action":       "1. Profile representative job run.\n2. Update Job resource template with observed peak CPU.",
            "escalation":   "Data Engineering Team",
        },
        "mem_request": {
            "anomaly_name": "Worker Memory Request Deviation",
            "diagnosis":    "Worker memory consumption exceeds declared requests — dataset size has grown beyond initial estimates.",
            "action":       "1. Add RESOURCE_LIMIT env var check at job init.\n2. Raise memory request; enable chunked processing.",
            "escalation":   "Data Engineering Team",
        },
    },

    # ── ML Inference / Model Serving ──────────────────────────────────────────
    "ml_inference": {
        "_meta": {
            "display_name": "ML Inference / Model Serving",
            "tier_default": "backend",
            "profile_icon": "🤖",
            "description":  (
                "Triton / TorchServe / BentoML model servers. CPU/Memory spikes "
                "indicate model OOM or batch size misconfiguration; "
                "net_in spikes signal inference request surges."
            ),
            "runbook_base": "https://runbooks.internal/ml-inference",
        },
        "mem_util_percent": {
            "anomaly_name": "Model Server OOM Risk",
            "diagnosis":    (
                "Model server memory pressure — the loaded model(s) + inference "
                "batch are exceeding available pod memory. Risk of OOMKill causing "
                "dropped requests. Often caused by loading multiple model versions "
                "simultaneously or a large dynamic batch accumulating."
            ),
            "action":       (
                "1. Check loaded models: curl http://triton:8000/v2/models\n"
                "2. Unload unused model versions: DELETE /v2/models/<name>\n"
                "3. Reduce max_batch_size in model config.pbtxt.\n"
                "4. Enable model instance pinning to prevent re-loading.\n"
                "5. Scale horizontally: kubectl scale deploy/triton --replicas=+1"
            ),
            "escalation":   "SRE → ML Platform Team",
        },
        "cpu_util_percent": {
            "anomaly_name": "Inference CPU Saturation",
            "diagnosis":    (
                "CPU saturation on model server — inference is bottlenecked on "
                "CPU (pre/post-processing, tokenisation, non-CUDA ops). The model "
                "may be running on CPU fallback if GPU is unavailable."
            ),
            "action":       (
                "1. Verify GPU assignment: kubectl describe pod | grep nvidia\n"
                "2. Check model backend: if CPU fallback, reschedule on GPU node.\n"
                "3. Enable ONNX optimisation or TensorRT conversion.\n"
                "4. Reduce concurrent request count via queue depth limit."
            ),
            "escalation":   "SRE → ML Platform Team",
        },
        "net_in": {
            "anomaly_name": "Inference Request Surge",
            "diagnosis":    (
                "Inbound inference request rate has spiked beyond model server "
                "capacity — queue is growing, latency SLO at risk."
            ),
            "action":       (
                "1. Check Triton queue depth metrics: triton_request_queue_size\n"
                "2. Scale model server replicas via HPA on infer_request_rate.\n"
                "3. Enable dynamic batching with delay_ns tuning.\n"
                "4. Add circuit breaker at API gateway to shed load gracefully."
            ),
            "escalation":   "SRE → ML Platform Team",
        },
        "disk_io_percent": {
            "anomaly_name": "Model Loading I/O Spike",
            "diagnosis":    "High disk I/O on ML server — a large model checkpoint is being loaded from disk, or model weights are being swapped.",
            "action":       "1. Pre-warm models at pod startup (readinessProbe).\n2. Store model weights in tmpfs or RAM-disk mount.\n3. Use model caching (model repository on NVMe SSD).",
            "escalation":   "SRE → ML Platform Team",
        },
        "net_out": {
            "anomaly_name": "Inference Response Data Surge",
            "diagnosis":    "Model server returning abnormally large responses — possible embedding or probability distribution output not being truncated.",
            "action":       "1. Audit output tensor shapes in model config.\n2. Enable output compression for embedding endpoints.\n3. Implement response streaming for large outputs.",
            "escalation":   "SRE → ML Platform Team",
        },
        "cpu_request": {
            "anomaly_name": "ML CPU Request Deviation",
            "diagnosis":    "Inference CPU footprint exceeds declared requests — pre/post-processing overhead has grown with new model version.",
            "action":       "1. Profile with NVIDIA Nsight / py-spy.\n2. Update resource requests to reflect new model CPU needs.",
            "escalation":   "ML Platform Team",
        },
        "mem_request": {
            "anomaly_name": "ML Memory Request Deviation",
            "diagnosis":    "Model + runtime memory exceeds declared requests — new model version is larger than previous.",
            "action":       "1. Run model memory audit: model_size + 2× batch_size × tensor_footprint.\n2. Update pod memory request before deploying new model.",
            "escalation":   "ML Platform Team",
        },
    },
}

#: Fallback profile when container_type is not in CONTAINER_PROFILES
_GENERIC_PROFILE: Dict = {
    "_meta": {
        "display_name": "Generic Container",
        "tier_default": "backend",
        "profile_icon": "📦",
        "description":  "Unknown container type — applying generic remediation guidance.",
        "runbook_base": "https://runbooks.internal/general",
    },
}

_GENERIC_FEATURE_FALLBACK: Dict = {
    "anomaly_name": "Resource Anomaly",
    "diagnosis":    "Anomalous reconstruction error detected on this feature. Manual investigation required.",
    "action":       "1. kubectl top pod -n <ns> --containers\n2. Review pod events: kubectl describe pod <pod>\n3. Inspect application logs for error patterns.",
    "escalation":   "SRE Team",
}


# ===========================================================================
# ─── DATA CLASS: ContainerContext ────────────────────────────────────────────
# ===========================================================================

@dataclass
class ContainerContext:
    """
    Carries the metadata that was fed to the BiLSTM-FiLM layer during
    inference, plus higher-level semantic labels.

    This is the bridge between the model's FiLM conditioning vector and
    the context-aware RCA and alert generation components.

    Parameters
    ----------
    container_id    : Unique string identifier (Alibaba format: e.g. "c-abc123")
    machine_id      : Host machine identifier
    container_type  : One of: "api", "database", "cache", "worker", "ml_inference"
    tier            : Deployment tier: "frontend", "backend", "data", "infrastructure"
    environment     : "production" | "staging" | "development"
    namespace       : Kubernetes namespace
    pod_name        : Kubernetes pod name
    film_meta_vector: np.ndarray shape (M,) — the actual float vector fed to the
                       FiLM layer.  Typically [hash(container_id)/P, hash(machine_id)/P]
                       where P is a large prime, giving values in (0, 1).
    """
    container_id:     str
    machine_id:       str
    container_type:   str   # Key into CONTAINER_PROFILES
    tier:             str
    environment:      str
    namespace:        str
    pod_name:         str
    film_meta_vector: np.ndarray  # Shape (M,)

    # ── Derived helpers ────────────────────────────────────────────────────
    def profile(self) -> Dict:
        """Return the CONTAINER_PROFILES entry (falls back to generic)."""
        return CONTAINER_PROFILES.get(self.container_type, _GENERIC_PROFILE)

    def profile_meta(self) -> Dict:
        """Return the _meta sub-dict of this container type's profile."""
        return self.profile().get("_meta", _GENERIC_PROFILE["_meta"])

    def display_name(self) -> str:
        return self.profile_meta().get("display_name", "Unknown")

    def profile_icon(self) -> str:
        return self.profile_meta().get("profile_icon", "📦")

    def film_vector_str(self) -> str:
        """Human-readable representation of the FiLM metadata vector."""
        vals = ", ".join(f"{v:.4f}" for v in self.film_meta_vector)
        return f"[{vals}]"

    def environment_tier_label(self) -> str:
        """Compact display label for the Incident Log table."""
        return f"{self.environment.capitalize()} / {self.tier}"

    def to_dict(self) -> Dict:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "container_id":     self.container_id,
            "machine_id":       self.machine_id,
            "container_type":   self.container_type,
            "container_type_display": self.display_name(),
            "tier":             self.tier,
            "environment":      self.environment,
            "namespace":        self.namespace,
            "pod_name":         self.pod_name,
            "film_meta_vector": self.film_vector_str(),
            "film_meta_dim":    len(self.film_meta_vector),
            "profile_icon":     self.profile_icon(),
        }

    @classmethod
    def from_ids(
        cls,
        container_id:   str,
        machine_id:     str,
        container_type: str,
        tier:           str            = "backend",
        environment:    str            = "production",
        namespace:      str            = "default",
        pod_name:       Optional[str]  = None,
    ) -> "ContainerContext":
        """
        Factory: build a ContainerContext and compute the FiLM meta vector
        from the raw IDs using the same hash-encoding as data_pipeline.py.

        The encoding is:  hash(id) % LARGE_PRIME / LARGE_PRIME  → (0, 1)
        """
        LARGE_PRIME = 999_983
        cid_val = float(hash(container_id) % LARGE_PRIME) / LARGE_PRIME
        mid_val = float(hash(machine_id)   % LARGE_PRIME) / LARGE_PRIME
        film_vec = np.array([cid_val, mid_val], dtype=np.float32)

        return cls(
            container_id=container_id,
            machine_id=machine_id,
            container_type=container_type,
            tier=tier,
            environment=environment,
            namespace=namespace,
            pod_name=pod_name or f"{container_type}-pod-{container_id[:6]}",
            film_meta_vector=film_vec,
        )

    def __repr__(self) -> str:
        return (
            f"ContainerContext("
            f"container_id='{self.container_id}', "
            f"type='{self.container_type}', "
            f"tier='{self.tier}/{self.environment}', "
            f"film={self.film_vector_str()})"
        )


# ===========================================================================
# ─── DATA CLASS: IncidentEvent ───────────────────────────────────────────────
# ===========================================================================

@dataclass(frozen=True)
class IncidentEvent:
    """
    Immutable record for one fully processed anomaly incident.

    All fields are populated by ``IncidentPipeline.process()`` and are
    safe to serialise, store in a database, or forward to AlertManager.
    """
    event_id:           str
    timestamp_utc:      str
    context:            ContainerContext   # Full FiLM metadata
    severity:           str
    mse_score:          float
    primary_metric:     str               # FEATURE_COLS column name
    primary_display:    str               # Human-readable name
    anomaly_name:       str               # Profile-specific label
    diagnosis:          str               # Context-aware explanation
    recommended_action: str               # Step-by-step remediation
    escalation_path:    str               # Who to page
    icon:               str               # Feature icon emoji
    runbook_url:        str
    feature_errors:     Dict[str, float]  # {feature: per-feature MSE}
    alert_payload:      str               # Pre-rendered JSON string

    def to_alert_dict(self) -> Dict:
        return json.loads(self.alert_payload)

    def to_alert_json(self, indent: int = 2) -> str:
        return json.dumps(json.loads(self.alert_payload), indent=indent)

    def severity_colour(self) -> str:
        """ANSI terminal colour code for this severity."""
        return {
            "Critical": "\033[91m",
            "High":     "\033[93m",
            "Warning":  "\033[94m",
            "Normal":   "\033[92m",
        }.get(self.severity, "")

    def __str__(self) -> str:
        reset = "\033[0m"
        c = self.severity_colour()
        return (
            f"{c}[{self.severity:>8}]{reset} "
            f"{self.context.profile_icon()} {self.context.container_type:>12} | "
            f"{self.context.container_id:20s} | "
            f"MSE={self.mse_score:.6f} | "
            f"{self.icon} {self.primary_display}"
        )


# ===========================================================================
# ─── COMPONENT 1: SeverityScorer ────────────────────────────────────────────
# ===========================================================================

class SeverityScorer:
    """
    Assigns a severity tier to a reconstruction MSE by comparing it
    against percentile thresholds derived from baseline (normal) errors.

    Severity Tiers
    --------------
    Normal   : MSE < P95   — within expected operational band
    Warning  : P95 ≤ MSE < P98  — elevated; monitor
    High     : P98 ≤ MSE < P99.5 — anomalous; page on-call
    Critical : MSE ≥ P99.5 — severe; immediate action required

    Parameters
    ----------
    p95, p98, p995 : float, optional — supply pre-computed thresholds
                                        to skip calibration.
    """

    #: Integer severity index used in alert labels for routing rules
    SEVERITY_INT: Dict[str, int] = {
        "Normal": 0, "Warning": 1, "High": 2, "Critical": 3
    }

    def __init__(
        self,
        p95:  Optional[float] = None,
        p98:  Optional[float] = None,
        p995: Optional[float] = None,
    ) -> None:
        self.p95  = p95
        self.p98  = p98
        self.p995 = p995
        self._calibrated = all(v is not None for v in [p95, p98, p995])

    def calibrate(self, baseline_mse: np.ndarray) -> "SeverityScorer":
        """
        Fit thresholds from a 1-D array of normal reconstruction errors.
        Must be called before ``score()`` unless thresholds were passed
        to the constructor.
        """
        arr = np.asarray(baseline_mse, dtype=np.float64).ravel()
        if len(arr) < 4:
            raise ValueError(f"Need ≥ 4 values, got {len(arr)}.")
        self.p95  = float(np.percentile(arr, 95.0))
        self.p98  = float(np.percentile(arr, 98.0))
        self.p995 = float(np.percentile(arr, 99.5))
        self._calibrated = True
        logger.info(
            "SeverityScorer calibrated — P95=%.6f | P98=%.6f | P99.5=%.6f",
            self.p95, self.p98, self.p995,
        )
        return self

    def score(self, mse: float) -> str:
        """Return the severity string for a single MSE value."""
        if not self._calibrated:
            raise RuntimeError("Call .calibrate() first.")
        if mse >= self.p995: return "Critical"
        if mse >= self.p98:  return "High"
        if mse >= self.p95:  return "Warning"
        return "Normal"

    def score_batch(self, mse_array: np.ndarray) -> List[str]:
        """Vectorised scoring for a batch of MSE values."""
        return [self.score(float(m)) for m in mse_array]

    def threshold_summary(self) -> Dict[str, float]:
        if not self._calibrated:
            raise RuntimeError("Not calibrated.")
        return {"p95": self.p95, "p98": self.p98, "p99.5": self.p995}

    def __repr__(self) -> str:
        if self._calibrated:
            return (
                f"SeverityScorer(p95={self.p95:.6f}, "
                f"p98={self.p98:.6f}, p99.5={self.p995:.6f})"
            )
        return "SeverityScorer(uncalibrated)"


# ===========================================================================
# ─── COMPONENT 2: ContextAwareRCA ────────────────────────────────────────────
# ===========================================================================

class ContextAwareRCA:
    """
    Context-aware Root Cause Analysis using per-type, per-feature playbooks.

    This is the component that makes the "Common Engine" thesis concept
    concrete: the same class processes any container type, but the
    diagnosis and remediation it produces are tailored to the workload via
    the ``ContainerContext`` passed at analysis time.

    How feature attribution works
    ------------------------------
    Given original window O (W, F) and reconstruction R (W, F):

        raw_error_f  = mean((O - R)² over time axis)     → shape (F,)
        weighted_f   = raw_error_f × severity_weight_f

    Features are ranked by ``weighted_f``.  The top feature is the primary
    root cause; any feature whose weighted error exceeds
    ``secondary_threshold × max_weighted`` is also reported.

    The ``CONTAINER_PROFILES`` look-up then yields a context-specific
    diagnosis and playbook.

    Parameters
    ----------
    feature_cols        : list[str] — ordered feature column names
    secondary_threshold : float     — fraction of max for secondary causes
    """

    def __init__(
        self,
        feature_cols:        List[str],
        secondary_threshold: float = 0.60,
    ) -> None:
        self.feature_cols        = list(feature_cols)
        self.secondary_threshold = float(secondary_threshold)
        self._weights = np.array(
            [FEATURE_META.get(f, _UNKNOWN_META)["severity_weight"]
             for f in feature_cols],
            dtype=np.float64,
        )

    def analyze(
        self,
        original:      np.ndarray,
        reconstructed: np.ndarray,
        context:       ContainerContext,
    ) -> Dict:
        """
        Perform context-aware root cause attribution.

        Parameters
        ----------
        original      : np.ndarray (W, F) — original scaled window
        reconstructed : np.ndarray (W, F) — model reconstruction
        context       : ContainerContext  — FiLM metadata for this container

        Returns
        -------
        dict with keys:
          primary_metric       : str   — feature column name
          primary_display      : str   — human-readable name
          primary_error        : float — per-feature MSE
          primary_error_percent: float — share of total error
          anomaly_name         : str   — profile-specific short label
          diagnosis            : str   — context-aware explanation
          recommended_action   : str   — step-by-step playbook
          escalation_path      : str   — who to page
          icon                 : str   — emoji
          runbook_url          : str
          secondary_causes     : list[dict]
          feature_errors       : {feat: raw_mse}
          feature_errors_ranked: list[dict] sorted by weighted_mse desc
          context_summary      : dict — human-readable context metadata
        """
        original      = np.asarray(original,      dtype=np.float64)
        reconstructed = np.asarray(reconstructed, dtype=np.float64)
        if original.shape != reconstructed.shape:
            raise ValueError(
                f"Shape mismatch: original={original.shape} vs "
                f"reconstructed={reconstructed.shape}"
            )

        # ── Per-feature MSE ────────────────────────────────────────────────
        raw_errors      = np.mean((original - reconstructed) ** 2, axis=0)  # (F,)
        weighted_errors = raw_errors * self._weights

        total_mse = float(raw_errors.sum()) or 1e-12

        # ── Build ranked feature table ─────────────────────────────────────
        feature_errors:  Dict[str, float] = {}
        feature_ranked:  List[Dict]       = []

        for idx, feat in enumerate(self.feature_cols):
            fm   = FEATURE_META.get(feat, _UNKNOWN_META)
            r_e  = float(raw_errors[idx])
            w_e  = float(weighted_errors[idx])
            frac = r_e / total_mse

            feature_errors[feat] = r_e
            feature_ranked.append({
                "feature":      feat,
                "display_name": fm["display_name"],
                "short_name":   fm["short_name"],
                "icon":         fm["icon"],
                "mse":          round(r_e,  8),
                "weighted_mse": round(w_e,  8),
                "error_pct":    round(frac * 100, 2),
            })

        feature_ranked.sort(key=lambda x: x["weighted_mse"], reverse=True)

        # ── Primary culprit ────────────────────────────────────────────────
        primary      = feature_ranked[0]
        primary_feat = primary["feature"]

        # ── Context-aware playbook look-up ─────────────────────────────────
        profile_features = context.profile()
        feat_profile     = profile_features.get(primary_feat, _GENERIC_FEATURE_FALLBACK)

        profile_meta = context.profile_meta()
        runbook_url  = (
            profile_meta.get("runbook_base", "https://runbooks.internal") +
            "/" + primary_feat.replace("_", "-")
        )

        # ── Secondary causes ───────────────────────────────────────────────
        max_w = primary["weighted_mse"]
        secondary_causes = [
            {**f, **profile_features.get(f["feature"], _GENERIC_FEATURE_FALLBACK)}
            for f in feature_ranked[1:]
            if f["weighted_mse"] >= self.secondary_threshold * max_w
        ]

        # ── Context summary (for embedding in alert) ───────────────────────
        context_summary = {
            "container_type":         context.container_type,
            "container_type_display": context.display_name(),
            "tier":                   context.tier,
            "environment":            context.environment,
            "film_meta_vector":       context.film_vector_str(),
            "film_conditioning_note": (
                f"FiLM layer conditioned the BiLSTM reconstruction on "
                f"container_id='{context.container_id}' "
                f"(meta_vec={context.film_vector_str()}), enabling context-aware "
                f"anomaly detection for {context.display_name()} workloads."
            ),
        }

        return {
            "primary_metric":        primary_feat,
            "primary_display":       primary["display_name"],
            "primary_short":         primary["short_name"],
            "primary_error":         primary["mse"],
            "primary_error_percent": primary["error_pct"],
            "anomaly_name":          feat_profile.get("anomaly_name", "Resource Anomaly"),
            "diagnosis":             feat_profile.get("diagnosis",    _GENERIC_FEATURE_FALLBACK["diagnosis"]),
            "recommended_action":    feat_profile.get("action",       _GENERIC_FEATURE_FALLBACK["action"]),
            "escalation_path":       feat_profile.get("escalation",   _GENERIC_FEATURE_FALLBACK["escalation"]),
            "icon":                  primary["icon"],
            "runbook_url":           runbook_url,
            "secondary_causes":      secondary_causes,
            "feature_errors":        feature_errors,
            "feature_errors_ranked": feature_ranked,
            "context_summary":       context_summary,
        }

    def __repr__(self) -> str:
        return (
            f"ContextAwareRCA("
            f"features={self.feature_cols}, "
            f"secondary_threshold={self.secondary_threshold})"
        )


# ===========================================================================
# ─── COMPONENT 3: AlertGenerator ────────────────────────────────────────────
# ===========================================================================

class AlertGenerator:
    """
    Produces Prometheus AlertManager v4 webhook-compatible JSON payloads
    with full FiLM context metadata embedded.

    The ``context`` dict fields from ``ContextAwareRCA`` are mapped into
    ``annotations`` so that downstream alert receivers (PagerDuty, Slack,
    OpsGenie) can display the full operational picture — including which
    container type was affected and what the FiLM layer's conditioning
    vector was.

    Parameters
    ----------
    cluster, namespace, job, receiver, external_url : str
        AlertManager routing configuration.
    """

    def __init__(
        self,
        cluster:      str = "production",
        namespace:    str = "default",
        job:          str = "bilstm-film-anomaly-detector",
        receiver:     str = "container-alerts-webhook",
        external_url: str = "http://alertmanager.monitoring.svc:9093",
    ) -> None:
        self.cluster      = cluster
        self.namespace    = namespace
        self.job          = job
        self.receiver     = receiver
        self.external_url = external_url

    def build_payload(
        self,
        context:       ContainerContext,
        severity:      str,
        mse_score:     float,
        rca_result:    Dict,
        timestamp_utc: str,
        event_id:      str,
    ) -> Dict:
        """
        Construct a complete AlertManager v4 webhook payload with FiLM
        context embedded in both ``labels`` and ``annotations``.
        """
        ctx      = context
        pm       = ctx.profile_meta()
        icon     = rca_result.get("icon", "⚠️")
        a_name   = rca_result.get("anomaly_name", "Resource Anomaly")
        cs       = rca_result.get("context_summary", {})

        # ── Compact feature breakdown string ──────────────────────────────
        breakdown = " | ".join(
            f"{e['short_name']}={e['mse']:.4f}({e['error_pct']:.0f}%)"
            for e in rca_result.get("feature_errors_ranked", [])
        )

        # ── Secondary causes label ────────────────────────────────────────
        secondary_str = ", ".join(
            s.get("short_name", s.get("feature", "")) + f"({s['mse']:.4f})"
            for s in rca_result.get("secondary_causes", [])
        ) or "None"

        # ── Concise summary / description lines ───────────────────────────
        summary = (
            f"{icon} [{severity.upper()}] {a_name} | "
            f"{pm.get('profile_icon','📦')} {ctx.display_name()} | "
            f"Container: {ctx.container_id}"
        )
        description = (
            f"The BiLSTM-FiLM autoencoder detected anomalous reconstruction error "
            f"(MSE={mse_score:.6f}) for {ctx.container_type.upper()} container "
            f"'{ctx.container_id}' in namespace '{ctx.namespace}'. "
            f"FiLM meta vector {ctx.film_vector_str()} conditioned the model for "
            f"{ctx.display_name()} workloads ({ctx.tier} tier / {ctx.environment}). "
            f"Root cause: {rca_result.get('primary_display','Unknown')} "
            f"({rca_result.get('primary_error_percent',0):.1f}% of total error). "
            f"Diagnosis: {rca_result.get('diagnosis','')}"
        )

        # ── Alert body ────────────────────────────────────────────────────
        alert_body = {
            "status": "firing" if severity != "Normal" else "resolved",

            # Labels — low-cardinality; used by AM for grouping, routing, silencing
            "labels": {
                "alertname":         "ContainerAnomaly",
                "severity":          severity.lower(),
                "severity_level":    str(SeverityScorer.SEVERITY_INT.get(severity, 0)),
                "container_id":      ctx.container_id,
                "container_type":    ctx.container_type,
                "pod_name":          ctx.pod_name,
                "namespace":         ctx.namespace,
                "tier":              ctx.tier,
                "environment":       ctx.environment,
                "root_cause_metric": rca_result.get("primary_metric", "unknown"),
                "anomaly_name":      a_name,
                "job":               self.job,
                "cluster":           self.cluster,
                "detector_model":    "BiLSTM-FiLM-Autoencoder",
            },

            # Annotations — high-cardinality, human-readable operational details
            "annotations": {
                "summary":                 summary,
                "description":             description,

                # ── FiLM context (thesis-specific) ───────────────────────
                "film_context_type":       ctx.container_type,
                "film_context_display":    ctx.display_name(),
                "film_context_tier":       ctx.tier,
                "film_context_env":        ctx.environment,
                "film_meta_vector":        ctx.film_vector_str(),
                "film_conditioning_note":  cs.get("film_conditioning_note", ""),

                # ── Context-aware RCA ─────────────────────────────────────
                "root_cause_metric":       rca_result.get("primary_metric", ""),
                "root_cause_display":      rca_result.get("primary_display", ""),
                "root_cause_anomaly_name": a_name,
                "diagnosis":               rca_result.get("diagnosis", ""),
                "recommended_action":      rca_result.get("recommended_action", ""),
                "escalation_path":         rca_result.get("escalation_path", "SRE"),
                "secondary_causes":        secondary_str,

                # ── Quantitative metrics ─────────────────────────────────
                "mse_score":               f"{mse_score:.8f}",
                "feature_breakdown":       breakdown,
                "detector_confidence":     self._confidence_label(severity),

                # ── Operational links ─────────────────────────────────────
                "runbook_url":             rca_result.get("runbook_url", ""),
                "event_id":                event_id,
                "profile_description":     pm.get("description", ""),
            },

            "startsAt":     timestamp_utc,
            "endsAt":       "0001-01-01T00:00:00Z",
            "generatorURL": (
                f"http://anomaly-detector.{self.namespace}.svc:8080"
                f"/graph?container={ctx.container_id}&event={event_id}"
            ),
            "fingerprint":  event_id[:16],
        }

        # ── AlertManager envelope ─────────────────────────────────────────
        payload = {
            "version":         "4",
            "groupKey":        f"{{cluster={self.cluster}}}/ContainerAnomaly:{ctx.container_id}",
            "truncatedAlerts": 0,
            "status":          alert_body["status"],
            "receiver":        self.receiver,
            "groupLabels": {
                "alertname":      "ContainerAnomaly",
                "container_type": ctx.container_type,
                "cluster":        self.cluster,
            },
            "commonLabels":      alert_body["labels"],
            "commonAnnotations": {},
            "externalURL":       self.external_url,
            "alerts":            [alert_body],
        }
        return payload

    def build_payload_json(self, *args, **kwargs) -> str:
        return json.dumps(self.build_payload(*args, **kwargs), indent=2)

    @staticmethod
    def _confidence_label(severity: str) -> str:
        return {
            "Critical": "Very High — exceedance of 99.5th percentile threshold",
            "High":     "High — exceedance of 98th percentile threshold",
            "Warning":  "Moderate — exceedance of 95th percentile threshold",
            "Normal":   "Low — within expected operational distribution",
        }.get(severity, "Unknown")

    # ── ─── ─── ─── ─── ─── ─── ─── ─── ─── ─── ─── ─── ─── ─── ─── ───
    # Telegram Chat-Ops Integration
    # ─── ─── ─── ─── ─── ─── ─── ─── ─── ─── ─── ─── ─── ─── ─── ─── ──

    @staticmethod
    def format_telegram_html_alert(
        context:          ContainerContext,
        severity:         str,
        mse_score:        float,
        threshold_p95:    float,
        rca_result:       Dict,
        timestamp_utc:    str,
        event_id:         str,
        llm_explanation:  Optional[str] = None,
    ) -> str:
        """
        Build a richly-formatted Telegram HTML message for one anomaly incident.

        Telegram's ``parse_mode=HTML`` supports: ``<b>``, ``<i>``, ``<code>``,
        ``<pre>``, ``<a href=...>``.  We use these to create a professional
        DevOps alert card that renders beautifully in any Telegram client.

        Parameters
        ----------
        context         : ContainerContext — FiLM metadata & container info
        severity        : str             — "Critical" | "High" | "Warning" | "Normal"
        mse_score       : float           — window-level reconstruction MSE
        threshold_p95   : float           — P95 calibrated threshold
        rca_result      : dict            — output of ContextAwareRCA.analyze()
        timestamp_utc   : str             — ISO-8601 event timestamp
        event_id        : str             — unique incident UUID
        llm_explanation : str, optional   — GenAI RCA narrative to embed

        Returns
        -------
        str — Telegram HTML message (max ~4096 chars; truncated if necessary)
        """
        # ── Severity badge ─────────────────────────────────────────────────
        SEV_EMOJI = {
            "Critical": "🚨",
            "High":     "🔴",
            "Warning":  "⚠️",
            "Normal":   "✅",
        }
        SEV_LABELS = {
            "Critical": "CRITICAL ANOMALY",
            "High":     "HIGH SEVERITY ANOMALY",
            "Warning":  "WARNING — ELEVATED ANOMALY",
            "Normal":   "RESOLVED — Normal",
        }
        sev_emoji = SEV_EMOJI.get(severity, "⚠️")
        sev_label = SEV_LABELS.get(severity, severity.upper())

        # ── Key RCA fields ─────────────────────────────────────────────────
        pm          = context.profile_meta()
        prof_icon   = context.profile_icon()
        disp_name   = context.display_name()
        primary_dis = rca_result.get("primary_display", "Unknown")
        anom_name   = rca_result.get("anomaly_name", "Resource Anomaly")
        diagnosis   = rca_result.get("diagnosis", "")
        rec_action  = rca_result.get("recommended_action", "")
        escalation  = rca_result.get("escalation_path", "SRE Team")
        runbook     = rca_result.get("runbook_url", "https://runbooks.internal")
        icon        = rca_result.get("icon", "⚠️")

        # ── Exceedance ratio ───────────────────────────────────────────────
        exceedance  = mse_score / threshold_p95 if threshold_p95 > 0 else 0.0
        excess_pct  = int((exceedance - 1) * 100) if exceedance > 1 else 0

        # ── Per-feature top-3 breakdown (HTML table rows) ──────────────────
        feat_ranked = rca_result.get("feature_errors_ranked", [])
        total_err   = sum(e.get("mse", 0) for e in feat_ranked) or 1e-12
        feat_lines  = []
        for entry in feat_ranked[:3]:
            e_pct = entry.get("error_pct", entry.get("mse", 0) / total_err * 100)
            bar   = "█" * max(1, int(e_pct / 10)) + "░" * (10 - max(1, int(e_pct / 10)))
            feat_lines.append(
                f"  {entry.get('icon','·')} <b>{entry.get('short_name','?'):7s}</b>"
                f"  {entry.get('mse', 0):.5f}  {e_pct:5.1f}%  {bar}"
            )
        feat_table = "\n".join(feat_lines) or "  (no breakdown available)"

        # ── Recommended action — first 3 steps only (keep message concise) ─
        action_lines = [
            l.strip() for l in rec_action.split("\n") if l.strip()
        ][:3]
        action_str = "\n".join(f"  {l}" for l in action_lines)

        # ── Assemble message ───────────────────────────────────────────────
        # NOTE: HTML entities (<, >, &) must be escaped inside Telegram HTML.
        def _esc(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        parts = [
            # ── Header ───────────────────────────────────────────────────
            f"{sev_emoji} <b>{_esc(sev_label)}</b>",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"{prof_icon} <b>{_esc(disp_name)}</b>",
            f"<i>BiLSTM-FiLM Anomaly Detection Engine</i>",
            "",
            # ── Container info ────────────────────────────────────────────
            f"📦 <b>Container ID:</b>  <code>{_esc(context.container_id)}</code>",
            f"☸️  <b>Pod Name:</b>      <code>{_esc(context.pod_name)}</code>",
            f"🗂️  <b>Namespace:</b>     <code>{_esc(context.namespace)}</code>",
            f"🏷️  <b>Environment:</b>   {_esc(context.tier)} / {_esc(context.environment)}",
            f"🔬 <b>FiLM Vector:</b>   <code>{_esc(context.film_vector_str())}</code>",
            f"⏰ <b>Timestamp:</b>     <code>{_esc(timestamp_utc)}</code>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            # ── Detection results ─────────────────────────────────────────
            "📊 <b>Detection Results</b>",
            f"  MSE Score:  <code>{mse_score:.6f}</code>",
            f"  Threshold:  <code>{threshold_p95:.6f}</code>  <i>(P95 calibrated)</i>",
            f"  Exceedance: <b>{exceedance:.1f}×</b>  ({excess_pct}% above boundary)",
            f"  Severity:   <b>{_esc(severity)}</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            # ── Feature error breakdown ───────────────────────────────────
            "📉 <b>Feature Error Breakdown (Top 3)</b>",
            "<pre>",
            feat_table,
            "</pre>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            # ── RCA ───────────────────────────────────────────────────────
            f"🎯 <b>Root Cause Analysis</b>",
            "",
            f"  {icon} <b>{_esc(anom_name)}</b>",
            f"  <i>{_esc(diagnosis[:260])}</i>",
            "",
            f"  🔧 <b>Recommended Action:</b>",
            f"<pre>{_esc(action_str)}</pre>",
            f"  📣 <b>Escalate to:</b>  {_esc(escalation)}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]

        # ── Optional GenAI explanation section ────────────────────────────
        if llm_explanation and llm_explanation.strip():
            parts += [
                "",
                "🤖 <b>AI Analysis (GenAI RCA):</b>",
                f"<pre>{_esc(llm_explanation.strip()[:600])}</pre>",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ]

        # ── Footer ────────────────────────────────────────────────────────
        parts += [
            "",
            f'🔗 <a href="{runbook}">Open Runbook</a>',
            f"🆔 Event: <code>{event_id[:20]}</code>",
        ]

        msg = "\n".join(parts)

        # Telegram hard limit is 4096 chars — truncate gracefully
        if len(msg) > 4000:
            msg = msg[:3970] + "\n\n<i>… (message truncated)</i>"

        return msg

    def send_telegram_alert(
        self,
        bot_token:        str,
        chat_id:          str,
        context:          ContainerContext,
        severity:         str,
        mse_score:        float,
        threshold_p95:    float,
        rca_result:       Dict,
        timestamp_utc:    str,
        event_id:         str,
        llm_explanation:  Optional[str] = None,
        dry_run:          bool = False,
    ) -> Dict:
        """
        Format and dispatch a Telegram alert for one anomaly incident.

        Uses Telegram's Bot API ``sendMessage`` endpoint with ``parse_mode=HTML``
        so the message renders with bold headings, code blocks, and inline links
        in any Telegram client (desktop, mobile, web).

        The method is designed to be **non-blocking** for the alerting pipeline:
        all network errors are caught and returned as a structured result dict,
        so a Telegram outage never raises an exception in the caller.

        Parameters
        ----------
        bot_token       : str  — Telegram Bot API token (from @BotFather)
        chat_id         : str  — Target channel/group/user ID (e.g. "-100XXXXXXXXX")
        context         : ContainerContext  — FiLM metadata & container info
        severity        : str  — Severity tier string
        mse_score       : float — Reconstruction MSE
        threshold_p95   : float — P95 anomaly threshold
        rca_result      : dict  — ContextAwareRCA.analyze() output
        timestamp_utc   : str  — ISO-8601 timestamp
        event_id        : str  — Unique incident identifier
        llm_explanation : str, optional — GenAI RCA narrative to embed
        dry_run         : bool — If True, build the message but do NOT send it

        Returns
        -------
        dict with keys:
          success      : bool   — True if message was delivered
          message_id   : int    — Telegram message ID (if delivered)
          message_text : str    — The formatted HTML string that was/would be sent
          status_code  : int    — HTTP response code (0 if dry_run or network error)
          error        : str    — Error description (empty on success)
        """
        # ── Build the HTML message ─────────────────────────────────────────
        html_msg = self.format_telegram_html_alert(
            context=context, severity=severity, mse_score=mse_score,
            threshold_p95=threshold_p95, rca_result=rca_result,
            timestamp_utc=timestamp_utc, event_id=event_id,
            llm_explanation=llm_explanation,
        )

        result: Dict = {
            "success":      False,
            "message_id":   None,
            "message_text": html_msg,
            "status_code":  0,
            "error":        "",
        }

        # ── Dry-run short-circuit ──────────────────────────────────────────
        if dry_run:
            logger.info(
                "Telegram DRY-RUN — container=%s severity=%s chars=%d",
                context.container_id, severity, len(html_msg),
            )
            result["success"] = True
            result["error"]   = "dry_run — message not sent"
            return result

        # ── Validate credentials before making the network call ────────────
        if not bot_token or "your" in bot_token.lower() or len(bot_token) < 10:
            logger.warning(
                "Telegram: placeholder bot_token detected — skipping send."
            )
            result["error"] = "placeholder bot_token — message not sent"
            return result

        if not chat_id or "your" in str(chat_id).lower():
            logger.warning("Telegram: placeholder chat_id detected — skipping send.")
            result["error"] = "placeholder chat_id — message not sent"
            return result

        # ── Attempt HTTP POST to Telegram Bot API ──────────────────────────
        try:
            import requests as _requests_lib

            api_url  = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload  = {
                "chat_id":                  str(chat_id),
                "text":                     html_msg,
                "parse_mode":               "HTML",
                "disable_web_page_preview": True,
                "disable_notification":     (severity in ("Warning", "Normal")),
            }

            resp = _requests_lib.post(
                api_url, json=payload,
                timeout=10,        # max 10 s — never block the pipeline
            )

            result["status_code"] = resp.status_code

            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    result["success"]    = True
                    result["message_id"] = data.get("result", {}).get("message_id")
                    logger.info(
                        "Telegram alert sent — container=%s severity=%s msg_id=%s",
                        context.container_id, severity, result["message_id"],
                    )
                else:
                    result["error"] = data.get("description", "Telegram ok=False")
                    logger.warning("Telegram API error: %s", result["error"])
            else:
                result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.warning("Telegram HTTP error: %s", result["error"])

        except ImportError:
            result["error"] = (
                "'requests' library not installed. "
                "Run: pip install requests"
            )
            logger.error("Telegram send failed: %s", result["error"])

        except Exception as exc:          # ConnectionError, Timeout, etc.
            result["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Telegram send failed (non-fatal) — %s", result["error"]
            )

        return result

    def __repr__(self) -> str:
        return (
            f"AlertGenerator(cluster='{self.cluster}', "
            f"namespace='{self.namespace}')"
        )


# ===========================================================================
# ─── ORCHESTRATOR: IncidentPipeline ─────────────────────────────────────────
# ===========================================================================

class IncidentPipeline:
    """
    Orchestrates SeverityScorer → ContextAwareRCA → AlertGenerator.

    This is the single entry point for real-time or batch processing.
    The "Common Engine" is embodied here: one pipeline object handles
    containers of type "api", "database", "cache", "worker", "ml_inference"
    — differentiated only by the ``ContainerContext`` passed at call time.

    Parameters
    ----------
    scorer  : SeverityScorer  (must be calibrated)
    rca     : ContextAwareRCA
    alerter : AlertGenerator
    """

    def __init__(
        self,
        scorer:  SeverityScorer,
        rca:     ContextAwareRCA,
        alerter: AlertGenerator,
    ) -> None:
        self.scorer  = scorer
        self.rca     = rca
        self.alerter = alerter

    def process(
        self,
        original:      np.ndarray,
        reconstructed: np.ndarray,
        mse_score:     float,
        context:       ContainerContext,
        timestamp_utc: Optional[str] = None,
    ) -> IncidentEvent:
        """
        Run the full incident response pipeline for one anomaly window.

        Parameters
        ----------
        original      : np.ndarray (W, F) — original scaled input window
        reconstructed : np.ndarray (W, F) — model reconstruction
        mse_score     : float             — pre-computed window MSE
        context       : ContainerContext  — FiLM metadata for this container
        timestamp_utc : str, optional     — ISO-8601; defaults to utcnow()

        Returns
        -------
        IncidentEvent — fully populated, immutable incident record
        """
        if timestamp_utc is None:
            timestamp_utc = (
                datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                + "Z"
            )
        event_id = str(uuid.uuid4())

        # Step 1 — Severity classification
        severity = self.scorer.score(mse_score)

        # Step 2 — Context-aware RCA
        rca_result = self.rca.analyze(original, reconstructed, context)

        # Step 3 — Alert payload generation
        payload_dict = self.alerter.build_payload(
            context=context,
            severity=severity,
            mse_score=mse_score,
            rca_result=rca_result,
            timestamp_utc=timestamp_utc,
            event_id=event_id,
        )

        # Step 4 — Assemble immutable IncidentEvent
        event = IncidentEvent(
            event_id=event_id,
            timestamp_utc=timestamp_utc,
            context=context,
            severity=severity,
            mse_score=float(mse_score),
            primary_metric=rca_result["primary_metric"],
            primary_display=rca_result["primary_display"],
            anomaly_name=rca_result["anomaly_name"],
            diagnosis=rca_result["diagnosis"],
            recommended_action=rca_result["recommended_action"],
            escalation_path=rca_result["escalation_path"],
            icon=rca_result["icon"],
            runbook_url=rca_result["runbook_url"],
            feature_errors=rca_result["feature_errors"],
            alert_payload=json.dumps(payload_dict, indent=2),
        )

        logger.info(
            "IncidentEvent | %s | [%s] %s container=%s mse=%.6f → %s",
            event_id[:8], severity, context.container_type,
            context.container_id, mse_score, rca_result["primary_metric"],
        )
        return event

    def process_batch(
        self,
        originals:      np.ndarray,
        reconstructions: np.ndarray,
        mse_scores:     np.ndarray,
        contexts:       List[ContainerContext],
        timestamps_utc: Optional[List[str]] = None,
    ) -> List[IncidentEvent]:
        """
        Process a batch of anomaly windows.

        Parameters
        ----------
        originals       : (N, W, F)
        reconstructions : (N, W, F)
        mse_scores      : (N,)
        contexts        : list[ContainerContext] length N
        timestamps_utc  : list[str] length N (optional)

        Returns
        -------
        list[IncidentEvent]
        """
        n = len(mse_scores)
        if timestamps_utc is None:
            now = datetime.datetime.utcnow()
            timestamps_utc = [
                (now + datetime.timedelta(seconds=i)).strftime(
                    "%Y-%m-%dT%H:%M:%S.%f"
                )[:-3] + "Z"
                for i in range(n)
            ]

        events = [
            self.process(
                original=originals[i],
                reconstructed=reconstructions[i],
                mse_score=float(mse_scores[i]),
                context=contexts[i],
                timestamp_utc=timestamps_utc[i],
            )
            for i in range(n)
        ]

        # Summary log
        sev_counts = {s: sum(1 for e in events if e.severity == s)
                      for s in ("Critical", "High", "Warning", "Normal")}
        logger.info(
            "Batch complete: %d events | Critical=%d High=%d Warning=%d Normal=%d",
            len(events), sev_counts["Critical"], sev_counts["High"],
            sev_counts["Warning"], sev_counts["Normal"],
        )
        return events

    def __repr__(self) -> str:
        return (
            f"IncidentPipeline(\n"
            f"  scorer  = {self.scorer!r}\n"
            f"  rca     = {self.rca!r}\n"
            f"  alerter = {self.alerter!r}\n"
            f")"
        )


# ===========================================================================
# ─── GENAI RCA TIER ──────────────────────────────────────────────────────────
#
# This tier adds a Large Language Model (LLM) as a second-opinion analyst on
# top of the statistical BiLSTM-FiLM anomaly detection pipeline.
#
# Architecture Position
# ---------------------
#
#   BiLSTM-FiLM model
#   → IncidentPipeline (severity + statistical RCA)
#   → GenAIRCAEngine   (LLM verification + human explanation)  ← NEW
#   → format_telegram_alert / generate_prometheus_alert_with_ai ← NEW
#
# LLM Providers (in order of preference)
# ----------------------------------------
#   1. Google Gemini (google-generativeai) — primary
#   2. OpenAI GPT-4  (openai)              — fallback
#   3. DEMO_MODE                           — offline / thesis demo
#
# The engine is designed to NEVER crash the alerting pipeline:
# all API errors are caught, logged, and handled with demo mode.
# ===========================================================================


# ===========================================================================
# ─── DATA CLASS: LLMAnalysis ─────────────────────────────────────────────────
# ===========================================================================

@dataclass(frozen=True)
class LLMAnalysis:
    """
    Immutable record of one GenAI-powered RCA call.

    Attributes
    ----------
    verdict        : "GENUINE ANOMALY" | "FALSE POSITIVE" | "UNCERTAIN"
    root_cause     : Two-sentence natural language root cause explanation.
    mitigation     : One-sentence immediate action for the on-call engineer.
    confidence_pct : LLM self-reported confidence (0–100).
    model_used     : LLM model identifier (e.g. "gemini-1.5-flash", "gpt-4o-mini").
    raw_response   : Complete LLM output string (for audit trail).
    prompt_tokens  : Estimated tokens in the prompt (informational).
    latency_ms     : API call wall-clock latency in milliseconds.
    success        : False if the API call failed and demo mode was used.
    error_message  : Exception message if ``success=False``, else empty string.
    """
    verdict:        str
    root_cause:     str
    mitigation:     str
    confidence_pct: int
    model_used:     str
    raw_response:   str
    prompt_tokens:  int
    latency_ms:     float
    success:        bool
    error_message:  str = ""

    def is_genuine(self) -> bool:
        return "GENUINE" in self.verdict.upper()

    def confidence_label(self) -> str:
        if self.confidence_pct >= 90: return "Very High"
        if self.confidence_pct >= 75: return "High"
        if self.confidence_pct >= 55: return "Moderate"
        return "Low"

    def to_dict(self) -> Dict:
        return {
            "verdict":        self.verdict,
            "root_cause":     self.root_cause,
            "mitigation":     self.mitigation,
            "confidence_pct": self.confidence_pct,
            "model_used":     self.model_used,
            "latency_ms":     round(self.latency_ms, 1),
            "success":        self.success,
            "error_message":  self.error_message,
        }

    def __str__(self) -> str:
        flag = "[AI]" if self.success else "[AI-DEMO]"
        return (
            f"{flag} {self.verdict}  (confidence={self.confidence_pct}%)  "
            f"via {self.model_used}  ({self.latency_ms:.0f} ms)\n"
            f"ROOT CAUSE: {self.root_cause}\n"
            f"MITIGATION: {self.mitigation}"
        )


# ===========================================================================
# ─── COMPONENT 4: GenAIRCAEngine ─────────────────────────────────────────────
# ===========================================================================

class GenAIRCAEngine:
    """
    GenAI-powered Root Cause Analysis engine using Google Gemini or OpenAI GPT.

    Purpose
    -------
    When the BiLSTM-FiLM autoencoder flags an anomaly, this engine sends the
    full incident context (metrics, reconstruction, FiLM metadata, logs) to an
    LLM and requests:
      1. A verdict — GENUINE ANOMALY or FALSE POSITIVE.
      2. A two-sentence root cause explanation.
      3. A one-sentence mitigation step.
      4. A confidence percentage.

    The structured prompt is designed by an Expert SRE persona so the LLM
    focuses on operational accuracy rather than generic advice.

    Provider Cascade
    ----------------
    1. Google Gemini (primary)       — ``gemini_api_key`` required
    2. OpenAI GPT    (fallback)      — ``openai_api_key`` required
    3. Demo Mode     (offline)       — always available; synthesises a
                                       realistic-looking response from the
                                       statistical RCA result.

    Parameters
    ----------
    gemini_api_key  : str, optional   — Google AI Studio API key
    openai_api_key  : str, optional   — OpenAI API key
    gemini_model    : str             — Gemini model ID
    openai_model    : str             — OpenAI model ID
    timeout_seconds : float           — HTTP request timeout
    max_retries     : int             — retries on transient errors
    temperature     : float           — LLM sampling temperature (lower = more deterministic)
    """

    #: Prompt template sent to the LLM.  Placeholders filled by _build_prompt().
    _PROMPT_TEMPLATE = """\
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

══════════════════ ORIGINAL METRICS (last 5 timesteps, Min-Max scaled [0,1]) ══════════════════
{metrics_table}

══════════════════ RECENT CONTAINER LOGS (tail -20) ══════════════════
```
{recent_logs}
```

══════════════════ INSTRUCTIONS ══════════════════
Based on ALL of the above evidence — the statistical anomaly score, the feature-level \
breakdown, the FiLM container context, the raw metric trends, and the log entries — \
provide your analysis in EXACTLY the following format:

VERDICT: <GENUINE ANOMALY|FALSE POSITIVE|UNCERTAIN>
ROOT CAUSE: <Exactly two sentences explaining what happened and why, referencing specific metrics and log entries where applicable.>
MITIGATION: <Exactly one concrete sentence describing what the on-call engineer should do RIGHT NOW.>
CONFIDENCE: <Integer percentage 0-100 reflecting your confidence in this analysis>

Do NOT add any other text, preamble, or explanation outside this format.\
"""

    def __init__(
        self,
        gemini_api_key:  Optional[str] = None,
        openai_api_key:  Optional[str] = None,
        gemini_model:    str           = "gemini-1.5-flash",
        openai_model:    str           = "gpt-4o-mini",
        timeout_seconds: float         = 30.0,
        max_retries:     int           = 2,
        temperature:     float         = 0.10,
    ) -> None:
        self.gemini_api_key  = gemini_api_key  or ""
        self.openai_api_key  = openai_api_key  or ""
        self.gemini_model    = gemini_model
        self.openai_model    = openai_model
        self.timeout_seconds = timeout_seconds
        self.max_retries     = max_retries
        self.temperature     = temperature

    # ── Public API ────────────────────────────────────────────────────────

    def analyze_with_llm(
        self,
        original_metrics:      np.ndarray,   # (W, F)
        reconstructed_metrics: np.ndarray,   # (W, F)
        mse:                   float,
        context:               ContainerContext,
        rca_result:            Dict,
        recent_logs:           str,
        threshold_p95:         float,
        timestamp_utc:         Optional[str] = None,
    ) -> LLMAnalysis:
        """
        Send the full anomaly context to an LLM and return a structured
        ``LLMAnalysis``.

        Parameters
        ----------
        original_metrics      : (W, F) — original scaled window
        reconstructed_metrics : (W, F) — model reconstruction
        mse                   : float  — window-level MSE
        context               : ContainerContext  — FiLM metadata
        rca_result            : dict   — output of ContextAwareRCA.analyze()
        recent_logs           : str    — simulated or real container logs
        threshold_p95         : float  — P95 anomaly threshold
        timestamp_utc         : str    — ISO-8601 timestamp

        Returns
        -------
        LLMAnalysis — always returns (never raises); uses demo mode on failure.
        """
        if timestamp_utc is None:
            timestamp_utc = (
                datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S") + "Z"
            )

        prompt = self._build_prompt(
            original_metrics, reconstructed_metrics, mse,
            context, rca_result, recent_logs, threshold_p95, timestamp_utc,
        )

        # ── Try Gemini first ───────────────────────────────────────────────
        if self.gemini_api_key and not self._is_placeholder(self.gemini_api_key):
            try:
                raw, latency = self._call_gemini(prompt)
                return self._parse_response(
                    raw, self.gemini_model, latency, True, len(prompt) // 4
                )
            except Exception as exc:
                logger.warning("Gemini call failed (%s). Trying OpenAI…", exc)

        # ── Try OpenAI fallback ────────────────────────────────────────────
        if self.openai_api_key and not self._is_placeholder(self.openai_api_key):
            try:
                raw, latency = self._call_openai(prompt)
                return self._parse_response(
                    raw, self.openai_model, latency, True, len(prompt) // 4
                )
            except Exception as exc:
                logger.warning("OpenAI call failed (%s). Falling back to demo.", exc)

        # ── Demo mode (offline / placeholder keys) ─────────────────────────
        logger.info("GenAIRCAEngine: using DEMO_MODE (no live API available).")
        raw     = self._demo_response(mse, context, rca_result, threshold_p95)
        return self._parse_response(
            raw, "DEMO_MODE (Gemini-1.5-Flash Simulated)", 0.0, False,
            len(prompt) // 4,
        )

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _is_placeholder(key: str) -> bool:
        lower = key.lower()
        return (
            not key
            or "your-api-key" in lower
            or "placeholder" in lower
            or "xxxx" in lower
            or len(key) < 12
        )

    def _build_prompt(
        self,
        original:      np.ndarray,
        reconstructed: np.ndarray,
        mse:           float,
        context:       ContainerContext,
        rca_result:    Dict,
        recent_logs:   str,
        threshold_p95: float,
        timestamp_utc: str,
    ) -> str:
        """Construct the structured SRE expert prompt."""
        original      = np.asarray(original)
        reconstructed = np.asarray(reconstructed)

        # ── Per-feature error table ────────────────────────────────────────
        feat_table_lines = []
        for entry in rca_result.get("feature_errors_ranked", []):
            marker = "  <<< PRIMARY" if entry["feature"] == rca_result.get("primary_metric") else ""
            feat_table_lines.append(
                f"  {entry['icon']} {entry['display_name']:30s}  "
                f"MSE={entry['mse']:.6f}  ({entry['error_pct']:.1f}%){marker}"
            )
        feature_table = "\n".join(feat_table_lines) or "  (no features ranked)"

        # ── Last-5-timestep metrics table ──────────────────────────────────
        feat_names = list(rca_result.get("feature_errors", {}).keys())
        header = "  Timestep | " + " | ".join(f"{n[:8]:>8}" for n in feat_names)
        metrics_lines = [header, "  " + "-" * (len(header) - 2)]
        tail = original[-5:]
        for i, row in enumerate(tail, start=len(original) - 5):
            vals = " | ".join(f"{v:8.3f}" for v in row[: len(feat_names)])
            metrics_lines.append(f"  {i:>8d} | {vals}")
        metrics_table = "\n".join(metrics_lines)

        # ── FiLM context note ──────────────────────────────────────────────
        film_note = (
            f"The FiLM layer conditioned the BiLSTM reconstruction on "
            f"'{context.container_type}' workload context, adjusting the "
            f"expected reconstruction baseline for {context.display_name()}."
        )

        exceedance = mse / threshold_p95 if threshold_p95 > 0 else 0.0

        return self._PROMPT_TEMPLATE.format(
            timestamp_utc=timestamp_utc,
            severity=rca_result.get("severity", "Unknown") if "severity" in rca_result else "Critical",
            mse=mse,
            threshold_p95=threshold_p95,
            exceedance=exceedance,
            container_id=context.container_id,
            container_type=context.container_type,
            container_display=context.display_name(),
            tier=context.tier,
            environment=context.environment,
            pod_name=context.pod_name,
            namespace=context.namespace,
            film_vector=context.film_vector_str(),
            film_note=film_note,
            feature_table=feature_table,
            primary_metric=rca_result.get("primary_display", "Unknown"),
            primary_pct=rca_result.get("primary_error_percent", 0.0),
            stat_diagnosis=rca_result.get("diagnosis", ""),
            stat_action=rca_result.get("recommended_action", "").split("\n")[0],
            metrics_table=metrics_table,
            recent_logs=recent_logs.strip() or "(no logs available)",
        )

    def _call_gemini(self, prompt: str) -> Tuple[str, float]:
        """Call the Google Gemini API. Returns (response_text, latency_ms)."""
        import time
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            )
        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel(self.gemini_model)
        t0 = time.perf_counter()
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature":      self.temperature,
                "max_output_tokens": 512,
                "top_p":            0.95,
            },
            request_options={"timeout": self.timeout_seconds},
        )
        latency = (time.perf_counter() - t0) * 1000.0
        return response.text.strip(), latency

    def _call_openai(self, prompt: str) -> Tuple[str, float]:
        """Call the OpenAI Chat API. Returns (response_text, latency_ms)."""
        import time
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "openai not installed. Run: pip install openai"
            )
        client = OpenAI(api_key=self.openai_api_key, timeout=self.timeout_seconds)
        t0 = time.perf_counter()
        completion = client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Expert SRE and DevOps Architect. "
                        "Respond ONLY in the exact structured format requested."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=512,
        )
        latency = (time.perf_counter() - t0) * 1000.0
        return completion.choices[0].message.content.strip(), latency

    def _demo_response(
        self,
        mse:           float,
        context:       ContainerContext,
        rca_result:    Dict,
        threshold_p95: float,
    ) -> str:
        """
        Generate a realistic demo LLM response from the statistical RCA result.
        Used when no live API key is available (offline / thesis demonstration).
        """
        ctype   = context.container_type
        primary = rca_result.get("primary_metric", "")
        diag    = rca_result.get("diagnosis", "")
        action  = rca_result.get("recommended_action", "").split("\n")[0]
        pct     = rca_result.get("primary_error_percent", 50.0)
        excess  = mse / threshold_p95 if threshold_p95 > 0 else 1.0

        # ── Verdict ────────────────────────────────────────────────────────
        verdict = "GENUINE ANOMALY" if excess >= 3.0 else "GENUINE ANOMALY"

        # ── Context-specific root cause narratives ─────────────────────────
        NARRATIVES: Dict[str, Dict[str, str]] = {
            "database": {
                "mem_util_percent": (
                    "The PostgreSQL container is experiencing severe buffer pool exhaustion, "
                    f"with heap memory climbing monotonically to {min(99, int(50 + pct*0.45))}% "
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

        # Look up narrative (primary metric in container type)
        type_narratives = NARRATIVES.get(ctype, {})
        if primary in type_narratives:
            cause_text, mit_text = type_narratives[primary]
        else:
            # Generic fallback using the statistical RCA
            cause_text = (
                f"The {context.display_name()} container shows anomalous reconstruction error "
                f"primarily driven by {rca_result.get('primary_display', primary)} "
                f"({pct:.0f}% of total MSE={mse:.6f}), which is {excess:.1f}× above the P95 threshold. "
                f"{diag}"
            )
            mit_text = action or (
                "Execute kubectl describe pod and kubectl logs to gather additional context "
                "before escalating to the responsible engineering team."
            )

        confidence = min(99, max(72, int(70 + min(excess, 10) * 2.5 - (0 if excess > 5 else 5))))

        return (
            f"VERDICT: {verdict}\n"
            f"ROOT CAUSE: {cause_text}\n"
            f"MITIGATION: {mit_text}\n"
            f"CONFIDENCE: {confidence}%"
        )

    @staticmethod
    def _parse_response(
        raw_text:     str,
        model_used:   str,
        latency_ms:   float,
        success:      bool,
        prompt_tokens: int,
    ) -> LLMAnalysis:
        """
        Parse the structured LLM output into an ``LLMAnalysis`` dataclass.
        Gracefully handles malformed responses.
        """
        import re
        verdict    = "UNCERTAIN"
        root_cause = raw_text   # fallback: entire response as root_cause
        mitigation = "Investigate with kubectl describe pod and review application logs."
        confidence = 70

        try:
            # VERDICT
            m = re.search(r"VERDICT\s*:\s*(.+?)(?:\n|ROOT)", raw_text, re.IGNORECASE)
            if m:
                verdict = m.group(1).strip().rstrip(".")

            # ROOT CAUSE
            m = re.search(r"ROOT CAUSE\s*:\s*(.+?)(?:\nMITIGATION|\nCONFIDENCE|$)",
                          raw_text, re.IGNORECASE | re.DOTALL)
            if m:
                root_cause = m.group(1).strip()

            # MITIGATION
            m = re.search(r"MITIGATION\s*:\s*(.+?)(?:\nCONFIDENCE|$)",
                          raw_text, re.IGNORECASE | re.DOTALL)
            if m:
                mitigation = m.group(1).strip()

            # CONFIDENCE
            m = re.search(r"CONFIDENCE\s*:\s*(\d+)", raw_text, re.IGNORECASE)
            if m:
                confidence = min(100, max(0, int(m.group(1))))

        except Exception as parse_err:
            logger.warning("LLM response parsing error: %s", parse_err)

        return LLMAnalysis(
            verdict=verdict,
            root_cause=root_cause,
            mitigation=mitigation,
            confidence_pct=confidence,
            model_used=model_used,
            raw_response=raw_text,
            prompt_tokens=prompt_tokens,
            latency_ms=latency_ms,
            success=success,
        )

    def __repr__(self) -> str:
        gemini_ok = bool(self.gemini_api_key and not self._is_placeholder(self.gemini_api_key))
        openai_ok = bool(self.openai_api_key and not self._is_placeholder(self.openai_api_key))
        return (
            f"GenAIRCAEngine("
            f"gemini={self.gemini_model}({'OK' if gemini_ok else 'no key'}), "
            f"openai={self.openai_model}({'OK' if openai_ok else 'no key'}), "
            f"temp={self.temperature})"
        )


# ===========================================================================
# ─── FORMATTING: Telegram Alert with AI Analysis ─────────────────────────────
# ===========================================================================

def format_telegram_alert(
    event:        IncidentEvent,
    llm_analysis: Optional[LLMAnalysis],
    threshold_p95: float,
    include_raw_payload: bool = False,
) -> str:
    """
    Format a rich plain-text Telegram alert message combining the statistical
    BiLSTM-FiLM incident data with the GenAI RCA explanation.

    Telegram supports MarkdownV2 but plain ASCII art is more portable.
    The output is also suitable for Slack/Teams webhook ``text`` fields.

    Parameters
    ----------
    event           : IncidentEvent   — output of IncidentPipeline.process()
    llm_analysis    : LLMAnalysis     — output of GenAIRCAEngine.analyze_with_llm()
    threshold_p95   : float           — P95 threshold for display
    include_raw_payload : bool        — if True, append the Prometheus JSON snippet

    Returns
    -------
    str — formatted alert message
    """
    ctx = event.context
    pm  = ctx.profile_meta()

    SEV_BADGE = {
        "Critical": "🔴 CRITICAL",
        "High":     "🟠 HIGH",
        "Warning":  "🟡 WARNING",
        "Normal":   "🟢 NORMAL",
    }
    badge    = SEV_BADGE.get(event.severity, event.severity)
    icon     = ctx.profile_icon()
    divider  = "─" * 48
    hdivider = "═" * 48

    lines: List[str] = []

    # ── Header ────────────────────────────────────────────────────────────
    lines += [
        hdivider,
        f"  AI-ASSISTED ANOMALY ALERT",
        f"  BiLSTM-FiLM + GenAI RCA Engine",
        hdivider,
        f"  {badge}  {icon} {ctx.display_name()}",
        divider,
    ]

    # ── Container info ────────────────────────────────────────────────────
    lines += [
        f"  Container  : {ctx.container_id}",
        f"  Pod        : {ctx.pod_name}",
        f"  Namespace  : {ctx.namespace}",
        f"  Tier / Env : {ctx.tier} / {ctx.environment}",
        f"  Timestamp  : {event.timestamp_utc}",
        divider,
    ]

    # ── Detection results ─────────────────────────────────────────────────
    exceedance = event.mse_score / threshold_p95 if threshold_p95 > 0 else 0.0
    lines += [
        "  DETECTION RESULTS",
        f"    MSE Score   : {event.mse_score:.6f}",
        f"    Threshold   : {threshold_p95:.6f}  (P95 calibrated)",
        f"    Exceedance  : {exceedance:.1f}x  ({(exceedance-1)*100:.0f}% above boundary)",
        f"    Severity    : {event.severity}",
        divider,
    ]

    # ── Statistical RCA ───────────────────────────────────────────────────
    lines += [
        "  BiLSTM-FiLM ROOT CAUSE ANALYSIS",
        f"    {event.icon}  {event.anomaly_name}",
        f"    Primary  : {event.primary_display}  ({event.feature_errors.get(event.primary_metric, 0):.6f} MSE)",
        f"    FiLM vec : {ctx.film_vector_str()}",
    ]
    # Per-feature mini-table (top 3)
    top3 = sorted(event.feature_errors.items(), key=lambda x: x[1], reverse=True)[:3]
    for feat, err in top3:
        fm = FEATURE_META.get(feat, _UNKNOWN_META)
        pct = err / (sum(event.feature_errors.values()) or 1) * 100
        lines.append(f"    {fm['icon']}  {fm['short_name']:8s}: {err:.5f}  ({pct:.0f}%)")
    lines.append(divider)

    # ── AI Analysis section ───────────────────────────────────────────────
    if llm_analysis is not None:
        ai_badge = "  AI ANALYSIS" if llm_analysis.success else "  AI ANALYSIS (DEMO)"
        lines += [
            ai_badge,
            f"    Model      : {llm_analysis.model_used}",
            f"    Confidence : {llm_analysis.confidence_pct}%  ({llm_analysis.confidence_label()})",
            f"    Latency    : {llm_analysis.latency_ms:.0f} ms",
            "",
            f"    VERDICT: {llm_analysis.verdict}",
            "",
            "    ROOT CAUSE:",
        ]
        # Word-wrap root cause at ~52 chars
        words     = llm_analysis.root_cause.split()
        cur_line  = "      "
        for w in words:
            if len(cur_line) + len(w) + 1 > 56:
                lines.append(cur_line)
                cur_line = "      " + w + " "
            else:
                cur_line += w + " "
        if cur_line.strip():
            lines.append(cur_line)

        lines += [
            "",
            "    MITIGATION:",
        ]
        words    = llm_analysis.mitigation.split()
        cur_line = "      "
        for w in words:
            if len(cur_line) + len(w) + 1 > 56:
                lines.append(cur_line)
                cur_line = "      " + w + " "
            else:
                cur_line += w + " "
        if cur_line.strip():
            lines.append(cur_line)

        lines.append(divider)

    # ── Footer ────────────────────────────────────────────────────────────
    lines += [
        "  LINKS & ACTIONS",
        f"    Runbook  : {event.runbook_url}",
        f"    Escalate : {event.escalation_path}",
        f"    Event ID : {event.event_id[:16]}",
        hdivider,
    ]

    return "\n".join(lines)


def generate_prometheus_alert_with_ai(
    event:        IncidentEvent,
    llm_analysis: Optional[LLMAnalysis],
) -> Dict:
    """
    Extend the existing Prometheus AlertManager payload with AI analysis
    annotations.  Returns the full payload dict ready for HTTP POST.

    This function augments the existing ``event.alert_payload`` JSON with
    three new annotations injected into the first alert's ``annotations``:
      - ``genai_verdict``
      - ``genai_root_cause``
      - ``genai_mitigation``
      - ``genai_confidence``
      - ``genai_model``
    """
    payload = event.to_alert_dict()

    if llm_analysis is not None and payload.get("alerts"):
        ai_annotations = {
            "genai_verdict":          llm_analysis.verdict,
            "genai_root_cause":       llm_analysis.root_cause,
            "genai_mitigation":       llm_analysis.mitigation,
            "genai_confidence":       f"{llm_analysis.confidence_pct}%",
            "genai_model":            llm_analysis.model_used,
            "genai_latency_ms":       f"{llm_analysis.latency_ms:.0f}",
            "genai_success":          str(llm_analysis.success),
        }
        payload["alerts"][0]["annotations"].update(ai_annotations)
        # Also update the common annotations
        payload["commonAnnotations"].update(ai_annotations)

    return payload
