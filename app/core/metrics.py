from prometheus_client import Counter, Histogram

anomaly_score_hist = Histogram(
    "audio_anomaly_score",
    "Anomaly score distribution",
    buckets=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
)

anomaly_detected_total = Counter(
    "audio_anomaly_detected_total",
    "Total anomalies detected"
)

files_processed_total = Counter(
    "audio_files_processed_total",
    "Total files processed",
    ["status"]
)
