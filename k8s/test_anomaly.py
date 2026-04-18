# test_anomaly.py
import requests
import time

# Simulate sudden high CPU + Memory spike for one container
payload = {
    "query": 'rate(container_cpu_usage_seconds_total{name="c15b86c3b463c4fc54fb3f504e4e879d91921687c0782f3525c3be4be0e19482"}[5m]) * 100'
}
# You can run this multiple times to create a spike
for i in range(8):
    try:
        requests.get("http://35.206.92.147:9090/api/v1/query", params=payload)
        print(f"Spike simulation {i+1}/8 sent")
    except:
        pass
    time.sleep(2)
print("Anomaly simulation completed - check your live_inference.py log and Telegram!")