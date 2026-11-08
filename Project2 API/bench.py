
import time
import httpx
import numpy as np

URL = "http://127.0.0.1:8000/predict"
N = 50

latencies = []

with open("sample.jpg", "rb") as f:
    data = f.read()

with httpx.Client(timeout=30) as client:

    for _ in range(N):

        t0 = time.perf_counter()

        response = client.post(
            URL,
            files={
                "file": (
                    "sample.jpg",
                    data,
                    "image/jpeg"
                )
            }
        )

        response.raise_for_status()

        latencies.append(
            (time.perf_counter() - t0) * 1000
        )

latencies = np.array(latencies)

print(
    f"n={N} "
    f"p50={np.percentile(latencies, 50):.1f} ms "
    f"p95={np.percentile(latencies, 95):.1f} ms "
    f"max={latencies.max():.1f} ms"
)

print(
    f"Throughput with 1 worker ~ "
    f"{1000 / latencies.mean():.1f} requests/s"
)
