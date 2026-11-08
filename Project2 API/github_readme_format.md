# Cotton Disease Vision + Text API

An API built with FastAPI and PyTorch (ResNet18) for cotton plant disease classification and field report analysis.

---

## 🚀 API Endpoints Summary

### 1. `GET /health`
* **Description:** Health check endpoint to verify server status and model loading.
* **Response:** Returns API operational status (`"status": "ok"`), confirmation that the model is loaded into memory (`"model_loaded": true`), and lists all target classification classes.

### 2. `POST /predict`
* **Description:** Image inference endpoint for single-image classification.
* **Input:** Multipart form-data with an image file (`file`). Requires `x-api-key` header.
* **Response:** Returns predicted label, confidence probabilities across classes, and execution latency in milliseconds (`latency_ms`).

### 3. `POST /analyze`
* **Description:** Multimodal endpoint combining leaf image diagnosis with field text notes.
* **Input:** Multipart form-data accepting an image file (`file`) and inspection notes (`note`). Enforces strict text length validation ($\le 2,000$ characters).
* **Response:** Returns disease assessment alongside field notes, or a `422 Unprocessable Entity` error if input validation fails.

---

## 📊 Benchmarks & System Tests

### 1. Pytest Output

```text
8 passed, 1 warning in 64.15s (0:01:04)
```

### 2. Benchmark Numbers & Hardware Specifications

```text
Model: ResNet18
Input size: 224 × 224
Device: CPU

CPU Model: Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz 1.80 GHz
RAM: 8.00 GB

Requests: 50
p50: 57.8 ms
p95: 99.2 ms
Maximum latency: 17484.5 ms
Throughput: ~2.4 requests/sec
```

### 3. Capacity Estimation using Little's Law

* **Answer:** Assuming one request in flight at a time, throughput is approximately the reciprocal of the $p50$ latency:

  $$
  \text{Throughput} = \frac{1}{0.0578\text{ s}} \approx 17.3\text{ requests/sec}
  $$

  Multiplying this by $86,400\text{ seconds}$ in a day yields a theoretical capacity of approximately **1,494,720 requests per day** under continuous saturation. *Note: Practical capacity must be verified through real-world load testing.*

### 4. Edge Case Input Experiment

* **Input Experiment:** Uploaded a valid JPEG image to the `/analyze` endpoint paired with a text note exceeding 2,000 characters to test input validation limits.
* **Status Code:** `422 Unprocessable Entity`
* **API Response:**

  ```json
  {
    "detail": "Note longer than 2,000 characters"
  }
  ```

---

## 🛠️ Architecture & Performance Q&A

### 1. Handling Invalid File Types at `/predict`
* **Expected Status Code:** `415 Unsupported Media Type` (or `400 Bad Request`).
* **Why not 500?** A `500 Internal Server Error` is reserved for unexpected server-side crashes. Uploading an unsupported file format (e.g., PDF instead of JPEG/PNG) is a client-side error. Returning a `415` or `400` status code informs the client precisely what went wrong without crashing the server.

### 2. Model Loading in `lifespan` Context
* **Resource Efficiency:** Loading the PyTorch model (`week6_model.pt`) inside FastAPI's `lifespan` context manager ensures the model weights are loaded into RAM once upon server startup.
* **Performance:** If initialized inside the endpoint function, the model file would be re-read from disk on every incoming request, causing severe latency spikes and high disk I/O.

### 3. Latency Metrics Analysis ($p50 = 40\text{ ms}$, $p95 = 400\text{ ms}$)
* **Analysis:** While median performance ($p50$) is fast at $40\text{ ms}$, a high tail latency ($p95 = 400\text{ ms}$) indicates that $5\%$ of requests take at least $10\times$ longer. This suggests queuing, resource contention, or thread-blocking behavior under load.
* **Checkpoints:**
  1. Concurrency limits and threadpool queue length.
  2. CPU throttling, context switching, or memory swapping on host hardware.
  3. Image size variations causing preprocessing delay.

### 4. Worker Allocation Calculation
Target load: 500,000 requests/day, peak factor of $5\times$, single-worker capacity of 25 requests/sec.

* **Average RPS:**

  $$
  \text{Average RPS} = \frac{500,000}{86,400} \approx 5.79\text{ req/sec}
  $$

* **Peak RPS:**

  $$
  \text{Peak RPS} = 5.79 \times 5 = 28.95\text{ req/sec}
  $$

* **Required Workers:**

  $$
  \text{Required Workers} = \frac{28.95}{25} \approx 1.16
  $$

* **Answer:** You need **2 workers** to handle peak traffic safely without dropping requests.

### 5. Synchronous Model Inference inside `async def`
* **Event Loop Blocking:** PyTorch model inference and image transformations are CPU-bound, synchronous operations. Executing them directly in an `async def` endpoint runs them on the main Python event loop thread, blocking all incoming async requests until inference completes.
* **Solution:** Offload CPU-bound tasks using `fastapi.concurrency.run_in_threadpool` or a background process pool.

### 6. Team Communication Channels
* **Formula:**

  $$
  \text{Channels} = \frac{N(N - 1)}{2}
  $$

* **Calculation ($N = 4$):**

  $$
  \text{Channels} = \frac{4 \times (4 - 1)}{2} = \frac{12}{2} = \mathbf{6\text{ channels}}
  $$