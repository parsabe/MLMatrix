# MPI Ping Pong Benchmark Analysis

## Overview
This project benchmarks the point-to-point communication performance between two MPI ranks. The program consists of two tasks:
1.  **Task A (Logic Check):** Runs a loop of 50,000 message exchanges (ping-pongs) to ensure the logic and synchronization are correct.
2.  **Task B (Timing):** Repeats the loop while measuring the execution time to calculate the network latency.

## Performance Analysis

### Raw Data from Task B
* **Total Round Trips:** 50,000
* **Total Messages Sent:** 100,000 (Rank 0 sends + Rank 1 sends)
* **Total Duration:** 0.176818 seconds

### Latency Calculation
The latency represents the time taken to send a single message from one rank to another.

**1. Base Calculation (Seconds)**
$$\text{Latency (s)} = \frac{\text{Total Time}}{\text{Total Messages}} = \frac{0.176818}{100,000} = 0.000001768 \text{ s}$$

**2. Conversion to Milliseconds ($ms$)**
$$0.000001768 \times 10^3 = 0.001768 \text{ ms}$$

**3. Conversion to Microseconds ($\mu s$)**
$$0.000001768 \times 10^6 = 1.768 \mu s$$

### Conclusion
The results confirm the code is now functioning correctly with accurate unit conversions:
* The reported latency of **1.768 microseconds** is highly consistent with shared-memory MPI communication on a single node.
* The millisecond value (**0.001768 ms**) correctly matches the microsecond value, verifying the formula logic.