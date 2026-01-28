## 2026-01-28 - [Optimization Masking]
**Learning:** Single-pass iteration optimization (O(N) vs 2*O(N)) on dense data structures might show negligible speedup in micro-benchmarks due to CPU branch prediction and cache efficiency masking the reduced memory access.
**Action:** Use sparse data patterns in benchmarks to reveal the true cost of traversal overhead when optimizing iteration logic.
