# NWOR/SCV Microbenchmark

## Configuration

```json
{
  "target_model": "meta-llama/Llama-3.2-3B-Instruct",
  "drafter_model": "linborui/EAGLE-Llama-3.2-3B-Instruct",
  "scenario": "medium",
  "num_requests": 48,
  "draft_tokens": 12,
  "batches": 6,
  "temperature": 0.7,
  "top_p": 1.0,
  "tensor_parallel_size": 1,
  "prompt_count": 3200,
  "prompt_shuffle_seed": 42,
  "max_model_len": 4096,
  "max_new_tokens": 128,
  "warmup_steps": 1,
  "measure_steps": 6,
  "spec_method": "eagle",
  "nwor_modes": [
    "off",
    "on"
  ],
  "scv_modes": [
    "off"
  ],
  "enable_ncu": false,
  "ncu_metrics": "dram__bytes_write.sum,lts__t_sectors_op_write.sum",
  "enable_nsys": false,
  "profile_only": false,
  "output_path": "sweeps/scenario2_test.json"
}
```

## Summary

| SCV Mode | NWOR Mode | Batches | Avg Latency (s) | P50 (s) | P95 (s) | Tokens Staged | Tokens Committed | Writes Saved % | Avg Accepted/window | Acceptance Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| off | off | 6 | 12.2582 | 12.2129 | 12.6768 | 0 | 0 | 0.00 | 0.48 | 0.04 |
| off | on | 6 | 8.2619 | 8.2263 | 8.4137 | 0 | 0 | 0.00 | 0.47 | 0.08 |