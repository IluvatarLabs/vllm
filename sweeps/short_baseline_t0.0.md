# NWOR/SCV Microbenchmark

## Configuration

```json
{
  "target_model": "meta-llama/Llama-3.2-3B-Instruct",
  "drafter_model": "linborui/EAGLE-Llama-3.2-3B-Instruct",
  "scenario": "short",
  "num_requests": 8,
  "draft_tokens": 4,
  "batches": 2,
  "temperature": 0.0,
  "top_p": 1.0,
  "tensor_parallel_size": 1,
  "prompt_count": 100,
  "prompt_shuffle_seed": 1234,
  "max_model_len": 8196,
  "max_new_tokens": 32,
  "warmup_steps": 1,
  "measure_steps": 1,
  "spec_method": "eagle",
  "nwor_modes": [
    "off"
  ],
  "scv_modes": [
    "off"
  ],
  "enable_ncu": false,
  "ncu_metrics": "dram__bytes_write.sum,lts__t_sectors_op_write.sum",
  "enable_nsys": false,
  "profile_only": false,
  "output_path": "sweeps/short_baseline_t0.0.json"
}
```

## Summary

| SCV Mode | NWOR Mode | Batches | Avg Latency (s) | P50 (s) | P95 (s) | Tokens Staged | Tokens Committed | Writes Saved % | Avg Accepted/window | Acceptance Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| off | off | 2 | 0.5739 | 0.5739 | 0.5982 | 0 | 0 | 0.00 | 0.64 | 0.16 |