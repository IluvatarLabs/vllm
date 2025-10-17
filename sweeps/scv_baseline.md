# NWOR/SCV Microbenchmark

## Configuration

```json
{
  "target_model": "meta-llama/Llama-3.2-3B-Instruct",
  "drafter_model": "linborui/EAGLE-Llama-3.2-3B-Instruct",
  "scenario": "short",
  "num_requests": 8,
  "draft_tokens": 4,
  "batches": 6,
  "temperature": 0.7,
  "top_p": 1.0,
  "tensor_parallel_size": 1,
  "prompt_count": 100,
  "prompt_shuffle_seed": 1234,
  "max_model_len": 8192,
  "max_new_tokens": 32,
  "warmup_steps": 1,
  "measure_steps": 1,
  "spec_method": "eagle",
  "nwor_modes": [
    "off",
    "stage"
  ],
  "scv_modes": [
    "off",
    "graph",
    "adaptive"
  ],
  "enable_ncu": false,
  "ncu_metrics": "dram__bytes_write.sum,lts__t_sectors_op_write.sum",
  "enable_nsys": false,
  "profile_only": false,
  "output_path": "sweeps/scv_baseline.json"
}
```

## Summary

| SCV Mode | NWOR Mode | Batches | Avg Latency (s) | P50 (s) | P95 (s) | Tokens Staged | Tokens Committed | Writes Saved % | Avg Accepted/window | Acceptance Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| off | off | 6 | 0.5965 | 0.6060 | 0.6196 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| off | stage | 6 | 0.6083 | 0.6199 | 0.6392 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| graph | off | 6 | 0.5934 | 0.6058 | 0.6211 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| graph | stage | 6 | 0.6078 | 0.6201 | 0.6373 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| adaptive | off | 6 | 0.5917 | 0.6031 | 0.6212 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| adaptive | stage | 6 | 0.6124 | 0.6256 | 0.6409 | 0 | 0 | 0.00 | 0.00 | 0.00 |