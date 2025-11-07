# NWOR/SCV Microbenchmark

## Configuration

```json
{
  "target_model": "meta-llama/Llama-3.1-8B-Instruct",
  "drafter_model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
  "scenario": "short",
  "num_requests": 36,
  "draft_tokens": 10,
  "batches": 10,
  "temperature": 0.0,
  "top_p": 0.9,
  "tensor_parallel_size": 2,
  "prompt_count": 3200,
  "prompt_shuffle_seed": 42,
  "max_model_len": 4096,
  "max_new_tokens": 64,
  "warmup_steps": 2,
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
  "output_path": "sweeps/scenario1_test.json"
}
```

## Summary

| SCV Mode | NWOR Mode | Batches | Avg Latency (s) | P50 (s) | P95 (s) | Tokens Staged | Tokens Committed | Writes Saved % | Avg Accepted/window | Acceptance Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| off | off | 10 | 4.9375 | 4.9620 | 5.0893 | 0 | 0 | 0.00 | 0.49 | 0.05 |
| off | on | 10 | 2.9652 | 3.0084 | 3.0637 | 0 | 0 | 0.00 | 0.47 | 0.09 |