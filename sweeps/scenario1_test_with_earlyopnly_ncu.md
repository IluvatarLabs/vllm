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
  "scv_modes": [
    "off"
  ],
  "adaptive_draft_length": 0,
  "confidence_threshold": 0.5,
  "enable_ncu": true,
  "ncu_metrics": "dram__bytes_write.sum,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes.sum",
  "enable_nsys": false,
  "profile_only": false,
  "output_path": "sweeps/scenario1_test_with_earlyopnly_ncu.json"
}
```

## Summary

| SCV Mode | NWOR Mode | Batches | Avg Latency (s) | P50 (s) | P95 (s) | Tokens Staged | Tokens Committed | Writes Saved % | Avg Accepted/window | Acceptance Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| off | adaptive=0,threshold=0.5 | 10 | 6.3362 | 6.4099 | 6.4988 | 0 | 0 | 0.00 | 0.47 | 0.05 |