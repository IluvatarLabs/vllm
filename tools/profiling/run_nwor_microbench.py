#!/usr/bin/env python3
"""
NWOR microbenchmark harness for speculative decoding.

Example:
  python tools/profiling/run_nwor_microbench.py \
      --scenario short --batches 4 --requests 8 --draft-tokens 4 \
      --temperature 0.0 --output results.json

Environment overrides:
  TARGET_MODEL=... DRAFT_MODEL=... python ...
"""

import argparse
import gc
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
import torch
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List

from datasets import load_dataset

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter as MetricCounter, Gauge as MetricGauge
from vllm.v1.metrics.reader import Vector as MetricVector


DEFAULT_TARGET_MODEL = os.getenv(
    "TARGET_MODEL", "meta-llama/Llama-3.2-3B-Instruct"
)
DEFAULT_DRAFT_MODEL = os.getenv(
    "DRAFT_MODEL", "linborui/EAGLE-Llama-3.2-3B-Instruct"
)

SCENARIOS = {
    "short": dict(
        dataset="OpenAssistant/oasst1",
        split="train",
        fields=["prompt", "text", "instruction"],
        min_chars=1,
        max_chars=800,
    ),
    "medium": dict(
        dataset="abisee/cnn_dailymail",
        name="3.0.0",
        split="train",
        fields=["article", "text"],
        min_chars=800,
        max_chars=2000,
    ),
    "long": dict(
        dataset="abisee/cnn_dailymail",
        name="3.0.0",
        split="train",
        fields=["article", "text"],
        min_chars=2000,
        max_chars=None,
    ),
    "mixed": dict(
        dataset="Open-Orca/OpenOrca",
        split="train",
        fields=["text", "response", "output"],
        min_chars=1,
        max_chars=None,
    ),
      "squad": dict(
      dataset="squad",
      split="train",
      fields=["context", "question"],
      min_chars=200,
      max_chars=1500,
    ),
      "coding": dict(
      dataset="bigcode/bigcodebench",
      split="v0.1.4",  # BigCodeBench uses versions as splits
      fields=["instruct_prompt"],  # Contains natural language instructions
      min_chars=50,
      max_chars=1000,
    ),
      "alpaca": dict(
      dataset="tatsu-lab/alpaca",
      split="train",
      fields=["instruction", "input"],
      min_chars=50,
      max_chars=1000,
    ),
    "quora": dict(
        dataset="quora/questions",
        split="train",
        fields=["question1", "question2"],
        min_chars=50,
        max_chars=1000,
    ),
}


@dataclass
class RunConfig:
    target_model: str
    drafter_model: str
    scenario: str
    num_requests: int
    draft_tokens: int
    batches: int
    temperature: float
    top_p: float
    tensor_parallel_size: int
    prompt_count: int
    prompt_shuffle_seed: int
    max_model_len: int | None
    max_new_tokens: int
    warmup_steps: int
    measure_steps: int
    spec_method: str
    scv_modes: List[str]
    adaptive_draft_length: int
    confidence_threshold: float
    no_speculation: bool
    enable_ncu: bool
    ncu_metrics: str
    enable_nsys: bool
    profile_only: bool
    output_path: str
    seed: int


def pick_prompts(config: RunConfig) -> List[str]:
    info = SCENARIOS[config.scenario]
    ds = load_dataset(
        info["dataset"],
        info.get("name"),
        split=info["split"],
    )
    min_chars = info.get("min_chars") or 0
    max_chars = info.get("max_chars") or 1_000_000

    candidates = []
    for record in ds:
        texts: List[str] = []
        for field in info["fields"]:
            value = record.get(field)
            if isinstance(value, str):
                texts.append(value)
        if not texts:
            continue
        text = "\n".join(t.strip() for t in texts if t)
        if min_chars <= len(text) <= max_chars:
            candidates.append(text)
        if len(candidates) >= config.prompt_count * config.num_requests:
            break

    if not candidates:
        raise RuntimeError(
            f"No prompts found for scenario '{config.scenario}'. "
            "Consider lowering min/max char filters."
        )

    random.seed(config.prompt_shuffle_seed)
    random.shuffle(candidates)
    total_needed = (config.warmup_steps + config.batches) * config.num_requests
    if len(candidates) < total_needed:
        raise RuntimeError(
            f"Not enough prompts ({len(candidates)}) for warmup + measurement "
            f"needs ({total_needed}). Increase --prompt-count or adjust batching."
        )
    return candidates[:total_needed]


def build_engine(config: RunConfig) -> LLM:
    llm_kwargs: dict[str, Any] = {
        "model": config.target_model,
        "tensor_parallel_size": config.tensor_parallel_size,
        # Enable Prometheus stats so NWOR metrics appear in microbench output.
        "disable_log_stats": False,
    }

    # Only add speculative_config if speculation is enabled
    if not config.no_speculation:
        speculative_config = {
            "method": config.spec_method,
            "model": config.drafter_model,
            "num_speculative_tokens": config.draft_tokens,
        }
        llm_kwargs["speculative_config"] = speculative_config

    if config.max_model_len is not None:
        llm_kwargs["max_model_len"] = config.max_model_len
    return LLM(**llm_kwargs)


def run_batch(
    engine: LLM,
    prompts: Iterable[str],
    config: RunConfig,
    nwor_mode: str,
    batch_index: int,
    scv_mode: str,
) -> dict[str, Any]:
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
    )

    prompt_list = list(prompts)
    start = time.time()
    request_outputs = engine.generate(prompt_list, sampling_params=sampling_params, use_tqdm=False)
    duration = time.time() - start

    texts = [
        output.outputs[0].text if output.outputs else ""
        for output in request_outputs
    ]

    return {
        "nwor_mode": nwor_mode,
        "scv_mode": scv_mode,
        "batch_index": batch_index,
        "latency_s": duration,
        "outputs": texts,
        "sampling_params": {
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "max_tokens": sampling_params.max_tokens,
        },
    }


def snapshot_metrics(engine: LLM | None = None) -> dict[str, float | list[int]]:
    totals: dict[str, float | list[int]] = defaultdict(float)
    metrics = engine.get_metrics() if engine is not None else []
    if engine is None:
        # Fallback path if an engine handle is not available.
        try:
            from vllm.v1.metrics.reader import get_metrics_snapshot  # type: ignore
        except ImportError:
            metrics = []
        else:
            metrics = get_metrics_snapshot()

    for metric in metrics:
        if isinstance(metric, MetricCounter):
            totals[metric.name] += metric.value
        elif isinstance(metric, MetricGauge):
            totals[metric.name] += metric.value
        elif isinstance(metric, MetricVector):
            if metric.name not in totals:
                totals[metric.name] = [0] * len(metric.values)
            current = totals[metric.name]
            assert isinstance(current, list)
            for idx, val in enumerate(metric.values):
                current[idx] += val
    return totals


def diff_metrics(
    after: dict[str, float | list[int]],
    before: dict[str, float | list[int]],
) -> dict[str, float]:
    diff: dict[str, float] = {}
    keys = set(before.keys()) | set(after.keys())
    for name in keys:
        after_val = after.get(name)
        before_val = before.get(name)
        if isinstance(after_val, list) or isinstance(before_val, list):
            # Skip vector metrics for now.
            continue
        base_value = float(after_val or 0.0) - float(before_val or 0.0)
        diff[name] = base_value
        if name.endswith("_total"):
            base_name = name[: -len("_total")]
            diff.setdefault(base_name, base_value)
    return diff


def run_microbenchmark(config: RunConfig) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, float]], dict[tuple[str, str], float]]:
    prompts = pick_prompts(config)
    results: list[dict[str, Any]] = []
    metrics_delta: dict[tuple[str, str], dict[str, float]] = {}
    peak_memory: dict[tuple[str, str], float] = {}

    # Set NWOR environment variables from config
    os.environ["VLLM_NWOR_ADAPTIVE_DRAFT_LENGTH"] = str(config.adaptive_draft_length)
    os.environ["VLLM_NWOR_CONFIDENCE_THRESHOLD"] = str(config.confidence_threshold)

    # Generate descriptive label for results
    if config.no_speculation:
        nwor_label = "vanilla"
    else:
        nwor_label = f"adaptive={config.adaptive_draft_length},threshold={config.confidence_threshold}"

    for scv_mode in config.scv_modes:
        os.environ["VLLM_SCV_MODE"] = scv_mode or "off"

        engine = build_engine(config)

        # Reset peak memory stats after engine initialization
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        prompt_offset = 0
        # Warmup (not recorded)
        for _ in range(config.warmup_steps):
            warm_prompts = prompts[prompt_offset : prompt_offset + config.num_requests]
            prompt_offset += config.num_requests
            run_batch(engine, warm_prompts, config, nwor_label, -1, scv_mode)

        metrics_before = snapshot_metrics(engine)

        for batch_idx in range(config.batches):
            start = prompt_offset + batch_idx * config.num_requests
            end = start + config.num_requests
            batch_prompts = prompts[start:end]
            result = run_batch(
                engine, batch_prompts, config, nwor_label, batch_idx, scv_mode
            )
            results.append(result)

        metrics_after = snapshot_metrics(engine)
        delta = diff_metrics(metrics_after, metrics_before)
        metrics_delta[(scv_mode, nwor_label)] = delta

        # Capture peak memory usage across all GPUs
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                peak_mem_gb = max(
                    torch.cuda.max_memory_allocated(device=i) / 1024**3
                    for i in range(torch.cuda.device_count())
                )
            else:
                peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        else:
            peak_mem_gb = 0.0

        peak_memory[(scv_mode, nwor_label)] = peak_mem_gb

        # Explicitly delete engine to free GPU memory before next iteration
        del engine
        gc.collect()

    return results, metrics_delta, peak_memory


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="NWOR microbenchmark harness")
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="short")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--prompt-count", type=int, default=100)
    parser.add_argument("--prompt-shuffle-seed", type=int, default=1234)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=1)
    parser.add_argument(
        "--adaptive-draft-length",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable/disable adaptive draft length (Multi-Graph Adaptive + Per-Request Predict). "
             "0=disabled (fixed draft length), 1=enabled (default). "
             "When enabled with threshold=0.0, preserves correctness while improving performance.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Confidence threshold for Batch Early Exit (0.0-1.0, default=0.0). "
             "Controls early stopping during draft generation. Set to 0.0 to disable (recommended "
             "for correctness). Higher values may improve performance but can change outputs.",
    )
    parser.add_argument(
        "--scv-modes",
        default="off",
        help="Comma-separated list of SCV modes to benchmark (default: off)",
    )
    parser.add_argument(
        "--spec-method",
        default="eagle",
        help="Speculative method to use (default: eagle).",
    )
    parser.add_argument(
        "--no-speculation",
        action="store_true",
        help="Disable speculative decoding entirely (vanilla vLLM, no draft model).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--enable-ncu",
        action="store_true",
        help="Run an additional pass under Nsight Compute (nv-nsight-cu-cli).",
    )
    parser.add_argument(
        "--ncu-metrics",
        default="dram__bytes_write.sum,lts__t_sectors_op_write.sum",
        help="Comma-separated Nsight Compute metrics to collect when --enable-ncu is set.",
    )
    parser.add_argument(
        "--enable-nsys",
        action="store_true",
        help="Run an additional pass under Nsight Systems.",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output", default="nwor_microbench.json")
    args = parser.parse_args()

    scv_modes = [mode.strip() for mode in args.scv_modes.split(",") if mode.strip()]

    # Validate confidence threshold
    if args.confidence_threshold < 0.0 or args.confidence_threshold > 1.0:
        parser.error("--confidence-threshold must be between 0.0 and 1.0")

    return RunConfig(
        target_model=args.target_model,
        drafter_model=args.draft_model,
        scenario=args.scenario,
        num_requests=args.requests,
        draft_tokens=args.draft_tokens,
        batches=args.batches,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        prompt_count=args.prompt_count,
        prompt_shuffle_seed=args.prompt_shuffle_seed,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        spec_method=args.spec_method,
        scv_modes=scv_modes or ["off"],
        adaptive_draft_length=args.adaptive_draft_length,
        confidence_threshold=args.confidence_threshold,
        no_speculation=args.no_speculation,
        enable_ncu=args.enable_ncu,
        ncu_metrics=args.ncu_metrics,
        enable_nsys=args.enable_nsys,
        profile_only=args.profile_only,
        output_path=args.output,
        seed=args.seed,
    )


def summarize_results(
    results: list[dict[str, Any]],
    metrics_delta: dict[tuple[str, str], dict[str, float]],
    peak_memory: dict[tuple[str, str], float] | None = None,
    ncu_metrics: dict[tuple[str, str], dict[str, float]] | None = None,
) -> dict[str, Any]:
    summary: dict[tuple[str, str], dict[str, Any]] = {}

    for result in results:
        key = (result["scv_mode"], result["nwor_mode"])
        entry = summary.setdefault(
            key,
            {
                "latencies": [],
                "batches": 0,
            },
        )
        entry["latencies"].append(result["latency_s"])
        entry["batches"] += 1

    summary_output = []
    for (scv_mode, nwor_mode), entry in summary.items():
        latencies = entry["latencies"]
        latency_avg = statistics.mean(latencies) if latencies else 0.0
        if len(latencies) >= 2:
            p50 = statistics.quantiles(latencies, n=100, method="inclusive")[49]
            p95 = statistics.quantiles(latencies, n=100, method="inclusive")[94]
        else:
            p50 = latencies[0] if latencies else 0.0
            p95 = p50

        metrics = metrics_delta.get((scv_mode, nwor_mode), {})
        committed = int(
            metrics.get(
                "vllm:nwor_committed_tokens",
                metrics.get("vllm:nwor_committed_tokens_total", 0),
            )
        )
        rejected = int(
            metrics.get(
                "vllm:nwor_rejected_tokens",
                metrics.get("vllm:nwor_rejected_tokens_total", 0),
            )
        )
        staged = committed + rejected
        writes_saved_pct = (
            (1 - committed / staged) * 100.0 if staged > 0 else 0.0
        )

        spec_drafts = int(metrics.get("vllm:spec_decode_num_drafts", 0))
        spec_draft_tokens = int(metrics.get("vllm:spec_decode_num_draft_tokens", 0))
        spec_accepted_tokens = int(metrics.get("vllm:spec_decode_num_accepted_tokens", 0))
        avg_acceptance_per_window = (
            spec_accepted_tokens / spec_drafts if spec_drafts > 0 else 0.0
        )
        acceptance_ratio = (
            spec_accepted_tokens / spec_draft_tokens
            if spec_draft_tokens > 0
            else 0.0
        )

        metrics_extra = (ncu_metrics or {}).get((scv_mode, nwor_mode), {})
        peak_mem_gb = (peak_memory or {}).get((scv_mode, nwor_mode), 0.0)
        summary_output.append(
            {
                "scv_mode": scv_mode,
                "nwor_mode": nwor_mode,
                "batches": entry["batches"],
                "latency_avg_s": latency_avg,
                "latency_p50_s": p50,
                "latency_p95_s": p95,
                "nwor_tokens_committed": committed,
                "nwor_tokens_staged": staged,
                "nwor_writes_saved_pct": writes_saved_pct,
                "spec_num_drafts": spec_drafts,
                "spec_num_draft_tokens": spec_draft_tokens,
                "spec_num_accepted_tokens": spec_accepted_tokens,
                "spec_avg_accepted_per_window": avg_acceptance_per_window,
                "spec_acceptance_ratio": acceptance_ratio,
                "peak_memory_gb": peak_mem_gb,
                "ncu_metrics": metrics_extra,
            }
        )

    return {"per_mode": summary_output}


def write_markdown_summary(config: RunConfig, summary: dict[str, Any], path: Path) -> None:
    lines = []
    lines.append(f"# NWOR/SCV Microbenchmark\n")
    lines.append("## Configuration\n")
    lines.append("```json")
    lines.append(json.dumps(config.__dict__, indent=2))
    lines.append("```")
    lines.append("\n## Summary\n")
    # Determine optional NCU metric columns
    metric_names: list[str] = []
    for row in summary["per_mode"]:
        for metric_name in row.get("ncu_metrics", {}):
            if metric_name not in metric_names:
                metric_names.append(metric_name)

    header_cols = [
        "SCV Mode",
        "NWOR Mode",
        "Batches",
        "Avg Latency (s)",
        "P50 (s)",
        "P95 (s)",
        "Tokens Staged",
        "Tokens Committed",
        "Writes Saved %",
        "Avg Accepted/window",
        "Acceptance Ratio",
    ] + metric_names
    header = "| " + " | ".join(header_cols) + " |"
    separator = "| " + " | ".join("---" for _ in header_cols) + " |"
    lines.append(header)
    lines.append(separator)
    for row in summary["per_mode"]:
        values = [
            row["scv_mode"],
            row["nwor_mode"],
            str(row["batches"]),
            f"{row['latency_avg_s']:.4f}",
            f"{row['latency_p50_s']:.4f}",
            f"{row['latency_p95_s']:.4f}",
            str(row["nwor_tokens_staged"]),
            str(row["nwor_tokens_committed"]),
            f"{row['nwor_writes_saved_pct']:.2f}",
            f"{row['spec_avg_accepted_per_window']:.2f}",
            f"{row['spec_acceptance_ratio']:.2f}",
        ]
        metrics_extra = row.get("ncu_metrics", {})
        for name in metric_names:
            value = metrics_extra.get(name)
            values.append(f"{value:.3e}" if value is not None else "")
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def config_to_args(
    config: RunConfig,
    *,
    output_path: str,
    profile_only: bool = False,
    override_scv_mode: str | None = None,
) -> list[str]:
    args = [
        "--target-model",
        config.target_model,
        "--draft-model",
        config.drafter_model,
        "--scenario",
        config.scenario,
        "--requests",
        str(config.num_requests),
        "--draft-tokens",
        str(config.draft_tokens),
        "--batches",
        str(config.batches),
        "--temperature",
        str(config.temperature),
        "--top-p",
        str(config.top_p),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--prompt-count",
        str(config.prompt_count),
        "--prompt-shuffle-seed",
        str(config.prompt_shuffle_seed),
    ]
    if config.max_model_len is not None:
        args.extend(["--max-model-len", str(config.max_model_len)])
    args.extend([
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--warmup-steps",
        str(config.warmup_steps),
        "--measure-steps",
        str(config.measure_steps),
        "--adaptive-draft-length",
        str(config.adaptive_draft_length),
        "--confidence-threshold",
        str(config.confidence_threshold),
        "--scv-modes",
        override_scv_mode or ",".join(config.scv_modes),
        "--seed",
        str(config.seed),
        "--output",
        output_path,
    ])
    if config.no_speculation:
        args.append("--no-speculation")
    if profile_only:
        args.append("--profile-only")
    return args


def run_ncu_profiles(config: RunConfig, output_json: Path) -> dict[tuple[str, str], dict[str, float]]:
    metrics_map: dict[tuple[str, str], dict[str, float]] = {}
    script_path = Path(__file__).resolve()
    env = os.environ.copy()
    metric_names = [m.strip() for m in config.ncu_metrics.split(",") if m.strip()]

    # Generate NWOR label for this configuration
    if config.no_speculation:
        nwor_label = "vanilla"
    else:
        nwor_label = f"adaptive={config.adaptive_draft_length},threshold={config.confidence_threshold}"

    for scv_mode in config.scv_modes:
        suffix = f".{scv_mode or 'off'}-adaptive{config.adaptive_draft_length}-t{config.confidence_threshold}"
        csv_path = output_json.with_suffix(f"{suffix}.ncu.csv")
        rep_path = output_json.with_suffix(f"{suffix}.ncu")
        profile_json = output_json.with_suffix(f"{suffix}.ncu.json")
        args = config_to_args(
            config,
            output_path=str(profile_json),
            profile_only=True,
            override_scv_mode=scv_mode,
        )
        # Try ncu in common locations
        ncu_cmd = None
        for candidate in ["/usr/local/cuda/bin/ncu", "ncu", "nv-nsight-cu-cli"]:
            if Path(candidate).exists() or shutil.which(candidate):
                ncu_cmd = candidate
                break
        if not ncu_cmd:
            print("[WARN] NCU not found. Skipping NCU collection.")
            return {}
        cmd = [
            ncu_cmd,
            "-f",  # Force overwrite existing report files
            "--csv",
            "--metrics",
            ",".join(metric_names),
            "--target-processes",
            "all",
            # Filter to only KV cache write kernels for faster profiling
            "--kernel-name", "regex:reshape_and_cache.*",
            "--launch-skip", "5000",   # Skip warmup cache kernels (~4,320)
            "--launch-count", "2000",  # Profile ~33 measurement requests
            "-o",
            str(rep_path),
            sys.executable,
            str(script_path),
        ] + args
        try:
            # Step 1: Run NCU profiling to create binary report
            # Note: --csv doesn't output to stdout with --target-processes all
            result = subprocess.run(cmd, check=True, env=env,
                                  capture_output=True, text=True)
            if result.stderr:
                print(f"[INFO] NCU capture stderr for scv_mode={scv_mode}:\n{result.stderr}")

            # Step 2: Export CSV from binary report
            # Multi-process profiling requires separate export step
            import time
            # Wait a moment for file system sync
            time.sleep(0.5)

            if not rep_path.exists():
                print(f"[WARN] NCU report not created: {rep_path}")
                continue

            print(f"[INFO] Exporting CSV from NCU report: {rep_path}")
            export_cmd = [
                ncu_cmd,
                "--import", str(rep_path),
                "--csv",
                "--page", "raw",
                "--metrics", ",".join(metric_names),
            ]

            try:
                export_result = subprocess.run(export_cmd, check=False,
                                              capture_output=True, text=True,
                                              timeout=300)

                if export_result.returncode != 0:
                    print(f"[WARN] NCU export failed with return code {export_result.returncode}")
                    print(f"[WARN] stderr: {export_result.stderr}")
                    continue

                # Write exported CSV to file
                csv_path.write_text(export_result.stdout, encoding="utf-8")

                if not export_result.stdout.strip():
                    print(f"[WARN] NCU export produced empty CSV for scv_mode={scv_mode}")
                else:
                    lines = len(export_result.stdout.strip().split('\n'))
                    print(f"[INFO] NCU export successful: {lines} lines written to {csv_path}")

            except subprocess.TimeoutExpired:
                print(f"[WARN] NCU export timed out after 300 seconds")
                continue

        except FileNotFoundError as exc:
            print(f"[WARN] {ncu_cmd} not found: {exc}. Skipping NCU collection.")
            return {}
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] {ncu_cmd} failed for scv_mode={scv_mode}: {exc}")
            if hasattr(exc, 'stderr') and exc.stderr:
                print(f"[WARN] stderr: {exc.stderr}")
            continue

        metrics = parse_ncu_csv(csv_path, metric_names)
        metrics_map[(scv_mode, nwor_label)] = metrics
    return metrics_map


def parse_ncu_csv(path: Path, metric_names: list[str]) -> dict[str, float]:
    """Parse NCU CSV output and sum metrics across all kernel launches.

    NCU --csv outputs one line per kernel launch per metric. We sum all values
    to get total bandwidth/operations across the profiled kernel launches.
    """
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            name, _unit, value = parts[:3]
            if name in metric_names:
                try:
                    # Sum values across multiple kernel launches
                    metrics[name] = metrics.get(name, 0.0) + float(value)
                except ValueError:
                    pass
    return metrics


def main() -> None:
    config = parse_args()

    # Build output directory structure: sweeps/{target_model}/{dataset}/seed_{seed}/
    target_name = Path(config.target_model).name
    model_pair = target_name  # Use target model only, not draft model

    output_dir = Path("sweeps") / model_pair / config.scenario / f"seed_{config.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use specified output filename, or generate simple default
    if config.output_path != "nwor_microbench.json":
        output_base = Path(config.output_path).name
    else:
        # Simple default (scripts should pass --output for descriptive names)
        spec_type = "vanilla" if config.no_speculation else f"adaptive{config.adaptive_draft_length}"
        output_base = f"run_{spec_type}.json"

    output_json = output_dir / output_base

    results, metrics_delta, peak_memory = run_microbenchmark(config)
    ncu_metrics_map: dict[tuple[str, str], dict[str, float]] | None = None

    if config.enable_ncu:
        ncu_metrics_map = run_ncu_profiles(config, output_json)

    summary = summarize_results(results, metrics_delta, peak_memory=peak_memory, ncu_metrics=ncu_metrics_map)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config.__dict__,
                "summary": summary,
                "results": results,
            },
            f,
            indent=2,
        )

    output_md = output_json.with_suffix(".md")
    write_markdown_summary(config, summary, output_md)
    print(f"Wrote benchmark output to {output_json} and {output_md}")

    if config.enable_nsys and not config.profile_only:
        # Run Nsight Systems once over all modes
        script_path = Path(__file__).resolve()
        env = os.environ.copy()
        nsys_output = output_json.with_suffix(".nsys")
        args = config_to_args(
            config,
            output_path=str(output_json.with_suffix(".nsys.json")),
            profile_only=True,
        )
        cmd = [
            "nsys",
            "profile",
            "-t",
            "cuda,nvtx,osrt",
            "-o",
            str(nsys_output),
            sys.executable,
            str(script_path),
        ] + args
        try:
            subprocess.run(cmd, check=True, env=env)
        except FileNotFoundError as exc:
            print(f"[WARN] nsys not found: {exc}. Skipping Nsight Systems collection.")
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] nsys failed: {exc}")


if __name__ == "__main__":
    main()
