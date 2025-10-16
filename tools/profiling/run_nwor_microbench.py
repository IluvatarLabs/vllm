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
import json
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, List

from datasets import load_dataset

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.spec_decode import SpeculativeConfig, SpeculativeMethod


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
    prompt_count: int
    prompt_shuffle_seed: int
    max_new_tokens: int
    warmup_steps: int
    measure_steps: int
    nwor_modes: List[str]
    scv_modes: List[str]
    enable_ncu: bool
    ncu_metrics: str
    enable_nsys: bool
    profile_only: bool
    output_path: str


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


def build_engine(config: RunConfig) -> AsyncLLMEngine:
    speculative_config = SpeculativeConfig(
        method=SpeculativeMethod.EAGLE,
        draft_model=config.drafter_model,
        num_speculative_tokens=config.draft_tokens,
    )
    engine_args = AsyncEngineArgs(
        model=config.target_model,
        target_device=os.getenv("VLLM_TARGET_DEVICE", "cuda"),
        tensor_parallel_size=1,
        speculative_config=speculative_config,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


def run_batch(
    engine: AsyncLLMEngine,
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

    start = time.time()
    futures = []
    for i, prompt in enumerate(prompts):
        request_id = f"nwor-run-{batch_index}-{nwor_mode}-{i}"
        futures.append(
            engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )
        )
    outputs = [future.result() for future in futures]
    duration = time.time() - start

    scheduler_stats_obj = engine.get_engine_context().scheduler_stats
    scheduler_stats = asdict(scheduler_stats_obj)

    return {
        "nwor_mode": nwor_mode,
        "scv_mode": scv_mode,
        "batch_index": batch_index,
        "latency_s": duration,
        "scheduler_stats": scheduler_stats,
        "outputs": [output.outputs[0].text if output.outputs else "" for output in outputs],
        "sampling_params": sampling_params.to_dict(),
    }


def run_microbenchmark(config: RunConfig) -> list[dict[str, Any]]:
    prompts = pick_prompts(config)
    results: list[dict[str, Any]] = []

    for scv_mode in config.scv_modes:
        os.environ["VLLM_SCV_MODE"] = scv_mode or "off"

        for nwor_mode in config.nwor_modes:
            os.environ["VLLM_NWOR_MODE"] = nwor_mode or "off"
            engine = build_engine(config)

            prompt_offset = 0
            # Warmup (not recorded)
            for _ in range(config.warmup_steps):
                warm_prompts = prompts[prompt_offset : prompt_offset + config.num_requests]
                prompt_offset += config.num_requests
                run_batch(engine, warm_prompts, config, nwor_mode, -1, scv_mode)

            for batch_idx in range(config.batches):
                start = prompt_offset + batch_idx * config.num_requests
                end = start + config.num_requests
                batch_prompts = prompts[start:end]
                result = run_batch(
                    engine, batch_prompts, config, nwor_mode, batch_idx, scv_mode
                )
                results.append(result)

            engine.shutdown()

    return results


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
    parser.add_argument("--prompt-count", type=int, default=100)
    parser.add_argument("--prompt-shuffle-seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=1)
    parser.add_argument(
        "--nwor-modes",
        default="off,stage",
        help="Comma-separated list of NWOR modes to benchmark (default: off,stage)",
    )
    parser.add_argument(
        "--scv-modes",
        default="off",
        help="Comma-separated list of SCV modes to benchmark (default: off)",
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

    nwor_modes = [mode.strip() for mode in args.nwor_modes.split(",") if mode.strip()]
    scv_modes = [mode.strip() for mode in args.scv_modes.split(",") if mode.strip()]

    return RunConfig(
        target_model=args.target_model,
        drafter_model=args.draft_model,
        scenario=args.scenario,
        num_requests=args.requests,
        draft_tokens=args.draft_tokens,
        batches=args.batches,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_count=args.prompt_count,
        prompt_shuffle_seed=args.prompt_shuffle_seed,
        max_new_tokens=args.max_new_tokens,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        nwor_modes=nwor_modes or ["off"],
        scv_modes=scv_modes or ["off"],
        enable_ncu=args.enable_ncu,
        ncu_metrics=args.ncu_metrics,
        enable_nsys=args.enable_nsys,
        profile_only=args.profile_only,
        output_path=args.output,
    )


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[tuple[str, str], dict[str, Any]] = {}

    for result in results:
        key = (result["scv_mode"], result["nwor_mode"])
        entry = summary.setdefault(
            key,
            {
                "latencies": [],
                "nwor_committed": 0,
                "nwor_rejected": 0,
                "nwor_tokens_staged": 0,
                "spec_num_drafts": 0,
                "spec_num_draft_tokens": 0,
                "spec_num_accepted_tokens": 0,
                "batches": 0,
            },
        )
        entry["latencies"].append(result["latency_s"])
        entry["batches"] += 1
        stats = result.get("scheduler_stats") or {}
        nwor_stats = stats.get("nwor_stats") or {}
        entry["nwor_committed"] += int(nwor_stats.get("tokens_committed", 0))
        entry["nwor_rejected"] += int(nwor_stats.get("tokens_rejected", 0))
        entry["nwor_tokens_staged"] += int(nwor_stats.get("tokens_staged", 0))

        spec_stats = stats.get("spec_decoding_stats") or {}
        entry["spec_num_drafts"] += int(spec_stats.get("num_drafts", 0))
        entry["spec_num_draft_tokens"] += int(spec_stats.get("num_draft_tokens", 0))
        entry["spec_num_accepted_tokens"] += int(
            spec_stats.get("num_accepted_tokens", 0)
        )

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

        committed = entry["nwor_committed"]
        staged = entry["nwor_tokens_staged"]
        writes_saved_pct = (
            (1 - committed / staged) * 100.0 if staged > 0 else 0.0
        )

        spec_drafts = entry["spec_num_drafts"]
        spec_draft_tokens = entry["spec_num_draft_tokens"]
        spec_accepted_tokens = entry["spec_num_accepted_tokens"]
        avg_acceptance_per_window = (
            spec_accepted_tokens / spec_drafts if spec_drafts > 0 else 0.0
        )
        acceptance_ratio = (
            spec_accepted_tokens / spec_draft_tokens
            if spec_draft_tokens > 0
            else 0.0
        )

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
    lines.append("| SCV Mode | NWOR Mode | Batches | Avg Latency (s) | P50 (s) | P95 (s) | Tokens Staged | Tokens Committed | Writes Saved % | Avg Accepted/window | Acceptance Ratio |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["per_mode"]:
        lines.append(
            f"| {row['scv_mode']} | {row['nwor_mode']} | {row['batches']} | "
            f"{row['latency_avg_s']:.4f} | {row['latency_p50_s']:.4f} | {row['latency_p95_s']:.4f} | "
            f"{row['nwor_tokens_staged']} | {row['nwor_tokens_committed']} | {row['nwor_writes_saved_pct']:.2f} | "
            f"{row['spec_avg_accepted_per_window']:.2f} | {row['spec_acceptance_ratio']:.2f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def config_to_args(config: RunConfig, *, output_path: str, profile_only: bool = False) -> list[str]:
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
        "--prompt-count",
        str(config.prompt_count),
        "--prompt-shuffle-seed",
        str(config.prompt_shuffle_seed),
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--warmup-steps",
        str(config.warmup_steps),
        "--measure-steps",
        str(config.measure_steps),
        "--nwor-modes",
        ",".join(config.nwor_modes),
        "--scv-modes",
        ",".join(config.scv_modes),
        "--output",
        output_path,
    ]
    if profile_only:
        args.append("--profile-only")
    return args


def run_with_profiler(config: RunConfig, profiler: str, base_args: list[str], output_stem: Path) -> None:
    script_path = Path(__file__).resolve()
    env = os.environ.copy()

    if profiler == "ncu":
        export_stem = str(output_stem) + ".ncu"
        cmd = [
            "nv-nsight-cu-cli",
            "--metrics",
            config.ncu_metrics,
            "--target-processes",
            "all",
            "-o",
            export_stem,
            sys.executable,
            str(script_path),
        ] + base_args
    elif profiler == "nsys":
        export_stem = str(output_stem) + ".nsys"
        cmd = [
            "nsys",
            "profile",
            "-t",
            "cuda,nvtx,osrt",
            "-o",
            export_stem,
            sys.executable,
            str(script_path),
        ] + base_args
    else:
        raise ValueError(f"Unsupported profiler: {profiler}")

    try:
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError as exc:
        print(f"[WARN] Profiler '{profiler}' not found: {exc}. Skipping.")


def main() -> None:
    config = parse_args()
    results = run_microbenchmark(config)
    summary = summarize_results(results)

    output_json = Path(config.output_path)
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

    if not config.profile_only:
        base_args = config_to_args(
            config,
            output_path=str(output_json.with_suffix(".profile.json")),
            profile_only=True,
        )
        if config.enable_ncu:
            run_with_profiler(config, "ncu", base_args, output_json.with_suffix(""))
        if config.enable_nsys:
            run_with_profiler(config, "nsys", base_args, output_json.with_suffix(""))


if __name__ == "__main__":
    main()
