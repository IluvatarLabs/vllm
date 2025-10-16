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
import time
from dataclasses import dataclass
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
        if len(candidates) >= config.prompt_count * config.batches * config.num_requests:
            break

    if not candidates:
        raise RuntimeError(
            f"No prompts found for scenario '{config.scenario}'. "
            "Consider lowering min/max char filters."
        )

    random.seed(config.prompt_shuffle_seed)
    random.shuffle(candidates)
    return candidates[: config.prompt_count * config.batches * config.num_requests]


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

    nwor_stats = engine.get_engine_context().scheduler_stats.nwor_stats

    return {
        "nwor_mode": nwor_mode,
        "batch_index": batch_index,
        "latency_s": duration,
        "nwor_stats": nwor_stats,
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

            for batch_idx in range(config.batches):
                start = batch_idx * config.num_requests
                end = start + config.num_requests
                batch_prompts = prompts[start:end]
                result = run_batch(
                    engine, batch_prompts, config, nwor_mode, batch_idx
                )
                result["scv_mode"] = scv_mode
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
        output_path=args.output,
    )


def main() -> None:
    config = parse_args()
    results = run_microbenchmark(config)

    with open(config.output_path, "w", encoding="utf-8") as f:
        json.dump({"config": config.__dict__, "results": results}, f, indent=2)

    print(f"Wrote benchmark output to {config.output_path}")


if __name__ == "__main__":
    main()
