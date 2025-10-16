import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import torch

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


@dataclass
class SpecScenario:
    name: str
    num_requests: int
    draft_tokens: int
    acceptance_ratio: float


def generate_dummy_prompt(num_tokens: int) -> list[int]:
    return [1] * num_tokens


def target_output_length(draft_tokens: int, acceptance_ratio: float) -> int:
    accepted = int(draft_tokens * acceptance_ratio)
    # +1 for bonus token
    return max(1, accepted + 1)


def run_iteration(
    engine: AsyncLLMEngine,
    scenario: SpecScenario,
    nwor_mode: str,
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, Any]:
    prompts = [
        generate_dummy_prompt(64) for _ in range(scenario.num_requests)
    ]
    sampling_params = SamplingParams(
        max_tokens=target_output_length(
            scenario.draft_tokens, scenario.acceptance_ratio
        )
    )

    # Warmup
    for _ in range(warmup_steps):
        futures = [
            engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=f"{scenario.name}-warmup-{i}",
            )
            for i, prompt in enumerate(prompts)
        ]
        for future in futures:
            future.result()

    # Measurement
    latencies = []
    for step in range(measure_steps):
        start = time.time()
        futures = [
            engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=f"{scenario.name}-{step}-{i}",
            )
            for i, prompt in enumerate(prompts)
        ]
        for future in futures:
            future.result()
        latencies.append(time.time() - start)

    return {
        "scenario": scenario.name,
        "nwor_mode": nwor_mode,
        "num_requests": scenario.num_requests,
        "draft_tokens": scenario.draft_tokens,
        "acceptance_ratio_estimate": scenario.acceptance_ratio,
        "latency_seconds": latencies,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=str, default="nwor_microbench.json")
    args = parser.parse_args()

    scenarios = [
        SpecScenario("accept_all", num_requests=8, draft_tokens=4, acceptance_ratio=1.0),
        SpecScenario("medium", num_requests=8, draft_tokens=4, acceptance_ratio=0.5),
        SpecScenario("low", num_requests=8, draft_tokens=4, acceptance_ratio=0.25),
    ]

    results: list[dict[str, Any]] = []
    for nwor_mode in ("off", "stage"):
        os.environ["VLLM_NWOR_MODE"] = nwor_mode

        engine_args = AsyncEngineArgs(
            model=args.model,
            target_device=args.device,
            tensor_parallel_size=1,
            speculative_config=None,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        for scenario in scenarios:
            result = run_iteration(
                engine,
                scenario,
                nwor_mode=nwor_mode,
                warmup_steps=args.warmup,
                measure_steps=args.steps,
            )
            results.append(result)

        engine.shutdown()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
