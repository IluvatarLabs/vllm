# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""

import multiprocessing
from vllm.logger import init_logger

logger = init_logger(__name__)

# Set the multiprocessing start method to 'spawn' for CUDA compatibility.
# This must be done before any CUDA context is created.
# NOTE: This may have side effects for users of vLLM as a library who
# also use multiprocessing and rely on the default 'fork' method.
# See: https://docs.vllm.ai/en/latest/dev/troubleshooting.html#python-multiprocessing
try:
    # The 'fork' start method is not compatible with CUDA.
    # We need to use 'spawn' to avoid deadlocks.
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # If the start method is already set, we check if it is 'spawn'.
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        logger.warning(
            "vLLM requires the 'spawn' start method for CUDA compatibility. "
            "The start method has already been set to '%s'. This may "
            "lead to deadlocks.", multiprocessing.get_start_method())

# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from .version import __version__, __version_tuple__  # isort:skip

import typing

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import vllm.env_override  # noqa: F401

MODULE_ATTRS = {
    "bc_linter_skip": "._bc_linter:bc_linter_skip",
    "bc_linter_include": "._bc_linter:bc_linter_include",
    "AsyncEngineArgs": ".engine.arg_utils:AsyncEngineArgs",
    "EngineArgs": ".engine.arg_utils:EngineArgs",
    "AsyncLLMEngine": ".engine.async_llm_engine:AsyncLLMEngine",
    "LLMEngine": ".engine.llm_engine:LLMEngine",
    "LLM": ".entrypoints.llm:LLM",
    "initialize_ray_cluster": ".executor.ray_utils:initialize_ray_cluster",
    "PromptType": ".inputs:PromptType",
    "TextPrompt": ".inputs:TextPrompt",
    "TokensPrompt": ".inputs:TokensPrompt",
    "ModelRegistry": ".model_executor.models:ModelRegistry",
    "SamplingParams": ".sampling_params:SamplingParams",
    "PoolingParams": ".pooling_params:PoolingParams",
    "ClassificationOutput": ".outputs:ClassificationOutput",
    "ClassificationRequestOutput": ".outputs:ClassificationRequestOutput",
    "CompletionOutput": ".outputs:CompletionOutput",
    "EmbeddingOutput": ".outputs:EmbeddingOutput",
    "EmbeddingRequestOutput": ".outputs:EmbeddingRequestOutput",
    "PoolingOutput": ".outputs:PoolingOutput",
    "PoolingRequestOutput": ".outputs:PoolingRequestOutput",
    "RequestOutput": ".outputs:RequestOutput",
    "ScoringOutput": ".outputs:ScoringOutput",
    "ScoringRequestOutput": ".outputs:ScoringRequestOutput",
}

if typing.TYPE_CHECKING:
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.llm_engine import LLMEngine
    from vllm.entrypoints.llm import LLM
    from vllm.executor.ray_utils import initialize_ray_cluster
    from vllm.inputs import PromptType, TextPrompt, TokensPrompt
    from vllm.model_executor.models import ModelRegistry
    from vllm.outputs import (ClassificationOutput,
                              ClassificationRequestOutput, CompletionOutput,
                              EmbeddingOutput, EmbeddingRequestOutput,
                              PoolingOutput, PoolingRequestOutput,
                              RequestOutput, ScoringOutput,
                              ScoringRequestOutput)
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams

    from ._bc_linter import bc_linter_include, bc_linter_skip
else:

    def __getattr__(name: str) -> typing.Any:
        from importlib import import_module

        if name in MODULE_ATTRS:
            module_name, attr_name = MODULE_ATTRS[name].split(":")
            module = import_module(module_name, __package__)
            return getattr(module, attr_name)
        else:
            raise AttributeError(
                f'module {__package__} has no attribute {name}')


__all__ = [
    "__version__",
    "bc_linter_skip",
    "bc_linter_include",
    "__version_tuple__",
    "LLM",
    "ModelRegistry",
    "PromptType",
    "TextPrompt",
    "TokensPrompt",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "PoolingOutput",
    "PoolingRequestOutput",
    "EmbeddingOutput",
    "EmbeddingRequestOutput",
    "ClassificationOutput",
    "ClassificationRequestOutput",
    "ScoringOutput",
    "ScoringRequestOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
    "PoolingParams",
]
