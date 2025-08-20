# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example dataset and reward function for MCQ distractor generation.

This module shows how to plug a custom dataset and reward into Verl. The
`MCQDataset` yields question stems and correct answers. The
`compute_entropy_reward` function queries a scoring model to compute the
entropy of answer probabilities.
"""

from __future__ import annotations

import json
import math

import torch
from vllm import LLM, SamplingParams


class MCQDataset(torch.utils.data.Dataset):
    """Simple dataset for (stem, answer) pairs stored in JSON.

    Each JSON entry must have keys ``stem`` and ``answer``.
    """

    def __init__(self, path: str) -> None:
        self.samples: list[dict[str, str]] = json.load(open(path))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, str]:  # pragma: no cover - trivial
        item = self.samples[idx]
        return {"prompt": item["stem"], "answer": item["answer"]}


def _score_options(model: LLM, prompt: str, options: list[str]) -> list[float]:
    """Return log-likelihood of each option being the correct answer."""
    sampling = SamplingParams(prompt_logprobs=True, max_tokens=0)
    logps = []
    for opt in options:
        out = model.generate([prompt + opt], sampling)
        # vLLM returns a list of logprobs per token
        token_logps = out[0].prompt_logprobs
        logp = sum(p for probs in token_logps for p in probs.values())
        logps.append(logp)
    return logps


def compute_entropy_reward(
    data_source: dict[str, str],
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    scoring_model: str | None = None,
) -> float:
    """Reward: entropy of the scoring model's answer distribution.

    Args:
        data_source: contains ``prompt`` and ``distractors`` generated so far.
        solution_str: the latest distractor produced by the agent.
        ground_truth: the correct answer string.
        scoring_model: model name or endpoint for vLLM.
    """
    options = data_source.get("distractors", []) + [solution_str]
    prompt = data_source["prompt"]
    correct = ground_truth
    all_options = options + [correct]

    llm = LLM(model=scoring_model) if scoring_model else LLM()  # pragma: no cover
    logps = _score_options(llm, prompt, all_options)  # pragma: no cover

    # Convert log-likelihoods to probabilities and compute Shannon entropy.
    max_logp = max(logps)
    probs = [math.exp(lp - max_logp) for lp in logps]
    total = sum(probs)
    probs = [p / total for p in probs]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy
