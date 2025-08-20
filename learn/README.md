# Training LLM distractor generator with Verl

This directory provides guidance for using [Verl](https://github.com/volcengine/verl) to train an LLM that produces high-quality distractors for multiple-choice questions (MCQs).

## Overview
1. **Generator LLM (agent)** – produces distractor options given a question stem and the correct answer.
2. **Scoring LLM (environment reward)** – given the completed MCQ (stem, correct answer, generated distractors), it computes the probability of each option being correct. The reward is the Shannon entropy of this distribution, encouraging high uncertainty.
3. **Algorithm** – use PPO (already implemented in Verl) to optimise the generator.

The rest of this document outlines the steps to implement a custom dataset and reward function, and to run PPO training.

## 1. Prepare the dataset
Create a dataset returning `{"prompt": <stem>, "answer": <correct_answer>}` for each question.

```python
# learn/distractor_reward.py contains the full example
class MCQDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.samples = json.load(open(path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "prompt": item["stem"],
            "answer": item["answer"],
        }
```

Reference this dataset from the config:

```yaml
# config snippet
train_dataset:
  path: learn/distractor_reward.py
  name: MCQDataset
  kwargs:
    path: data/mcq_train.json
```

## 2. Implement the entropy-based reward
`learn/distractor_reward.py` contains an example function `compute_entropy_reward` that:
1. Assembles the full MCQ using the stem, correct answer and generated distractors.
2. Queries a scoring LLM (served via [vLLM](https://github.com/vllm-project/vllm)) for the log-likelihood of each option.
3. Converts log-likelihoods to a probability distribution and returns its Shannon entropy.

In the Verl config, point to this function:

```yaml
custom_reward_function:
  path: learn/distractor_reward.py
  name: compute_entropy_reward
  kwargs:
    scoring_model: <model-name-or-endpoint>
```

## 3. Launch PPO training
Use one of the example training scripts as a base (e.g. `examples/gemma/run_gemma.sh`) and override dataset and reward paths:

```bash
bash run_gemma.sh \
  trainer.n_gpus_per_node=1 \
  data.train_dataset.path=learn/distractor_reward.py \
  data.custom_reward_function.path=learn/distractor_reward.py \
  # plus usual PPO hyperparameters
```

Training will optimise the generator LLM to maximise entropy of the scoring model's answer distribution, yielding strong distractors.

## 4. Extending
You can incorporate additional signals, such as correctness checks or style constraints, by editing the reward function.

