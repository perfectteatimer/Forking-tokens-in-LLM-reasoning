# Forking Tokens in LLM Reasoning

**English** | [Русский](README.ru.md)

A research project on finding branching points in the reasoning of large language models (LLMs).

**Research supervisor:** Gleb Rodionov.

## Project plan

### Checkpoint 1. Literature review

We read 7 papers on branching points, reinforcement learning, and how to interpret model behavior.

- [Targeting Pivotal Decisions for Credit Assignment in Agentic Reinforcement Learning](https://arxiv.org/abs/2609.36178)
- [Rethinking RL for LLM Reasoning: It's Sparse Policy Selection, Not Capability Learning](https://arxiv.org/abs/2605.06241)
- [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- [TreePO](https://arxiv.org/abs/2508.17445)
- [Forking Paths in Neural Text Generation](https://arxiv.org/abs/2412.07961)
- [Forking Fast: Efficiently Estimating Uncertainty Dynamics in Text Generation](https://arxiv.org/abs/2608.19611)
- [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2506.01939)

### Checkpoint 2. Resampling every 30 tokens

We analyzed 4 reasoning traces from Qwen3.5-9B on 3 math problems. We call each original trace a *trunk*. Every 30 tokens, the model generated 100 continuations from the same trunk prefix. This gave us 57,500 continuations from 575 prefixes.

### Checkpoint 3. Changes in the answer distribution

We plotted the answers at each prefix and read the trunk around the largest changes. Each plot counts the last boxed answer in the prefix and continuation together, including runs that hit the length limit.

| Task | Trunk seed | What changed | Plot |
| --- | --- | --- | --- |
| 4 | 2 | From 1140 to 1350 tokens, the trunk counted the size of the range and listed primes through 101. The share with the correct answer, 70, rose gradually from 62% to 99%. | [View plot](figures/task4-seed2-distribution.png) |
| 7 | 2 | From 480 to 510 tokens, the trunk finished listing the possible cycle types of a permutation but left out `3+1+1+1`. The share with the correct answer, 396, fell from 65% to 14%. | [View plot](figures/task7-seed2-distribution.png) |
| 7 | 3 | From 810 to 840 tokens, the trunk started checking the list again. The share with 396 rose from 57% to 97% before the missing case appeared in the trunk. | [View plot](figures/task7-seed3-distribution.png) |
| 11 | 2 | From 2520 to 2550 tokens, the trunk started checking whether its grid arrangement was possible. The share with the correct answer, 896, rose from 62% to 88%. Only 25 and 24 runs, respectively, finished with 896. | [View plot](figures/task11-seed2-distribution.png) |

### Interpretability experiments

We ranked 571 windows of 30 tokens each. In 6 windows, the answer distribution changed enough to pass a permutation test with a 5% false discovery rate.

- **A1, token surprisal.** We added up the negative log probabilities of the 30 tokens the model actually chose, using their probabilities after sampling filters. None of the 113 windows with zero surprisal contained a marked fork.
- **A2+B1, logits and clusters.** At each prefix, we took the next 30 tokens of the trunk and up to 30 tokens from each of its 100 continuations. We grouped them by similarity using MiniLM embeddings. We added three features to A2. They measure how rare the trunk's cluster is, how evenly the continuations split across clusters, and the mean distance between continuation embeddings. We trained a logistic regression on three trunks and tested it on the fourth, repeating this for each trunk.

Reading windows in score order, we needed **16.6%** of the windows with A1 and **15.9%** with A2+B1 to find every marked fork, on average across the three trunks that had forks. A2 alone needed 17.5%. With random ordering, the expected share was 64.7%.

To reproduce the window scores from the saved features and labels, run the following from the repository root.

```sh
cd interpretability
python -m pip install numpy pandas pyarrow scipy scikit-learn
python surprisal.py windows.parquet --output a1.csv
python cluster_score.py windows.parquet --output a2_b1.csv
```

### Checkpoint 4. ML service

- Build a simple ML service with FastAPI.
- Implement inference with the selected model through an API.
- Choose the exact task for the service based on the project results at this stage.

### Interim defense

- Present the work completed and initial results.
- Discuss plans for the second semester.

### Checkpoint 5. Further experiments

- Improve existing approaches or test new ideas.
- Run additional experiments.
- Compare results and discuss further work.

### Checkpoint 6. Results analysis

- Continue work on the selected research direction.
- Run additional checks and experiments.
- Analyze the results and possible reasons for the models' behavior.

### Checkpoint 7. Project completion

- Run the final experiments.
- Collect and organize the results.
- Prepare for the final defense.

### Final defense

- Present the project results and main conclusions.

This plan is preliminary and may change during the project based on experiment results and discussions.

## Copyright

Copyright (c) 2026 perfectteatimer. All rights reserved.

See [LICENSE](LICENSE) for terms.
