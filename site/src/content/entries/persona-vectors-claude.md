---
slug: persona-vectors-claude
title: Persona vectors
category: steering
collection: anthropic-tc
models: [Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct]
fidelity: redrawn
evidence: causal
order: 430
summary: Directions for traits like sycophancy or "evil" can be extracted automatically from a description, then used to monitor, steer and even vaccinate a model during fine-tuning.
viz: persona-vectors
tags: [persona, traits, monitoring, preventative steering, data filtering]
related: [emotions, refusal-direction, actadd]
citations:
  - id: chen2025
    title: "Persona Vectors: Monitoring and Controlling Character Traits in Language Models"
    authors: Chen, Arditi, Sleight, Evans, Lindsey
    year: 2025
    url: https://arxiv.org/abs/2507.21509
  - id: pv-blog
    title: Persona vectors (blog post)
    authors: Anthropic
    year: 2025
    url: https://www.anthropic.com/research/persona-vectors
  - id: pv-code
    title: persona_vectors (code, Apache-2.0)
    authors: Chen et al.
    year: 2025
    url: https://github.com/safety-research/persona_vectors
  - id: assistant-axis
    title: The Assistant Axis
    authors: Lu et al.
    year: 2026
    url: https://arxiv.org/abs/2601.10387
replications:
  - who: LessWrong — do persona vectors detect implicitly elicited traits?
    url: https://www.lesswrong.com/posts/aCxCkzPoRKHxkqsML/do-persona-vectors-detect-implicitly-elicited-traits-1
    model: Qwen2.5-7B/32B, Qwen3-32B
    outcome: partial
    note: Extends the method to larger models; "evil" resists elicitation by situation alone.
---

## In plain words

Give the pipeline a trait name and a one-line description (for "evil", roughly: actively seeking to harm, manipulate, and cause suffering). An LLM writes system prompts
that encourage and discourage the trait, the model answers questions under both, and the difference in its average
activations becomes a "persona vector". The vector lights up before the model starts behaving that way (monitoring),
adding it produces the trait (steering), and — counter-intuitively — adding it *during* fine-tuning stops the fine-tuning
data from pushing the model toward that trait ("preventative steering"). It can also flag training samples likely to
induce the trait.

## How it works (technical)

Difference of mean residual activations over response tokens (trait − anti-trait) per layer; the steering layer is chosen
by effect. Projection of the last prompt token onto the vector predicts trait expression (r = 0.75–0.83); fine-tuning
shifts along the vector correlate with trait changes (r = 0.76–0.97, vs r = 0.34–0.86 for cross-trait baselines); preventative steering
keeps coherence above 80 and MMLU better than inference-time steering. Models: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct.

## Limitations

- We redraw the published pipeline and numbers; the official evaluation needs an external LLM judge, so we did not rerun
  it. Related open entries: emotion vectors and the refusal direction (both recomputed).
