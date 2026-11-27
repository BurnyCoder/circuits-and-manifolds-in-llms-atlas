# Learnings: IOI

- **Single-token filtering matters.** Names and places that split into several GPT-2 tokens shift positions within a
  template and break position-wise mean-ablation. We filter the vocabulary lists to single tokens first.
- **Mean-ablation must be per template and per position.** A global mean over all p_ABC prompts mixes positions with
  different roles (S1, IO, S2, END) and inflates knockout effects.
- **Faithfulness depends on the ablation choice.** Miller et al. 2024 show that circuit faithfulness numbers move a lot
  with the ablation method and dataset. We report one setting (p_ABC mean, per template) and treat the 87% (paper) vs
  93% (ours) difference as unremarkable.
- **Speed.** 128 class-combination knockouts × 320 prompts take seconds on GPU. Hooks are built per combination with
  closures that bind their arguments as defaults, which avoids Python's late-binding lambda pitfall.
