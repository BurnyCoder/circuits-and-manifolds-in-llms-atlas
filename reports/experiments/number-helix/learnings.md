# Learnings: The number helix (trigonometric addition)

- Pythia ≤ 410M solved only a handful of the few-shot additions. Tokenization was checked and is not the cause. The minimum-solved-prompts rule prevents a meaningless verdict on n = 1.
- Patching the a-token position needs an exact token index. The prefix tokenization length gives it, which is checked by an assertion.
