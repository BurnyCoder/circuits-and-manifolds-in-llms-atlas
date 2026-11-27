# Learnings: Belief-state fractal (trained here)

- Comparing training-batch losses with a separately sampled entropy estimate made the model look "better than optimal". Both are now computed on the same held-out sequences.
- A strong random baseline is a finding in itself. It shows how much of a "representation" can come from the input alone.
