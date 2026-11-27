# Learnings: Steering along manifolds

- Check task accuracy first. The pre-stated prompt was answered correctly too rarely, and both steering paths collapsed to the same output.
- An endpoint check (does writing day j's centroid make the model say j?) shows where steering can work at all. Below layer 22 nothing worked, even replacing the whole residual.
