# Learnings: Can SAEs capture neural geometry? (synthetic benchmark)

- The first version showed each manifold alone (TopK then always fires k atoms) and used 16 instances (co-occurring manifolds crossed the 10% support threshold). It reported 0% capture at every k and still printed "pass" (argmax of zeros). Re-reading the paper's appendix fixed all three problems.
- We had pre-stated the regime direction backwards. The amendment records it.
