# Learnings: Othello-GPT

- **Relative vs absolute board.** Absolute (black/white) probes plateau around 75% while relative (mine/theirs) probes
  reach ~99%. This reproduces Nanda et al.'s reframing and explains why Li et al. needed non-linear probes for the
  absolute board.
- **Majority-class baseline.** Accuracy over all squares includes many empty squares, so a probe that knows nothing
  scores ~54%. We added occupied-square accuracy (`occ_acc_best`) and two controls (random-init model, shuffled labels)
  to make the baseline explicit.
- **Intervention strength.** A linear flip at one layer roughly halves the error. Li et al.'s gradient-based multi-layer
  edits do much better. We report the weaker effect as it is.
- **Engine correctness.** `tests/test_othello.py` checks flips, passes and the opening. The model's near-zero illegal
  top-1 rate on our games is an independent check that our engine matches the training distribution.
