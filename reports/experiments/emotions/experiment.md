# Experiment: Emotion vectors (open model)

Code [`src/atlas/entries/emotions.py`](../../../src/atlas/entries/emotions.py) (the docstring gives the full method) · site
data `site/public/data/emotions.json` · page `#/entry/emotions` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

Do emotion vectors in an open model organize like human emotions, react to situations, and causally shift preferences (Sofroniew et al. 2026, [Transformer Circuits](https://transformer-circuits.pub/2026/emotions/index.html))?

## Setup

Qwen3-1.7B wrote short stories for 71 emotions × topics (emotion word excluded) and neutral texts. Emotion vectors are mean layer-18 residuals over each emotion's stories, minus the overall mean, with neutral-text PCs projected out. Analyses: PCA vs human valence/arousal norms (Warriner et al.) with a shuffled-label control; a medicine-dose sweep; a preference test adding blissful/hostile vectors to one option of a forced choice, with a random-vector control.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| The main axis of emotion space is valence | PC1 26% of variance; r = 0.81 with human valence (Claude Sonnet 4.5) | PC1 48% of variance; \|r\| = 0.94 with human valence (shuffled-label 95th pct 0.12); 71 emotions, 1651 stories | \|r\| ≥ 0.6 and above the shuffled-label 95th percentile | ✅ pass |
| The second axis is arousal | PC2 15%; r = 0.66 with human arousal | PC2 9% of variance; \|r\| = 0.06 with human arousal | \|r\| ≥ 0.4 | ❌ fail |
| Emotion vectors track danger in the user's situation | 'afraid' rises and 'calm' falls with the Tylenol dose (Fig. 3) | slope per decade of dose: afraid +4.74, calm -7.28 | afraid slope > 0 and calm slope < 0 | ✅ pass |
| Steering with emotion vectors shifts preferences | blissful steering on an option raises its Elo (+212), hostile lowers it (−303) | P(choose A): baseline 0.54, +blissful on A 0.55, +hostile on A 0.54, random vector 0.53 | blissful > baseline > hostile, each by ≥ 0.05, with \|random − baseline\| < 0.05 | ❌ fail |
<!-- /table:replication -->

## Interpretation

Valence replicates strongly: the first principal axis is pleasant ↔ unpleasant, with a correlation above the paper's
and far above the shuffled-label null. The dose-sweep pass carries little weight: it is the sign of two slopes over 7
single prompts with no control prompt, and because "afraid" and "calm" sit at opposite ends of the valence axis their
slopes are strongly anti-correlated, so a joint pass by chance is closer to 50% than 25%. The second axis does not
correspond to arousal, and steering an option with blissful or hostile vectors does not change which option the model
picks. One hypothesis (not tested) is that a 1.7B model's emotion space is dominated by valence; PC2 still holds
noticeable variance. Details: [methods review](../../reviews/methods-review.md), M12.

## Limitations

- Self-generated, stereotyped stories; one model; one layer.
- The preference test is a forced A/B choice, not the paper's Elo tournament.
- To do: dose sweep with a matched benign control ("{d} ml of water"), paraphrases per dose and a permutation test.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `n_emotions` | 71 |
| `n_stories` | 1651 |
| `var_pc1` | 0.47746 |
| `var_pc2` | 0.088475 |
| `r_valence` | 0.943759 |
| `r_arousal` | 0.063154 |
| `null_valence_p95` | 0.118639 |
| `afraid_slope` | 4.736686 |
| `calm_slope` | -7.275177 |
| `pref_base` | 0.541102 |
| `pref_blissful` | 0.545317 |
| `pref_hostile` | 0.543841 |
| `pref_random` | 0.529033 |
| `layer` | 18 |
<!-- /table:summary -->
