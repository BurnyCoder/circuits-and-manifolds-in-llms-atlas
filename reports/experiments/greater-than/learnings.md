# Learnings: greater-than

- **Early MLPs belong to the embedding pathway.** Patching MLPs 0–7 with "01" activations erased the start year itself
  and gave PD −53%: a method bug, not a finding. Circuit-only now patches the 137 non-circuit heads only (global
  learnings file).
- **Memory spill.** The first full run stalled because all 1,164 prompts × every patch combination were processed at
  once and the Windows driver spilled VRAM to shared memory. Chunking into 128-prompt accumulators fixed it (99 s total).
- **PD higher than the paper.** Our 12 nouns differ from the paper's, and PD is sensitive to the noun (plausible
  durations differ for "war" vs "dynasty"). The criterion only requires PD > 50%.
