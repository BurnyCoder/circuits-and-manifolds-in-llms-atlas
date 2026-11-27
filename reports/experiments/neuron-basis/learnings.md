# Learnings: Neurons as an interpretable basis (Goodfire)

- Read the method from the primary source. Goodfire zeroes; their band experiments mean-ablate; ericzhou12 mean-fills. The three rules give very different numbers.
- Caching reference log-probs over a 152k vocabulary needs ~7 GB in full mode and crashed the run. The reference is now recomputed per sequence.
