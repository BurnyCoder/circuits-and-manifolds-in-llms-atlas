# Learnings: copy suppression

- **Effective embedding.** The paper's MLP0(W_E) (embedding plus the first MLP's output; GPT-2 uses MLP0 as an extended
  embedding) is what the OV/QK tests are defined on. We followed it rather than raw W_E.
- **Row vs column ranking.** The paper ranks each column of the OV matrix; we rank rows. Both are valid views. We made
  the criterion relative to the other late heads so that the verdict does not depend on this choice.
- **Partial behavioural effect.** Zero-ablating L10H7 raised the copied token's probability on only 5/8 prompts, even
  though its direct effect is always negative. This is consistent with compensation by other heads (11.10 has the
  next-highest OV score among layer-10/11 heads). It is an honest partial result inside a passed claim, whose criterion
  was the direct effect.
