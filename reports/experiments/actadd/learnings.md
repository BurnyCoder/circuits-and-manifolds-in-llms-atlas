# Learnings: Activation addition (ActAdd)

- In fp32, GPT-2 XL filled the 8 GB GPU and spilled to shared memory (54-minute smoke test). bfloat16, batched sampling and a KV cache made the full run take minutes. The cached steering was verified token-identical.
