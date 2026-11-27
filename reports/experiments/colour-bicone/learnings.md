# Learnings: The colour bicone in an image autoencoder

- The official FLUX repositories are gated by click-through terms on the user's account. The ungated diffusers copy avoided accepting terms on the user's behalf.
- With HF_HOME moved to another disk the pipeline ran unauthenticated. Setting HF_TOKEN_PATH fixed this without copying the token.
