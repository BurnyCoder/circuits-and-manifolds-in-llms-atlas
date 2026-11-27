# State-of-the-art review — Circuits & Manifolds Atlas

**Date:** 2026-11-27 · **Project:** Circuits & Manifolds Atlas (author: Libor "Burny" Burian) · **Companion:** [literature-review.md](literature-review.md)

**Scope.** (1) Existing interpretability sites and catalogues, and what a cross-paper atlas adds; (2) the current tools (TransformerLens, SAELens, nnsight, circuit-tracer, Neuronpedia, jacobian-lens and others), with versions, licences and compatibility; (3) open models with released SAEs, transcoders, CLTs and lenses; (4) best current methods per mechanism type; (5) open problems.

**How this was verified.** Every site, repository, package and paper below was loaded on 2026-11-27 (web pages, GitHub API, PyPI JSON API, Hugging Face API, arXiv abstract/HTML). Package versions and dependency pins for TransformerLens, SAELens, nnsight, circuit-tracer and the Neuronpedia client were re-checked directly against the PyPI JSON API by the author of this file. Status markers: ✅ verified from the source; ◐ partially verified (e.g. abstract or listing only, or JS-rendered page); ⚠ not verified (search snippet only, or page would not load). Licences: "no licence" means no LICENSE file and none detected by GitHub; "no tag" means no licence field in Hugging Face metadata.

## 1. Prior-art sites and what a cross-paper atlas adds

Star counts and last-push dates were read from the GitHub API on 2026-11-27.

| Site / tool | What it does | Models | Licence | Gap relative to this atlas |
|---|---|---|---|---|
| [Distill Circuits thread](https://distill.pub/2020/circuits/) ✅ | Nine interactive articles reverse-engineering one vision CNN: Zoom In, Early Vision overview, Curve Detectors, Naturally Occurring Equivariance, High-Low Frequency Detectors, Curve Circuits, Visualizing Weights, Branch Specialization, Weight Banding. Frozen archive; transformer-circuits.pub dates it Mar 2020 – Apr 2021. | InceptionV1 | CC-BY 4.0 (text and diagrams) | One vision model, no LMs, no updates since 2021, no replication tracking. |
| [OpenAI Microscope](https://microscope.openai.com) ✅ (live) / ◐ (contents) | Feature visualisations of every significant layer and neuron of a set of vision "model organisms" ([announcement, 14 Apr 2020](https://openai.com/index/microscope/); the announcement says eight in one place and nine in another). Still served (HTTP 200, `Last-Modified` 2026-11-07); no shutdown notice found. JS app, so model list not read; code references `inceptionv1`, `resnetv2_152_slim`. | Vision models only | Images CC-BY 4.0 | One method (feature visualisation), vision only. |
| [Neuronpedia](https://www.neuronpedia.org) ✅ ([repo](https://github.com/hijohnnylin/neuronpedia), last push 2026-11-26, 1,236★) | Hosted platform to browse, search and steer latents of 43 models (GPT-2 small, Pythia-70M-deduped, Qwen3-1.7B, Gemma-2/3/4, Llama 3.x, Qwen3/3.5/3.6, OLMo-3, GPT-OSS-20B, …); hosts Gemma Scope / Gemma Scope 2 / Llama Scope / Qwen3 SAEs and transcoders; attribution graphs via circuit-tracer; steering; API with Python and TypeScript clients; also Jacobian Lens, Natural Language Autoencoder and Assistant Axis views; search over > 50M latents. | Many open models | Apache-2.0 (repo also has a LICENSE-MIT file); the separate `decoderesearch/interp-engine` licence not checked | Organised by latent, not by paper finding; no circuit/manifold catalogue; no replication status. |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) ✅ (last push 2026-06-06, 9,224★) | Runs GPT-2 small (124M, ONNX) in the browser; shows embeddings, attention, next-token probabilities, sampling knobs. Georgia Tech Polo Club (Cho, Kim, …, Chau), CHI 2026. | GPT-2 small | MIT | Teaches the forward pass; contains no interpretability findings. |
| [bbycroft LLM visualization](https://bbycroft.net/llm) ✅ ([repo](https://github.com/bbycroft/llm-viz), last push 2026-08-10, 5,604★) | 3D animated walkthrough of a GPT; live weights are a tiny minGPT that sorts A/B/C; renderer can draw GPT-2 small topology. | Toy minGPT | MIT (except homepage and third-party assets) | Architecture only, no learned features or circuits. |
| [CircuitsVis](https://github.com/TransformerLensOrg/CircuitsVis) ✅ (last push 2026-04-30, 377★) | Python/React widgets: attention patterns, token activations, top-k samples/tokens, logits. Cooney & Nanda. | Any (library) | MIT | Widget library, not a collection of findings. Reusable by the atlas. |
| [PAIR, *Mapping LLMs with Sparse Autoencoders*](https://pair.withgoogle.com/explorables/sae/) ✅ (Oct 2025) | Interactive essay on SAEs: top activations, labelling walkthrough, steering buttons, searchable UMAP of 16,384 Gemma Scope features, embedded Neuronpedia steering. Hussein, Raval, Reif, … Nanda, Dixon, Thain. | Gemma-2 (one Gemma Scope SAE) | not found | One method, one SAE. |
| [Goodfire Neuron Atlas](https://www.goodfire.com/research/public_neuron_atlas/all_labels) ✅ (10 Nov 2026) | Interactive semantic map of automated labels for all 10.5M "neuron poles" of Qwen3.6-35B-A3B plus a per-layer MoE-expert view; labels generated by a fine-tuned Qwen3-8B (see the companion report [Neurons as an Interpretable Basis](https://www.goodfire.com/research/neurons-as-an-interpretable-basis)). Tigges, Kowal, McGrath, Lewis, Cammarata. Related: Shilov et al., *Large-scale Neuron Community Discovery Reveals Neuron-Sparse Behaviors in Language Models*, arXiv [2611.23112](https://arxiv.org/abs/2611.23112) (Goodfire blog title *Discovering Neuron Communities*, 17 Nov 2026; explorer at goodfire.com/demos/neuron-communities; ◐ title checked via the arXiv API, paper not read). | Qwen3.6-35B-A3B | not found | One model, one unit type (neurons). |
| Awesome lists ✅ | [JShollaj/awesome-llm-interpretability](https://github.com/JShollaj/awesome-llm-interpretability) (1,662★, no licence, active); [ruizheliUOA/Awesome-Interpretability-in-LLMs](https://github.com/ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models) (415★, CC0-1.0, stale since 2024-11); [Dakingrai/awesome-mechanistic-interpretability-lm-papers](https://github.com/Dakingrai/awesome-mechanistic-interpretability-lm-papers) (278★, follows the survey [Rai et al. 2024, 2407.02646](https://arxiv.org/abs/2407.02646)); [AI-in-Transportation-Lab/awesome-mechanistic-interpretability](https://github.com/AI-in-Transportation-Lab/awesome-mechanistic-interpretability) (180★, ~1,388 links, active); Neel Nanda's [Extremely Opinionated Annotated List v2](https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite) (AF, 7 Jul 2024). | — | various | Link lists: no structured findings, no recomputation, no replication status. |

**Existing catalogues and benchmarks (closest neighbours)**

| Resource | What it does | How the atlas differs |
|---|---|---|
| [transformer-circuits.pub](https://transformer-circuits.pub) index + Circuits Updates ✅ | Anthropic's dated index from the Distill thread (2020) to *Circuits Updates — October 2026* (which covers the J++ lens). | One lab, date-ordered, mostly closed models; the atlas is organised by mechanism and adds open-model recomputation. |
| [The Circuits Research Landscape](https://www.neuronpedia.org/graph/info) ✅ (Aug 2025) | Multi-lab report (Anthropic, Decode, EleutherAI, Goodfire, GDM) replicating attribution-graph case studies on Gemma-2-2B, Llama-3.2-1B and GPT-2; > 7,000 graphs generated on Neuronpedia by then. | One method, one report. |
| [MIB](https://mib-bench.github.io) ✅ (Mueller et al., ICML 2025, [2504.13151](https://arxiv.org/abs/2504.13151); [leaderboard](https://huggingface.co/spaces/mib-bench/leaderboard)) | Benchmark with circuit-localisation and causal-variable-localisation tracks on IOI, MCQA, arithmetic and ARC over GPT-2 small, Qwen-2.5-0.5B, Gemma-2-2B, Llama-3.1-8B and an InterpBench model. | Ranks methods; does not catalogue findings. Its tasks and models are a natural backbone for atlas recomputation. |
| [InterpBench](https://arxiv.org/abs/2407.14494) ✅ (NeurIPS 2024 D&B) | Semi-synthetic transformers trained with SIIT so that their circuits are known (Tracr-derived and IOI). | Synthetic ground truth vs. natural models. |
| [Tracr](https://arxiv.org/abs/2301.05062) ✅ ([repo](https://github.com/google-deepmind/tracr), Apache-2.0, last push 2024-02) | Compiles RASP programs into transformer weights. | Compiler for test models, not a catalogue. |
| ["Zoo of Known Circuits"](https://ttsugriy.github.io/mechinterp-first-principles/zoo-of-circuits.html) ✅ (ch. 24 of *First Principles of Mechanistic Interpretability*, created Jan 2026, no licence) | 13 prose entries: induction, previous-token and duplicate-token heads, IOI, greater-than, factual recall, copy suppression, grokking, deception/sycophancy/refusal features, … | Closest existing analogue, but prose only: no recomputation, no interactivity, no replication tracking, no geometry entries. |

Not found despite searching: anything named "Circuit Atlas", a "LessWrong circuits wiki", or an "interpretability findings database".

**What a cross-paper atlas adds.** No resource we found combines (i) findings across papers and labs organised by mechanism type (circuit, manifold, neuron, SAE feature, steering direction, attribution graph, lens), (ii) results recomputed on small open models with code, (iii) faithfully redrawn closed-model figures clearly labelled as such, (iv) geometry/manifold entries next to circuit entries, and (v) per-finding tracking of open-model replications with a succeeded / partial / failed verdict. Neuronpedia is the natural data and tooling partner (latents, graphs, API), MIB the natural source of standard tasks, and CircuitsVis a reusable widget layer.

## 2. Tools: current state (versions and pins checked on PyPI, 2026-11-27)

| Tool | Latest | Licence | Role in the atlas |
|---|---|---|---|
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) ([PyPI](https://pypi.org/project/transformer-lens/)) | 4.4.1 (2026-11-25); last 3.x is 3.9.0 (2026-09-12) | MIT | hooks, activation caching, patching on GPT-2 / Pythia / Qwen3 / Gemma |
| [SAELens](https://github.com/decoderesearch/SAELens) ([PyPI](https://pypi.org/project/sae-lens/)) | 6.56.4 (2026-11-26) | MIT | loading pretrained SAEs/transcoders; training SAEs |
| [nnsight](https://github.com/ndif-team/nnsight) ([PyPI](https://pypi.org/project/nnsight/)) | 0.8.4 (2026-11-23) | MIT | interventions on any HF/PyTorch model; remote execution on NDIF |
| [circuit-tracer](https://github.com/decoderesearch/circuit-tracer) ([PyPI](https://pypi.org/project/circuit-tracer/)) | 0.5.5 (2026-11-25) | MIT | attribution graphs from transcoders/CLTs |
| [Neuronpedia](https://github.com/hijohnnylin/neuronpedia) ([client on PyPI](https://pypi.org/project/neuronpedia/)) | client 1.2.1 (2026-08-08) | Apache-2.0 (repo also has LICENSE-MIT) | hosted latents, graphs, steering, lens API |
| [jacobian-lens](https://github.com/anthropics/jacobian-lens) | GitHub only; **not on PyPI** (`jacobian-lens` → 404) | Apache-2.0 | J-lens reference implementation (`jlens` import name) |

#### 2.1 TransformerLens 3.x vs 4.x and TransformerBridge ✅
- **TransformerBridge** wraps the native Hugging Face model through a per-architecture adapter instead of re-implementing it ([docs](https://transformerlensorg.github.io/TransformerLens/)). Loaders: `TransformerBridge.boot_transformers(name)` (HF models), `boot_native(TransformerBridgeConfig)` (toy/from-scratch), `boot_tl_legacy(name)` (old TL-format repos such as the attn-only models). It loads **raw HF weights by default** (like the old `from_pretrained_no_processing`); `enable_compatibility_mode()` applies HookedTransformer-style processing (LayerNorm folding, weight centring, value-bias folding) in place, irreversibly, and not for post-norm models (OLMo-2) or MLA blocks. Canonical hook names are `hook_in`/`hook_out`, with aliases for legacy names such as `hook_resid_pre`. Adapters exist for GPT-2, GPT-NeoX (Pythia), Qwen3, Gemma-2 and Gemma-3.
- **3.x** (3.0.0 on 2026-04-17 → 3.9.0 on 2026-09-12): Bridge introduced as the recommended path; `HookedTransformer` deprecated but present. 3.9.0 adds a `jacobian_lens` module and node attribution patching on the Bridge.
- **4.x** (4.0.0 on 2026-09-21 → 4.4.1): **`HookedTransformer` and the other `Hooked*` classes are removed**; the [v4 migration guide](https://github.com/TransformerLensOrg/TransformerLens/blob/main/docs/source/content/migrating_to_v4.md) says bug fixes for `HookedTransformer` continue only on the 3.x branch. `HookedRootModule` and `HookPoint` remain. 4.4.1 requires `transformers>=5.9.0,<5.24`.
- **Compatibility consequence (verified from PyPI metadata):** circuit-tracer 0.5.5 requires `transformer-lens>=2.16.0,<4` and `nnsight>=0.8.0rc1,<0.9`; SAELens 6.56.4 requires `transformer-lens>=3.0.0,<5.0.0`. **TransformerLens 3.9.0 is the only line that satisfies both** (it needs Python ≥ 3.10 and `transformers>=5.9.0`, compatible with circuit-tracer's `transformers>=4.56.0`). Alternatives: run circuit-tracer with its nnsight backend, or keep circuit-tracer in a separate environment from TL 4.x code.

#### 2.2 SAELens ✅
- Trainable SAE types (each with a `*TrainingSAEConfig`): Standard, Gated, TopK (labelled legacy); JumpReLU, BatchTopK, MatryoshkaBatchTopK (labelled state of the art); AbsTopK, Matching Pursuit (research). Transcoder classes `Transcoder`, `SkipTranscoder`, `JumpReLUTranscoder`, `JumpReLUSkipTranscoder` exist in the source; CLTs load only "as per-layer" (e.g. `mntss-gemma-2-2b-2.5m-clt-as-per-layer`).
- Pretrained registry ([docs](https://decoderesearch.github.io/SAELens/latest/pretrained_saes/)): gpt2-small 18 releases / 159 SAEs (incl. `gpt2-small-res-jb`, OpenAI v5 32k/128k); pythia-70m-deduped 7 / 207; gemma-2-2b 20 / 1,732 (Gemma Scope res/mlp/att/transcoders, Matryoshka, SAEBench, temporal); Gemma-3 270M–27B; Qwen3 0.6B–30B-A3B; Llama-3.1-8B 7 releases; **qwen3-1.7b: one release, `mwhanna-qwen3-1.7b-transcoders-lowl0`** (hooks `blocks.{0–27}.mlp.hook_in`, d_sae 163,840).
- Integrations: `SAETransformerBridge` (the docs say to use it because TL 4 removed `HookedTransformer`), `HookedSAETransformer`, nnsight, or framework-free `encode()`/`decode()`.

#### 2.3 nnsight and NDIF ✅ (NDIF access policy ◐)
- `with model.trace(...)` runs Python against intermediate values alongside the real forward pass; `tracer.invoke` batches prompts; `.save()` keeps values; wrappers for HF (`TransformersModel`), any `nn.Module`, vLLM, diffusers; `remote=True` sends the trace to [NDIF](https://ndif.us).
- NDIF live status (api.ndif.us/status, 2026-11-27): loaded ("HOT") models included gpt2, pythia-70m, pythia-160m, pythia-2.8b, gpt-j-6b, Qwen3-1.7B-Base, Qwen3-4B(-Base), Qwen3-8B-Base, Qwen3-14B-Base, gemma-2-9b-it, gemma-3-4b-it, Llama-3.1-8B(-Instruct), Llama-3.1-70B(-Instruct), gpt-oss-120b, OLMo-2/3; listed but not loaded ("COLD"): Qwen3-1.7B, gemma-2-2b, gemma-3-1b-pt, Llama-3.2-1B, Llama-3.1-405B, … ⚠ Search snippets (not verified) say access requires US affiliation via CILogon and remote gradients are unsupported.

#### 2.4 circuit-tracer + transcoders ✅
- Hanna, Piotrowski, Lindsey, Ameisen, [BlackboxNLP 2025](https://aclanthology.org/2025.blackboxnlp-1.14/); moved from `safety-research` to `decoderesearch` (old URL redirects). Workflow: `circuit-tracer attribute` → prune (node threshold 0.8, edge threshold 0.98, ≤ 7,500 feature nodes) → feature interventions.
- Backends: TransformerLens (default; `ReplacementModel` subclasses `HookedTransformer`, no Bridge support) and nnsight (experimental, slower, broader HF coverage).
- Supported transcoder sets (README): Gemma-2-2B PLTs `mntss/gemma-scope-transcoders`, CLTs `mntss/clt-gemma-2-2b-426k` / `-2.5M`; Llama-3.2-1B `mntss/transcoder-Llama-3.2-1B` and `mntss/clt-llama-3.2-1b-524k`; Qwen3 PLTs `mwhanna/qwen3-{0.6b,1.7b,14b}-transcoders-lowl0`, `mwhanna/qwen3-{4b,8b}-transcoders`; Gemma-3 270M–27B PLTs (collection `mwhanna/gemma-scope-2-transcoders-circuit-tracer`, **nnsight backend only**); GPT-OSS-20B CLT `mntss/clt-131k`; Llama-3.1-8B-Instruct TopK PLTs `facebook/crv-8b-instruct-transcoders`.
- Demo notebooks cover two-hop reasoning, multilingual, simple addition swaps, garden-path sentences and analogies — no couplet/rhyme or 36 + 59 demo.

#### 2.5 Neuronpedia API ✅
- OpenAPI spec at [neuronpedia.org/api-doc](https://www.neuronpedia.org/api-doc); auth via `x-api-key` header (some endpoints keyless). Endpoint groups: features (`GET /api/feature/{modelId}/{layer}/{index}`), activations, search (`/api/search-all`, `/api/search-topk-by-token`), auto-interp (`/api/explanation/{generate,score,export}`), steering (`/api/steer`, `/api/steer-chat`; [240 requests/hour](https://docs.neuronpedia.org/steering)), attribution graphs (`POST /api/graph/generate` with prompt, modelId, sourceSetName, thresholds), Jacobian lens (`/api/lens/{prompt,oracle,share}`), NLA, vectors, lists.
- Self-hostable services: webapp, inference, autointerp, graph, nla, sparsity (Postgres 16 + pgvector).

#### 2.6 Lenses ✅
- **jacobian-lens** ([repo](https://github.com/anthropics/jacobian-lens), Apache-2.0; reference code, not maintained): lens_ℓ(h) = unembed(J_ℓ h) with J_ℓ = E[∂h_final/∂h_ℓ]. **Name collision:** PyPI `jlens` is an unrelated JSONata library.
- Pre-fitted lenses: [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens) (licence "other": per-folder base-model terms) for GPT-2 small, Pythia-70M-deduped, Qwen3 1.7B–32B, Gemma-2 2B/9B/27B, Gemma-3 270M–27B, Llama-3.1-8B and others, plus J++ and R-lens folders. J++: [safety-research/jpp_lens](https://github.com/safety-research/jpp_lens), [koayon/jpp-lenses](https://huggingface.co/koayon/jpp-lenses) (Apache-2.0).
- **tuned-lens** ([repo](https://github.com/AlignmentResearch/tuned-lens), MIT; PyPI 0.2.0 from 2023-07, last push 2025-08) — older but still the standard learned affine lens.

#### 2.7 Other tools (verified one-liners)

| Tool | Version / status | Licence | Use |
|---|---|---|---|
| [pyvene](https://github.com/stanfordnlp/pyvene) | 0.1.8 (2025-05) | Apache-2.0 | interchange interventions, DAS |
| [pyreft](https://github.com/stanfordnlp/pyreft) | 0.1.0 (2025-02) | Apache-2.0 | ReFT steering baselines (AxBench) |
| [Inseq](https://github.com/inseq-team/inseq) | 0.7.1 (2026-03) | Apache-2.0 | input attribution |
| [dictionary_learning](https://github.com/saprmarks/dictionary_learning) | PyPI 0.1.0; repo active (2026-11) | MIT | SAE training (Marks) |
| [delphi](https://github.com/EleutherAI/delphi) (`eai-delphi`) | 0.2.1 (2026-11-26) | Apache-2.0 | automated feature interpretation (PyPI `delphi` is unrelated) |
| [sparsify](https://github.com/EleutherAI/sparsify) (`eai-sparsify`) | 1.3.3 (2026-07) | MIT | TopK SAE / transcoder training |
| [CLT-Forge](https://github.com/LLM-Interp/CLT-Forge) | [2603.21014](https://arxiv.org/abs/2603.21014) | MIT | distributed CLT training |
| [EAP-IG](https://github.com/hannamw/EAP-IG) | active (2026-05) | MIT | edge attribution patching on TransformerLens |
| [ACDC](https://github.com/ArthurConmy/Automatic-Circuit-Discovery) | last push 2024-10 | MIT | automatic circuit discovery |
| [CircuitsVis](https://github.com/TransformerLensOrg/CircuitsVis) | last push 2026-04 | MIT | attention/activation widgets |
| [Penzai](https://github.com/google-deepmind/penzai) | **archived** (GitHub API) | Apache-2.0 | JAX model surgery (not recommended) |
| Garçon (Anthropic) | internal, closed | — | not available |

## 3. Open models with released interpretability artefacts

Licences are from HF metadata ("no tag" = no licence field, not necessarily unlicensed). The base model's own licence (e.g. Gemma Terms of Use, Llama Community Licence) also applies to derived artefacts; base-model licences were not re-checked for this review.

| Model | SAEs | Per-layer transcoders | CLTs | Lenses | Other | NDIF |
|---|---|---|---|---|---|---|
| **GPT-2 small** | `jbloom/GPT2-Small-SAEs-Reformatted` (res-jb, MIT); OpenAI v5 32k/128k via `jbloom/GPT2-Small-OAI-v5-*` (MIT); [openai/sparse_autoencoder](https://github.com/openai/sparse_autoencoder) (MIT) | EleutherAI `gpt2-mntss-transcoder-relu-sp6-skip` (no tag) | EleutherAI `gpt2-mntss-transcoder-clt-relu-sp8`, `-sp10-1b` (no tag) | J-lens (neuronpedia) | IOI, greater-than, copy-suppression code; MIB tasks | hot |
| **attn-only-1l…4l** (`NeelNanda/Attn_Only_*L512W_C4_Code`) | 1L SAE replication (Nanda) | — | — | — | Mathematical Framework / docstring substrates (no licence tag) | — |
| **Pythia-70M/160M/410M (deduped)** | `pythia-70m-deduped-res-sm`; EleutherAI `sae-pythia-70m-32k`, `-160m-32k` (MIT), deduped / 410M variants (no tag), `Pythia-160m-SAE-k*` | — | — | J-lens (70M-deduped) | training checkpoints for formation studies | 70M, 160M, 2.8B hot |
| **Gemma-2-2B** | [Gemma Scope](https://huggingface.co/google/gemma-scope) res/mlp/att, 16K–1M (CC-BY-4.0) | `google/gemma-scope-2b-pt-transcoders` (CC-BY-4.0) = `mntss/gemma-scope-transcoders` | `mntss/clt-gemma-2-2b-426k`, `-2.5M` (no tag) | J-lens | Neuronpedia in-browser graphs; MIB model; AxBench, absorption, Ferrando entity latents | cold |
| **Gemma-3 270M–27B (PT/IT)** | [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2) (CC-BY-4.0) | Gemma Scope 2 transcoders / skip-transcoders | Gemma Scope 2 CLTs (270M and 1B per HF metadata) | J-lens | NLAs for Gemma-3-12B-IT (L32) and 27B-IT (L41) | gemma-3-4b-it hot |
| **Llama-3.2-1B** | — | `mntss/transcoder-Llama-3.2-1B` (MIT) | `mntss/clt-llama-3.2-1b-524k` (no tag) | — | Jacopin planning studies | cold |
| **Llama-3.1-8B** | [Llama Scope](https://huggingface.co/OpenMOSS-Team/Llama-Scope) 256 SAEs, 32K/128K (Apache-2.0; base model; [2410.20526](https://arxiv.org/abs/2410.20526)); Arditi BatchTopK SAEs via Neuronpedia | `facebook/crv-8b-instruct-transcoders` (Instruct, TopK) | — | J-lens | Goodfire manifold papers; Arora neuron circuits; MIB model | hot |
| **Qwen3-0.6B / 1.7B / 4B / 8B / 14B** | SAELens Qwen3 releases; Neuronpedia Qwen3 SAEs | `mwhanna/qwen3-{0.6b,1.7b,14b}-transcoders-lowl0`, `qwen3-{4b,8b}-transcoders` (MIT) | `bluelightai/clt-qwen3-{0.6b,1.7b}-base-20k` (Apache-2.0, **Base** models) | J-lens 1.7B–32B | Neuronpedia OpenMOSS replacement-model graphs on Qwen3-1.7B (⚠ not re-checked); Hanna & Ameisen planning study | 1.7B-Base hot, 1.7B cold |
| **Qwen2.5-7B-Instruct** | — | — | — | — | NLA (L20); *When Models Don't Manipulate Manifolds* | — |
| **GPT-OSS-20B** | — | — | `mntss/clt-131k` (no tag) | — | — | gpt-oss-120b hot |
| **Qwen3.6-27B, Qwen3.5-9B, Gemma-4-31B, OLMo-3-32B, …** | — | — | — | J++ lenses (Apache-2.0) | Goodfire neuron atlas (Qwen3.6-35B-A3B) | — |
| **Llama-3.3-70B-Instruct** | — | — | — | J-lens (Viswanath, trained with Anthropic code) | NLA (L53) | — |

**Implication for Qwen3-1.7B as an atlas model:** it is one of the few small models with per-layer transcoders, a CLT (for the Base checkpoint), J-lenses, Neuronpedia graphs and NDIF hosting at once — but the transcoders and the CLT target different checkpoints, so pick one and keep it fixed.

## 4. Best current methods by mechanism type

| Object | Current best practice (evidence) | Known failure modes |
|---|---|---|
| Circuits in small models | Path / activation patching for hypothesis tests ([Wang et al.](https://arxiv.org/abs/2211.00593)); automated discovery with EAP-IG or mask optimisation, which lead MIB's circuit-localisation track ([MIB](https://arxiv.org/abs/2504.13151)); ACDC re-discovers greater-than and docstring ([2304.14997](https://arxiv.org/abs/2304.14997)) | faithfulness depends on ablation type, positions and metric ([Miller et al.](https://arxiv.org/abs/2407.08734)); specific heads change over training while the algorithm persists ([Tigges et al.](https://arxiv.org/abs/2407.10827)) |
| Attribution graphs | CLTs + local replacement model with error nodes ([Ameisen et al.](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)); circuit-tracer on Gemma-2-2B / Llama-3.2-1B / Qwen3 | conclusions can change by orders of magnitude across transcoders ([Jacopin 2609.18440](https://arxiv.org/abs/2609.18440)); transcoders shatter geometric features into lookup tables ([landscape post](https://www.neuronpedia.org/graph/info)); attention is frozen, so QK-mediated computation is invisible |
| Feature dictionaries | JumpReLU / BatchTopK / Matryoshka SAEs (SAELens); transcoders and CLTs for circuits | absorption ([Chanin et al.](https://arxiv.org/abs/2409.14507)); manifolds split into many local features ([Bhalla et al.](https://arxiv.org/abs/2604.28119)); SAE steering loses to prompting ([AxBench](https://arxiv.org/abs/2501.17148)) |
| Neurons | revived as a basis: RelP attribution gives sparse neuron circuits ([Arora et al.](https://arxiv.org/abs/2601.22594)); J-lens readouts of neuron output vectors ([Shone & Dunlap](https://alignment.anthropic.com/2026/neuron-basis-theory/)) | evidence so far is on ≥ 8B models and in-distribution data ([critique](https://www.lesswrong.com/posts/EMKRpxKswPZEbP8ey/datasets-determine-the-difficulty-of-interpretability)) |
| Geometry / manifolds | PCA + spline manifolds and manifold steering ([Wurgaft et al.](https://arxiv.org/abs/2605.05115)); subspace patching ([Engels et al.](https://arxiv.org/abs/2405.14860)); block-sparse featurizers in vision ([Fel et al.](https://arxiv.org/abs/2606.25234)) | smooth late-layer behaviour does not imply a continuous variable ([Gupta et al.](https://arxiv.org/abs/2610.37796)); whitening in high dimension produces orthogonality for random concepts ([Golechha et al.](https://github.com/iclr-blogposts/2025/blob/main/_posts/2025-04-28-feature-geometry.md)); curved geometry can coexist with linear computation ([Hindupur et al.](https://arxiv.org/abs/2609.37680)) |
| Steering directions | difference-of-means is the strongest simple direction ([Im & Li](https://arxiv.org/abs/2502.02716); [AxBench](https://arxiv.org/abs/2501.17148)); directional ablation / weight orthogonalisation for refusal ([Arditi et al.](https://arxiv.org/abs/2406.11717)) | prompting still beats activation steering ([SteerScope](https://arxiv.org/abs/2610.07722)); steering reverses for many inputs ([Tan et al.](https://arxiv.org/abs/2407.12404)); refusal is multi-dimensional ([Wollschläger et al.](https://arxiv.org/abs/2502.17420)) and may not transfer across languages ([Zhang, MRL 2026](https://aclanthology.org/2026.mrl-1.19/)) |
| Lenses / readouts | J++ lens beats J-lens and R-lens on recall@10 ([Ayonrinde & Lindsey](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace)); NLAs for free-text readouts ([Fraser-Taliente et al.](https://transformer-circuits.pub/2026/nla/index.html)) | workspace tests pass trivially on GPT-2 ([Cairns et al.](https://www.lesswrong.com/posts/GeKveaNdruATcj2Fj/even-gpt-2-passes-most-functional-global-workspace-tests)); selectivity not reproduced below 9B ([avalur/JLens](https://github.com/avalur/JLens)) |

## 5. Open problems

**Field-level agenda** (Sharkey et al., *Open Problems in Mechanistic Interpretability*, [2501.16496](https://arxiv.org/abs/2501.16496), Jan 2025; anchors from the arXiv HTML):
- **Methods and foundations (§2):** reverse engineering (decomposition §2.1.2, describing functional roles §2.1.3, validation §2.1.4); concept-based interpretability, where probes find correlations rather than causal variables (§2.2); circuit-discovery pipelines (§2.3); automation (§2.4).
- **Applications (§3):** monitoring and auditing (§3.2), prediction (§3.3), improving inference and training (§3.4), "microscope AI" (§3.5), a broader range of models (§3.6), human–computer interaction (§3.7).
- **Socio-technical (§4):** policy and governance (§4.1), social and philosophical questions (§4.2).

Later agenda posts: GDM's [*A Pragmatic Vision for Interpretability*](https://www.lesswrong.com/posts/StENzDcD3kpfGJssR) (Nanda et al., 1 Dec 2025) and [*How Can Interpretability Researchers Help AGI Go Well?*](https://www.lesswrong.com/posts/MnkeepcGirnJn736j) (model biology, monitoring, model diffing, proxy tasks); Barez et al., *Understanding AI from the Inside* ([2611.08438](https://arxiv.org/abs/2611.08438), Nov 2026 survey; ◐ not read beyond the listing).

**Open problems this review surfaced that an atlas can make visible** (each backed by the cited entry in the literature review):
1. **Do closed-model findings transfer to open models?** Dallas → Austin does; newline rhyme planning, 36 + 59 addition and the J-lens workspace's selectivity have not been reproduced on small open models (literature review §2.3, §8.3).
2. **Dictionary dependence.** Attribution-graph conclusions vary strongly with the transcoder (§2.5); SAEs fragment manifolds (§4.5) and absorb features (§6.4). A finding should be checked across at least two dictionaries or against a dictionary-free method (patching, neurons).
3. **Faithfulness metrics are not standardised.** IOI faithfulness moves with ablation choices (§1.3); the docstring circuit "recovers half" or "75%" depending on the baseline (§1.7).
4. **Correlational vs causal geometry.** Many manifold results are probe/PCA-only (§3.5–3.7, §3.10, T-Tech replication of §3.3); static-embedding baselines already explain part of space/time and emotion geometry (§3.7, §7b.2).
5. **Discrete vs continuous representations.** Weekdays look like a ring under late-layer steering and like seven spokes under early-layer steering (§4.2–4.3).
6. **Neurons vs dictionaries.** 2026 work argues neurons are already a usable basis (§5.3–5.6) against the superposition picture (§3.4); evaluation on out-of-distribution data is the open test.
7. **Steering reliability.** Prompting beats representation steering on standard benchmarks (§6.5, §7.7); emotion-steering effects on open models are small or reversed (§7b.2).
8. **Introspection validity.** Open replications range from success to null results with confidence intervals spanning zero (§8.4).

## 6. Practical recommendations for the atlas (derived from §2–§5)

1. **Pin `transformer-lens==3.9.*`** if circuit-tracer (TransformerLens backend) and SAELens must share one environment; write new hook code against `TransformerBridge` so a later move to 4.x only affects circuit-tracer.
2. **Install the Jacobian lens from GitHub** (`anthropics/jacobian-lens`) or use TL 3.9's `jacobian_lens` module; never `pip install jlens`.
3. **Recompute on models with complete artefacts:** GPT-2 small (classic circuits, SAEs, J-lens), Pythia (training dynamics), Gemma-2-2B (Gemma Scope + CLTs + Neuronpedia graphs), Qwen3-1.7B (transcoders, CLT for Base, J-lens). Use Llama-3.1-8B only where the finding does not exist at smaller scale (manifold steering, neuron circuits).
4. **Redraw, don't copy, closed-model figures.** No licence statement was found on the transformer-circuits.pub pages checked; the arXiv copies of *Scaling Monosemanticity* ([2605.29358](https://arxiv.org/abs/2605.29358)) and *When Models Manipulate Manifolds* ([2601.04480](https://arxiv.org/abs/2601.04480)) are CC BY 4.0, and Distill articles are CC-BY 4.0. Redraw from reported numbers and cite the section/figure.
5. **Record evidence type and dictionary per entry** (causal vs correlational; which SAE/transcoder), because §4 shows conclusions change with both.
6. **Use MIB tasks** ([2504.13151](https://arxiv.org/abs/2504.13151)) where an entry overlaps (IOI, arithmetic), so recomputed circuits are comparable with published method rankings.

## 7. Corrections to the task brief (tooling and prior art)

- **TransformerLens 4.x removes `HookedTransformer`** (4.0.0, 2026-09-21); 3.x is the bug-fix-only line for it. circuit-tracer 0.5.5 pins `transformer-lens<4`, so "3.x vs 4.x" is a hard compatibility decision, not a style choice.
- **circuit-tracer moved** from `safety-research/circuit-tracer` to `decoderesearch/circuit-tracer` (redirect in place). Its demos contain **no rhyme/poetry and no 36 + 59 replication**; of the four Biology case studies only Dallas → Austin and multilingual are replicated on Gemma-2-2B (details in the literature review §2.3).
- **jacobian-lens is not on PyPI**, and the PyPI name `jlens` belongs to an unrelated package.
- **Gemma Scope 2 (Dec 2025) already ships CLTs** (for Gemma-3 270M and 1B per HF metadata), contrary to the claim in Jacopin 2604.15010 that mntss CLTs are the only public CLTs.
- **OpenAI Microscope is still online** (HTTP 200, last modified 2026-11-07); no shutdown notice was found.
- **Goodfire Neuron Atlas** covers a single model, Qwen3.6-35B-A3B (10 Nov 2026).

## 8. Items not verified

- NDIF access policy (US affiliation via CILogon; no remote gradients) — search snippets only.
- Gemma Scope 2 technical report PDF — not opened; coverage taken from the DeepMind blog and HF metadata.
- OpenAI Microscope model list — JavaScript-rendered, not readable by the fetch tools.
- Neel Nanda's "200 Concrete Open Problems" — not fetched.
- Barez et al. 2611.08438 — listing only.
- `decoderesearch/interp-engine` licence (used by Neuronpedia) — not checked.
- Gemma Scope tutorials — not fetched.
