// Architecture specs for the Model Explorer. Numbers come from each model's config/paper (sources inline); the
// explorer draws them and overlays the `locations` declared by atlas entries.
// Global context: entries say *where* a finding lives via frontmatter `locations: [{model, part, layer, head, ...}]`;
// ModelExplorer matches those against the parts drawn from these specs.

export type ModelKind = 'transformer' | 'cnn' | 'pipeline'

export interface CnnLayer {
  name: string
  channels: number
}

export interface ModelSpec {
  id: string
  name: string
  kind: ModelKind
  open: boolean
  params: string
  /** Transformer: number of blocks, heads per block, attention-only flag. */
  layers?: number
  heads?: number
  attnOnly?: boolean
  dModel?: number
  /** Closed models: layer count unknown — the explorer draws a relative-depth axis (0–100%). */
  relativeDepth?: boolean
  /** CNN / pipeline stages. */
  stages?: CnnLayer[]
  /** 'llm' ⊂ 'transformer'; used by the General LLM / General transformer aggregate views. */
  family?: 'llm' | 'transformer' | 'vision' | 'other'
  /** Aggregate view: findings of these member families are placed on one generic block by relative depth. */
  aggregate?: ('llm' | 'transformer')[]
  blurb: string
  source: string
}

export const MODELS: ModelSpec[] = [
  {
    id: 'general-llm',
    name: 'General LLM (all language models)',
    kind: 'transformer',
    open: true,
    params: 'aggregate view',
    relativeDepth: true,
    aggregate: ['llm'],
    blurb:
      'Every finding from every language model in the atlas (GPT-2, Pythia, Qwen3, Llama, Claude…) placed on one generic decoder-only transformer by relative depth: where in a typical LLM do circuits and manifolds live?',
    source: 'https://transformer-circuits.pub/2021/framework/index.html',
  },
  {
    id: 'general-transformer',
    name: 'General transformer (all transformer models)',
    kind: 'transformer',
    open: true,
    params: 'aggregate view',
    relativeDepth: true,
    aggregate: ['llm', 'transformer'],
    blurb:
      'Language models plus transformers trained on other tasks (Othello moves, modular arithmetic), all mapped onto one generic transformer block stack by relative depth.',
    source: 'https://arxiv.org/abs/1706.03762',
  },
  {
    id: 'gpt2-small',
    family: 'llm',
    name: 'GPT-2 small',
    kind: 'transformer',
    open: true,
    params: '124M',
    layers: 12,
    heads: 12,
    dModel: 768,
    blurb: 'The workhorse of circuit discovery: IOI, copy suppression, greater-than, induction and calendar circles all live here.',
    source: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
  },
  {
    id: 'attn-only-2l',
    family: 'llm',
    name: 'Attention-only 2-layer',
    kind: 'transformer',
    open: true,
    params: '~4M (TransformerLens)',
    layers: 2,
    heads: 8,
    attnOnly: true,
    dModel: 512,
    blurb: 'The smallest model with induction heads: a previous-token head in layer 0 feeds an induction head in layer 1.',
    source: 'https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html',
  },
  {
    id: 'pythia-160m',
    family: 'llm',
    name: 'Pythia-160M',
    kind: 'transformer',
    open: true,
    params: '160M (+ training checkpoints)',
    layers: 12,
    heads: 12,
    dModel: 768,
    blurb: 'Released with 154 training checkpoints, so you can watch circuits form (induction phase change, number geometry).',
    source: 'https://arxiv.org/abs/2304.01373',
  },
  {
    id: 'modadd-1l',
    family: 'transformer',
    name: '1-layer modular-addition transformer',
    kind: 'transformer',
    open: true,
    params: '~0.2M (trained here)',
    layers: 1,
    heads: 4,
    dModel: 128,
    blurb: 'Trained in this repo on (a + b) mod 113: it groks by placing numbers on circles (embedding) and multiplying waves (MLP).',
    source: 'https://arxiv.org/abs/2301.05217',
  },
  {
    id: 'othello-gpt',
    family: 'transformer',
    name: 'Othello-GPT',
    kind: 'transformer',
    open: true,
    params: '25M',
    layers: 8,
    heads: 8,
    dModel: 512,
    blurb: 'Trained only on move sequences, yet its residual stream holds a linear map of the board (mine vs. theirs).',
    source: 'https://arxiv.org/abs/2210.13382',
  },
  {
    id: 'mess3-4l',
    family: 'transformer',
    name: '4-layer Mess3 transformer',
    kind: 'transformer',
    open: true,
    params: '~0.2M (trained here)',
    layers: 4,
    heads: 1,
    dModel: 64,
    blurb: 'Trained in this repo to predict tokens of a 3-state hidden Markov process; its residual stream holds the Bayesian belief fractal.',
    source: 'https://arxiv.org/abs/2405.15943',
  },
  {
    id: 'qwen3-1.7b',
    family: 'llm',
    name: 'Qwen3-1.7B',
    kind: 'transformer',
    open: true,
    params: '1.7B',
    layers: 28,
    heads: 16,
    dModel: 2048,
    blurb: 'Our main modern open model: counting manifolds, emotion vectors, J-lens workspace, neuron sparsity, attribution graphs.',
    source: 'https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json',
  },
  {
    id: 'llama-3.1-8b',
    family: 'llm',
    name: 'Llama-3.1-8B',
    kind: 'transformer',
    open: true,
    params: '8B',
    layers: 32,
    heads: 32,
    dModel: 4096,
    blurb: "Goodfire's Neural Geometry series studies it: a base-10 calculator in layer 18 and weekday manifolds you can steer along.",
    source: 'https://arxiv.org/abs/2407.21783',
  },
  {
    id: 'claude-3.5-haiku',
    family: 'llm',
    name: 'Claude 3.5 Haiku (closed)',
    kind: 'transformer',
    open: false,
    params: 'undisclosed',
    relativeDepth: true,
    blurb: "Anthropic's attribution-graph 'biology' and the linebreak counting manifolds. Depth shown as a fraction of the model.",
    source: 'https://transformer-circuits.pub/2025/attribution-graphs/biology.html',
  },
  {
    id: 'claude-sonnet-4.5',
    family: 'llm',
    name: 'Claude Sonnet 4.5 (closed)',
    kind: 'transformer',
    open: false,
    params: 'undisclosed',
    relativeDepth: true,
    blurb: 'Emotion concepts (~⅔ depth) and the verbalizable "global workspace" band found with the Jacobian lens.',
    source: 'https://transformer-circuits.pub/2026/workspace/index.html',
  },
  {
    id: 'inceptionv1',
    family: 'vision',
    name: 'InceptionV1 (GoogLeNet)',
    kind: 'cnn',
    open: true,
    params: '7M',
    // Lucid layer names and output channel counts (Szegedy et al. 2015, Table 1; Distill Circuits naming).
    stages: [
      { name: 'conv2d0', channels: 64 },
      { name: 'conv2d1', channels: 64 },
      { name: 'conv2d2', channels: 192 },
      { name: 'mixed3a', channels: 256 },
      { name: 'mixed3b', channels: 480 },
      { name: 'mixed4a', channels: 508 },
      { name: 'mixed4b', channels: 512 },
      { name: 'mixed4c', channels: 512 },
      { name: 'mixed4d', channels: 528 },
      { name: 'mixed4e', channels: 832 },
      { name: 'mixed5a', channels: 832 },
      { name: 'mixed5b', channels: 1024 },
    ],
    blurb: 'The vision model of the Distill Circuits thread: curve detectors, high-low frequency detectors, curve circuits.',
    source: 'https://arxiv.org/abs/1409.4842',
  },
  {
    id: 'flux-vae',
    family: 'vision',
    name: 'FLUX VAE',
    kind: 'pipeline',
    open: true,
    params: 'see paper',
    stages: [
      { name: 'image', channels: 3 },
      { name: 'encoder', channels: 512 },
      { name: 'latent', channels: 16 },
      { name: 'decoder', channels: 512 },
      { name: 'image out', channels: 3 },
    ],
    blurb: 'The autoencoder under FLUX image generation: colors live on a 3-D hue–saturation–lightness bicone in its latent.',
    source: 'https://arxiv.org/abs/2603.12261',
  },
  {
    id: 'mountain-car-wm',
    family: 'other',
    name: 'Mountain-car world model',
    kind: 'pipeline',
    open: true,
    params: 'small (trained here)',
    stages: [
      { name: 'frame', channels: 3 },
      { name: 'conv encoder', channels: 64 },
      { name: 'latent', channels: 64 },
      { name: 'GRU', channels: 64 },
      { name: 'conv decoder', channels: 64 },
      { name: 'next frame', channels: 3 },
    ],
    blurb: "Goodfire's toy world model: frame embeddings trace a 1-D curve indexed by the car's position.",
    source: 'https://arxiv.org/abs/2605.05115',
  },
  {
    id: 'evo2',
    family: 'other',
    name: 'Evo 2 (DNA model, StripedHyena 2)',
    kind: 'transformer',
    open: true,
    params: '1B / 7B / 40B variants',
    layers: 32,
    heads: 0,
    blurb: 'A genome model built on StripedHyena 2 (convolution + attention hybrid, not a pure transformer) whose layer-24 activations encode the bacterial tree of life as distances on a curved manifold.',
    source: 'https://www.goodfire.com/research/phylogeny-manifold',
  },
]

export const modelById: Record<string, ModelSpec> = Object.fromEntries(MODELS.map((m) => [m.id, m]))
