// Methods glossary: the techniques used across the atlas, in plain words + one-line technical definitions + sources.
const METHODS: { name: string; plain: string; tech: string; src: [string, string] }[] = [
  {
    name: 'Attention pattern',
    plain: 'Which earlier tokens a head is "looking at" when processing the current token.',
    tech: 'softmax(QKᵀ/√d) for one head; rows = destination positions, columns = source positions.',
    src: ['Elhage et al. 2021', 'https://transformer-circuits.pub/2021/framework/index.html'],
  },
  {
    name: 'Ablation (zero / mean)',
    plain: 'Switch a component off and see what breaks.',
    tech: 'Replace a component’s output with 0 or with its mean over a reference distribution, then measure the metric.',
    src: ['Wang et al. 2022, §4', 'https://arxiv.org/abs/2211.00593'],
  },
  {
    name: 'Activation patching',
    plain: 'Swap one component’s activation between two inputs that differ in one detail; if the output flips, that component carries the detail.',
    tech: 'Run clean and corrupted inputs, overwrite a hook with the other run’s value, measure Δ metric (noising or denoising).',
    src: ['Heimersheim & Nanda 2024', 'https://arxiv.org/abs/2404.15255'],
  },
  {
    name: 'Path patching',
    plain: 'Patching restricted to one route through the network (e.g. head A → head B only).',
    tech: 'Patch the sender’s output only along the edge into the chosen receiver, freezing all other paths.',
    src: ['Wang et al. 2022', 'https://arxiv.org/abs/2211.00593'],
  },
  {
    name: 'Direct logit attribution',
    plain: 'How much a component directly pushes a particular answer up or down.',
    tech: 'Project the component’s residual-stream output (÷ final LayerNorm scale) onto W_U[answer] − W_U[alternative].',
    src: ['Elhage et al. 2021', 'https://transformer-circuits.pub/2021/framework/index.html'],
  },
  {
    name: 'Linear probe',
    plain: 'A simple classifier that tries to read a property (e.g. board state) from activations. Correlational unless paired with an intervention.',
    tech: 'Logistic/ridge regression from a layer’s activations to labels, evaluated on held-out data.',
    src: ['Alain & Bengio 2016', 'https://arxiv.org/abs/1610.01644'],
  },
  {
    name: 'Fourier analysis',
    plain: 'Express a representation as a sum of waves to find periodic (circular) structure.',
    tech: 'Change of basis to cos/sin(2πkx/p); energy per frequency k.',
    src: ['Nanda et al. 2023', 'https://arxiv.org/abs/2301.05217'],
  },
  {
    name: 'Sparse autoencoder (SAE)',
    plain: 'Learns a large dictionary of directions so each activation is a sparse combination of interpretable "features".',
    tech: 'x ≈ W_dec · f(W_enc x + b), f sparse (ReLU+L1, TopK, JumpReLU).',
    src: ['Bricken et al. 2023', 'https://transformer-circuits.pub/2023/monosemantic-features'],
  },
  {
    name: 'Transcoder / attribution graph',
    plain: 'Replace MLPs with sparse, interpretable features and draw the graph of which features cause which on one prompt.',
    tech: 'Cross-layer transcoder replacement model with frozen attention; edges = activation × virtual weight; pruned graph.',
    src: ['Ameisen et al. 2025', 'https://transformer-circuits.pub/2025/attribution-graphs/methods.html'],
  },
  {
    name: 'Steering vector',
    plain: 'Add a direction to the activations to push behavior (e.g. more positive, more refusing).',
    tech: 'h ← h + α·v, v often a difference of means between contrastive prompts.',
    src: ['Turner et al. 2023', 'https://arxiv.org/abs/2308.10248'],
  },
  {
    name: 'Logit lens / Jacobian lens',
    plain: 'Read what an intermediate layer "would say" if it were the last layer.',
    tech: 'Logit lens: softmax(W_U · norm(h_ℓ)). J-lens: softmax(W_U · norm(J_ℓ h_ℓ)) with J_ℓ the mean Jacobian to a late layer.',
    src: ['Gurnee et al. 2026', 'https://arxiv.org/abs/2607.15495'],
  },
  {
    name: 'Manifold (representation geometry)',
    plain: 'Related values (days, numbers, colors, positions) lie on a smooth curved shape in activation space, not on a straight line.',
    tech: 'Low-dimensional curve/surface fitted through class means (PCA, splines); tested by steering along vs across it.',
    src: ['Engels et al. 2024', 'https://arxiv.org/abs/2405.14860'],
  },
]

export default function Methods() {
  return (
    <>
      <h1>Methods glossary</h1>
      <p className="prose">The techniques that appear across the atlas. Causal methods intervene on the model; correlational ones only read it.</p>
      <div className="grid">
        {METHODS.map((m) => (
          <div className="panel" key={m.name} style={{ margin: 0 }}>
            <h2>{m.name}</h2>
            <p>{m.plain}</p>
            <p className="viz-note">
              <b>Technical:</b> {m.tech}
            </p>
            <p className="viz-note">
              Source:{' '}
              <a href={m.src[1]} target="_blank" rel="noopener">
                {m.src[0]}
              </a>
            </p>
          </div>
        ))}
      </div>
    </>
  )
}
